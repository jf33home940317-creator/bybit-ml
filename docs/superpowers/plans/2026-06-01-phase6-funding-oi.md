# Phase 6: Funding Rate + Open Interest Features — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add 6 Funding Rate / Open Interest features to the ML pipeline, retrain models, and deploy only if Sharpe improves.

**Architecture:** New data fetchers pull FR/OI history from Bybit V5 API with while-loop pagination. Features are computed in `indicators.py` via `merge_asof` (FR) and timestamp join (OI). Models retrain automatically — the validation report drives which features the model expects. A Sharpe comparison gate decides deploy vs rollback.

**Tech Stack:** Bybit V5 REST API, pandas merge_asof, XGBoost, existing pytest infrastructure.

---

## File Map

| Action | File | Responsibility |
|---|---|---|
| Modify | `data/fetcher.py` | Add `fetch_funding_rate()` and `fetch_open_interest()` |
| Create | `tests/test_fetcher_fr_oi.py` | Tests for new fetcher functions |
| Modify | `main.py` | Call new fetchers in data pipeline |
| Modify | `features/indicators.py` | Add `_attach_funding_features()` and `_attach_oi_features()` |
| Create | `tests/test_indicators_fr_oi.py` | Tests for new indicator functions |
| Modify | `features/builder.py` | Load FR/OI parquets, pass to `indicators.compute()` |
| Modify | `live/fetcher.py` | Add `fetch_funding_rate_latest()` and `fetch_oi_latest()` |
| Create | `tests/test_live_fetcher_fr_oi.py` | Tests for live FR/OI fetchers |
| Modify | `live/pipeline.py` | Fetch FR/OI in `compute_signal()`, pass to `indicators.compute()` |
| Create | `compare_models.py` | Sharpe comparison + auto-rollback script |

---

## Task 1: Historical Funding Rate Fetcher

**Files:**
- Modify: `data/fetcher.py`
- Create: `tests/test_fetcher_fr_oi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_fetcher_fr_oi.py
import pytest
from unittest.mock import MagicMock, patch
from datetime import timezone


def _make_fr_rows(base_ms: int, count: int, interval_ms: int = 28_800_000):
    """Build count funding rate records, newest first (Bybit order)."""
    rows = []
    for i in range(count - 1, -1, -1):
        ts = base_ms + i * interval_ms
        rows.append({
            "symbol": "ETHUSDT",
            "fundingRate": "0.0001",
            "fundingRateTimestamp": str(ts),
        })
    return rows


BASE_MS = 1_640_995_200_000  # 2022-01-01 00:00 UTC


class TestFetchFundingRate:

    def test_returns_timestamp_and_funding_rate_columns(self):
        from data.fetcher import fetch_funding_rate
        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("ETHUSDT", "2022-01-01", "2022-01-02")
        assert "timestamp" in df.columns
        assert "funding_rate" in df.columns
        assert len(df) == 3

    def test_funding_rate_is_float(self):
        from data.fetcher import fetch_funding_rate
        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 2)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("ETHUSDT", "2022-01-01", "2022-01-02")
        assert df["funding_rate"].dtype == "float64"

    def test_chronological_order(self):
        from data.fetcher import fetch_funding_rate
        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("ETHUSDT", "2022-01-01", "2022-01-02")
        assert list(df["timestamp"]) == sorted(df["timestamp"])

    def test_timestamp_is_utc(self):
        from data.fetcher import fetch_funding_rate
        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 1)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("ETHUSDT", "2022-01-01", "2022-01-02")
        assert str(df["timestamp"].dt.tz) == "UTC"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher_fr_oi.py -q`
Expected: FAIL — `ImportError: cannot import name 'fetch_funding_rate'`

- [ ] **Step 3: Write minimal implementation**

Add to `data/fetcher.py` after `fetch_ohlcv`:

```python
def fetch_funding_rate(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch historical funding rates from Bybit V5 (linear perpetual).

    Returns chronological DataFrame with columns: [timestamp, funding_rate].
    Funding rates are published every 8 hours.
    """
    session = HTTP(testnet=False, api_key=config.API_KEY, api_secret=config.API_SECRET)
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches = []
    current_end = end_dt
    batch_count = 0

    while current_end > start_dt:
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = session.get_funding_rate_history(
                    category="linear",
                    symbol=symbol,
                    endTime=_to_ms(current_end),
                    limit=200,
                )
                break
            except Exception as exc:
                if attempt == config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        rows = resp["result"]["list"]
        if not rows:
            break

        batch_df = pd.DataFrame(rows)
        batch_df["timestamp"] = pd.to_datetime(
            batch_df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True
        )
        batch_df["funding_rate"] = batch_df["fundingRate"].astype(float)
        batch_df = batch_df[["timestamp", "funding_rate"]]
        batch_df = batch_df[batch_df["timestamp"] >= start_dt]

        if batch_df.empty:
            break

        all_batches.append(batch_df)
        batch_count += 1
        oldest_ts = batch_df["timestamp"].min().to_pydatetime()
        current_end = oldest_ts - timedelta(seconds=1)

        if batch_count % 10 == 0:
            total = sum(len(b) for b in all_batches)
            logger.info(f"  [{symbol} FR] batch {batch_count} | {total:,} records | back to {oldest_ts.strftime('%Y-%m-%d')}")

        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])

    df = pd.concat(all_batches, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher_fr_oi.py::TestFetchFundingRate -q`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add data/fetcher.py tests/test_fetcher_fr_oi.py
git commit -m "feat: add fetch_funding_rate to data/fetcher.py"
```

---

## Task 2: Historical Open Interest Fetcher

**Files:**
- Modify: `data/fetcher.py`
- Modify: `tests/test_fetcher_fr_oi.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_fetcher_fr_oi.py`:

```python
def _make_oi_rows(base_ms: int, count: int, interval_ms: int = 3_600_000):
    """Build count OI records, newest first (Bybit order)."""
    rows = []
    for i in range(count - 1, -1, -1):
        ts = base_ms + i * interval_ms
        rows.append({
            "openInterest": "500000.00",
            "timestamp": str(ts),
        })
    return rows


class TestFetchOpenInterest:

    def test_returns_timestamp_and_oi_columns(self):
        from data.fetcher import fetch_open_interest
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": _make_oi_rows(BASE_MS, 3), "nextPageCursor": ""},
        }
        mock_resp.raise_for_status.return_value = None
        with patch("data.fetcher.requests.get", return_value=mock_resp):
            df = fetch_open_interest("ETHUSDT", "2022-01-01", "2022-01-02")
        assert "timestamp" in df.columns
        assert "open_interest" in df.columns
        assert len(df) == 3

    def test_oi_is_float(self):
        from data.fetcher import fetch_open_interest
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": _make_oi_rows(BASE_MS, 2), "nextPageCursor": ""},
        }
        mock_resp.raise_for_status.return_value = None
        with patch("data.fetcher.requests.get", return_value=mock_resp):
            df = fetch_open_interest("ETHUSDT", "2022-01-01", "2022-01-02")
        assert df["open_interest"].dtype == "float64"

    def test_chronological_order(self):
        from data.fetcher import fetch_open_interest
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": _make_oi_rows(BASE_MS, 3), "nextPageCursor": ""},
        }
        mock_resp.raise_for_status.return_value = None
        with patch("data.fetcher.requests.get", return_value=mock_resp):
            df = fetch_open_interest("ETHUSDT", "2022-01-01", "2022-01-02")
        assert list(df["timestamp"]) == sorted(df["timestamp"])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_fetcher_fr_oi.py::TestFetchOpenInterest -q`
Expected: FAIL — `ImportError: cannot import name 'fetch_open_interest'`

- [ ] **Step 3: Write minimal implementation**

Add to `data/fetcher.py` after `fetch_funding_rate`:

```python
def fetch_open_interest(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch historical open interest from Bybit V5 (linear perpetual, 1h interval).

    Uses the public REST endpoint (no pybit session needed).
    Returns chronological DataFrame with columns: [timestamp, open_interest].
    """
    import requests as _requests

    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches = []
    current_start = start_dt
    batch_count = 0

    while current_start < end_dt:
        for attempt in range(config.MAX_RETRIES):
            try:
                resp = _requests.get(
                    "https://api.bybit.com/v5/market/open-interest",
                    params={
                        "category": "linear",
                        "symbol": symbol,
                        "intervalTime": "1h",
                        "startTime": _to_ms(current_start),
                        "endTime": _to_ms(end_dt),
                        "limit": 200,
                    },
                    timeout=10,
                )
                resp.raise_for_status()
                body = resp.json()
                break
            except Exception as exc:
                if attempt == config.MAX_RETRIES - 1:
                    raise
                time.sleep(2 ** attempt)

        if body.get("retCode", 0) != 0:
            raise ValueError(f"Bybit OI API error: {body.get('retMsg')}")

        rows = body["result"]["list"]
        if not rows:
            break

        batch_df = pd.DataFrame(rows)
        batch_df["timestamp"] = pd.to_datetime(
            batch_df["timestamp"].astype("int64"), unit="ms", utc=True
        )
        batch_df["open_interest"] = batch_df["openInterest"].astype(float)
        batch_df = batch_df[["timestamp", "open_interest"]]
        batch_df = batch_df.sort_values("timestamp").reset_index(drop=True)

        all_batches.append(batch_df)
        batch_count += 1
        last_ts = batch_df["timestamp"].iloc[-1].to_pydatetime()
        current_start = last_ts + timedelta(hours=1)

        if batch_count % 10 == 0:
            total = sum(len(b) for b in all_batches)
            logger.info(f"  [{symbol} OI] batch {batch_count} | {total:,} records")

        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=["timestamp", "open_interest"])

    df = pd.concat(all_batches, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_fetcher_fr_oi.py -q`
Expected: 7 passed (4 FR + 3 OI)

- [ ] **Step 5: Commit**

```bash
git add data/fetcher.py tests/test_fetcher_fr_oi.py
git commit -m "feat: add fetch_open_interest to data/fetcher.py"
```

---

## Task 3: Feature Engineering — FR + OI indicators

**Files:**
- Modify: `features/indicators.py`
- Create: `tests/test_indicators_fr_oi.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_indicators_fr_oi.py
import numpy as np
import pandas as pd
import pytest


class TestAttachFundingFeatures:

    def _make_hourly(self, n=100):
        ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "close": np.random.uniform(2000, 3000, n)})

    def _make_fr(self, n=40):
        """40 records * 8h = ~13 days of FR data."""
        ts = pd.date_range("2022-01-01", periods=n, freq="8h", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts,
            "funding_rate": np.random.uniform(-0.001, 0.001, n),
        })

    def test_produces_three_funding_columns(self):
        from features.indicators import _attach_funding_features
        hourly = self._make_hourly(200)
        fr = self._make_fr(80)
        result = _attach_funding_features(hourly, fr)
        assert "funding_rate" in result.columns
        assert "funding_rate_ma_24" in result.columns
        assert "funding_zscore_30d" in result.columns

    def test_no_look_ahead_bias(self):
        from features.indicators import _attach_funding_features
        hourly = self._make_hourly(50)
        fr = pd.DataFrame({
            "timestamp": pd.to_datetime(["2022-01-03T08:00:00+00:00"], utc=True),
            "funding_rate": [0.001],
        })
        result = _attach_funding_features(hourly, fr)
        # Hours before 2022-01-03T08:00 should NOT see this FR
        before = result[result["timestamp"] < pd.Timestamp("2022-01-03T08:00:00+00:00")]
        assert before["funding_rate"].isna().all()


class TestAttachOiFeatures:

    def _make_hourly(self, n=100):
        ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
        roc_24 = np.random.uniform(-0.05, 0.05, n)
        return pd.DataFrame({"timestamp": ts, "close": 2500.0, "roc_24": roc_24})

    def _make_oi(self, n=100):
        ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({
            "timestamp": ts,
            "open_interest": np.cumsum(np.random.uniform(-1000, 1000, n)) + 500_000,
        })

    def test_produces_three_oi_columns(self):
        from features.indicators import _attach_oi_features
        hourly = self._make_hourly()
        oi = self._make_oi()
        result = _attach_oi_features(hourly, oi)
        assert "oi_change_1h" in result.columns
        assert "oi_change_24h" in result.columns
        assert "oi_price_divergence" in result.columns

    def test_divergence_is_binary(self):
        from features.indicators import _attach_oi_features
        hourly = self._make_hourly(50)
        oi = self._make_oi(50)
        result = _attach_oi_features(hourly, oi)
        valid = result["oi_price_divergence"].dropna()
        assert set(valid.unique()).issubset({0, 1, 0.0, 1.0})
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_indicators_fr_oi.py -q`
Expected: FAIL — `ImportError: cannot import name '_attach_funding_features'`

- [ ] **Step 3: Write minimal implementation**

Add to `features/indicators.py` before the final `return df` in `compute()`:

```python
# Add these two new functions at module level:

def _attach_funding_features(
    hourly_df: pd.DataFrame,
    fr_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge funding rate onto hourly bars using merge_asof (backward = no look-ahead).
    Then compute rolling features: ma_24 and zscore_30d."""
    if fr_df is None or fr_df.empty:
        hourly_df = hourly_df.copy()
        hourly_df["funding_rate"] = np.nan
        hourly_df["funding_rate_ma_24"] = np.nan
        hourly_df["funding_zscore_30d"] = np.nan
        return hourly_df

    hourly_sorted = hourly_df.sort_values("timestamp").copy()
    fr_sorted = fr_df[["timestamp", "funding_rate"]].sort_values("timestamp").copy()
    hourly_sorted["timestamp"] = hourly_sorted["timestamp"].astype("datetime64[us, UTC]")
    fr_sorted["timestamp"] = fr_sorted["timestamp"].astype("datetime64[us, UTC]")

    merged = pd.merge_asof(
        hourly_sorted, fr_sorted,
        on="timestamp", direction="backward",
    )
    # Rolling features (computed on the merged hourly index)
    merged["funding_rate_ma_24"] = merged["funding_rate"].rolling(24, min_periods=1).mean()

    # zscore over 30 days = 30*24 = 720 hourly bars (FR is forward-filled so this works)
    window = 720
    roll_mean = merged["funding_rate"].rolling(window, min_periods=1).mean()
    roll_std = merged["funding_rate"].rolling(window, min_periods=1).std()
    merged["funding_zscore_30d"] = (merged["funding_rate"] - roll_mean) / roll_std.replace(0, np.nan)

    return merged


def _attach_oi_features(
    hourly_df: pd.DataFrame,
    oi_df: pd.DataFrame,
) -> pd.DataFrame:
    """Merge open interest onto hourly bars and compute derived features."""
    if oi_df is None or oi_df.empty:
        hourly_df = hourly_df.copy()
        hourly_df["oi_change_1h"] = np.nan
        hourly_df["oi_change_24h"] = np.nan
        hourly_df["oi_price_divergence"] = np.nan
        return hourly_df

    hourly_sorted = hourly_df.sort_values("timestamp").copy()
    oi_sorted = oi_df[["timestamp", "open_interest"]].sort_values("timestamp").copy()
    hourly_sorted["timestamp"] = hourly_sorted["timestamp"].astype("datetime64[us, UTC]")
    oi_sorted["timestamp"] = oi_sorted["timestamp"].astype("datetime64[us, UTC]")

    merged = pd.merge_asof(
        hourly_sorted, oi_sorted,
        on="timestamp", direction="backward",
    )
    merged["oi_change_1h"] = merged["open_interest"].pct_change(1)
    merged["oi_change_24h"] = merged["open_interest"].pct_change(24)
    merged["oi_price_divergence"] = (
        np.sign(merged["oi_change_24h"]) * np.sign(merged["roc_24"]) < 0
    ).astype(int)
    merged = merged.drop(columns=["open_interest"])

    return merged
```

Then modify the `compute()` function signature and body to accept and use FR/OI:

```python
def compute(hourly_df, daily_df, ref_df=None, fr_df=None, oi_df=None):
    # ... existing code ...

    # 13. Funding Rate features
    df = _attach_funding_features(df, fr_df)

    # 14. Open Interest features
    df = _attach_oi_features(df, oi_df)

    return df
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_indicators_fr_oi.py -q`
Expected: 4 passed

- [ ] **Step 5: Run all existing tests to verify no regression**

Run: `pytest -q`
Expected: 131+ passed (existing tests pass because `fr_df=None, oi_df=None` defaults produce NaN columns that get dropped by `dropna()` in builder.py — old feature matrices are unchanged)

- [ ] **Step 6: Commit**

```bash
git add features/indicators.py tests/test_indicators_fr_oi.py
git commit -m "feat: add funding rate + OI feature engineering"
```

---

## Task 4: Wire FR/OI into data pipeline + feature builder

**Files:**
- Modify: `main.py`
- Modify: `features/builder.py`

- [ ] **Step 1: Modify `main.py` to fetch FR/OI**

After the K-line fetch loop, add:

```python
# After the existing for symbol loop, add FR/OI fetching:
for symbol in config.SYMBOLS:
    logger.info(f"抓取 {symbol} Funding Rate | {config.HISTORY_START} → {end_date}")
    fr_df = fetcher.fetch_funding_rate(symbol, config.HISTORY_START, end_date)
    fr_path = config.STORAGE_RAW / f"{symbol}_funding_rate.parquet"
    fr_df.to_parquet(fr_path, index=False)
    logger.info(f"  → 儲存 {fr_path}（{len(fr_df):,} 筆）")

    logger.info(f"抓取 {symbol} Open Interest | {config.HISTORY_START} → {end_date}")
    oi_df = fetcher.fetch_open_interest(symbol, config.HISTORY_START, end_date)
    oi_path = config.STORAGE_RAW / f"{symbol}_open_interest.parquet"
    oi_df.to_parquet(oi_path, index=False)
    logger.info(f"  → 儲存 {oi_path}（{len(oi_df):,} 筆）")
```

- [ ] **Step 2: Modify `features/builder.py` to pass FR/OI to indicators**

In `build()`, after loading hourly/daily parquets, add:

```python
# Load FR/OI if available (Phase 6 data — gracefully skip if not yet fetched)
fr_df = None
oi_df = None
fr_path = raw_dir / f"{symbol}_funding_rate.parquet"
oi_path = raw_dir / f"{symbol}_open_interest.parquet"
if fr_path.exists():
    fr_df = pd.read_parquet(fr_path)
    logger.info(f"[{symbol}] Loaded funding rate: {len(fr_df):,} rows")
if oi_path.exists():
    oi_df = pd.read_parquet(oi_path)
    logger.info(f"[{symbol}] Loaded open interest: {len(oi_df):,} rows")
```

Then change the `indicators.compute()` call:

```python
df = indicators.compute(df, daily_df, ref_df=ref_df, fr_df=fr_df, oi_df=oi_df)
```

- [ ] **Step 3: Run all tests**

Run: `pytest -q`
Expected: All pass (builder gracefully handles missing FR/OI parquets)

- [ ] **Step 4: Commit**

```bash
git add main.py features/builder.py
git commit -m "feat: wire FR/OI into data pipeline and feature builder"
```

---

## Task 5: Fetch historical data + rebuild features

**Files:** No code changes — this is a data pipeline run.

- [ ] **Step 1: Fetch FR/OI history**

```bash
python main.py
```

This will re-fetch all K-lines (incremental append via parquet) plus NEW FR/OI data.
Expected: ~4,000 FR records (3/day * ~1,400 days) and ~33,000 OI records (24/day * ~1,400 days) per symbol.

- [ ] **Step 2: Rebuild features**

```bash
python build_features.py
```

Expected: validation reports now include 6 new feature columns. Check:

```bash
python -c "import json; r=json.load(open('storage/features/ETHUSDT_validation_report.json')); print([c for c in r['metadata']['feature_columns'] if 'funding' in c or 'oi_' in c])"
```

Should print: `['funding_rate', 'funding_rate_ma_24', 'funding_zscore_30d', 'oi_change_1h', 'oi_change_24h', 'oi_price_divergence']`

- [ ] **Step 3: Commit data artifacts note (not the data itself)**

```bash
git add features/builder.py features/indicators.py
git commit -m "data: Phase 6 FR/OI history fetched + features rebuilt"
```

---

## Task 6: Backup old models + retrain + backtest

**Files:**
- Create: `compare_models.py`

- [ ] **Step 1: Backup current models**

```bash
# PowerShell
Copy-Item -Recurse storage/models storage/models_v1
Copy-Item -Recurse storage/features storage/features_v1
Copy-Item -Recurse storage/backtest storage/backtest_v1
```

- [ ] **Step 2: Retrain models**

```bash
python train_models.py
```

Expected: New .pkl files in `storage/models/` that expect 30+ old features plus 6 new ones.

- [ ] **Step 3: Run backtests**

```bash
python run_backtest.py
python run_portfolio_backtest.py
```

- [ ] **Step 4: Write comparison script**

```python
# compare_models.py
"""Compare old vs new model backtest results. Auto-rollback if Sharpe drops."""
import json
import shutil
import logging
from pathlib import Path
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "ETHUSDT"
TARGET = "target_atr"


def main():
    old_report = Path("storage/backtest_v1") / f"{SYMBOL}_{TARGET}_portfolio_report.json"
    new_report = config.STORAGE_BACKTEST / f"{SYMBOL}_{TARGET}_portfolio_report.json"

    if not old_report.exists() or not new_report.exists():
        logger.error("Cannot compare — missing report files")
        return

    old = json.loads(old_report.read_text())
    new = json.loads(new_report.read_text())

    old_sharpe = old["metrics"]["sharpe_ratio"]
    new_sharpe = new["metrics"]["sharpe_ratio"]
    old_return = old["metrics"]["total_return_pct"]
    new_return = new["metrics"]["total_return_pct"]
    old_mdd = old["metrics"]["max_drawdown_pct"]
    new_mdd = new["metrics"]["max_drawdown_pct"]

    print("=" * 60)
    print(f"  Phase 6 Model Comparison: {SYMBOL} {TARGET}")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Old':>12} {'New':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Sharpe Ratio':<25} {old_sharpe:>12.4f} {new_sharpe:>12.4f} {new_sharpe-old_sharpe:>+12.4f}")
    print(f"  {'Total Return %':<25} {old_return:>12.2f} {new_return:>12.2f} {new_return-old_return:>+12.2f}")
    print(f"  {'Max Drawdown %':<25} {old_mdd:>12.2f} {new_mdd:>12.2f} {new_mdd-old_mdd:>+12.2f}")
    print("=" * 60)

    if new_sharpe >= old_sharpe:
        print(f"\n  ✅ NEW MODEL WINS (Sharpe {old_sharpe:.4f} → {new_sharpe:.4f})")
        print("  Keeping new models. Safe to deploy.")
    else:
        print(f"\n  ❌ OLD MODEL BETTER (Sharpe {new_sharpe:.4f} < {old_sharpe:.4f})")
        print("  Rolling back to v1 models...")
        # Rollback
        shutil.rmtree(config.STORAGE_MODELS)
        shutil.copytree("storage/models_v1", config.STORAGE_MODELS)
        shutil.rmtree(config.STORAGE_FEATURES)
        shutil.copytree("storage/features_v1", config.STORAGE_FEATURES)
        shutil.rmtree(config.STORAGE_BACKTEST)
        shutil.copytree("storage/backtest_v1", config.STORAGE_BACKTEST)
        print("  Rollback complete. Old models restored.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run comparison**

```bash
python compare_models.py
```

If `✅ NEW MODEL WINS` → proceed to Task 7.
If `❌ OLD MODEL BETTER` → skip Task 7, deployment not needed. Phase 6 data is still stored for future experiments.

- [ ] **Step 6: Commit**

```bash
git add compare_models.py
git commit -m "feat: model comparison + auto-rollback script"
```

---

## Task 7: Live Pipeline — FR/OI Fetching (only if new model won)

**Files:**
- Modify: `live/fetcher.py`
- Create: `tests/test_live_fetcher_fr_oi.py`
- Modify: `live/pipeline.py`

- [ ] **Step 1: Write failing tests for live FR/OI fetchers**

```python
# tests/test_live_fetcher_fr_oi.py
from unittest.mock import MagicMock, patch
import pandas as pd
import pytest


class TestFetchFundingRateLatest:

    def test_returns_funding_rate_dataframe(self, monkeypatch):
        from live import fetcher
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": [
                {"symbol": "ETHUSDT", "fundingRate": "0.0001", "fundingRateTimestamp": "1700000000000"},
                {"symbol": "ETHUSDT", "fundingRate": "0.0002", "fundingRateTimestamp": "1699971200000"},
            ]},
        }
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr(fetcher, "_sleep", lambda s: None)
        monkeypatch.setattr(fetcher.requests, "get", lambda *a, **kw: mock_resp)
        df = fetcher.fetch_funding_rate_latest("ETHUSDT", 2)
        assert "timestamp" in df.columns
        assert "funding_rate" in df.columns
        assert len(df) == 2


class TestFetchOiLatest:

    def test_returns_oi_dataframe(self, monkeypatch):
        from live import fetcher
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": [
                {"openInterest": "800000.00", "timestamp": "1700000000000"},
                {"openInterest": "790000.00", "timestamp": "1699996400000"},
            ]},
        }
        mock_resp.raise_for_status.return_value = None
        monkeypatch.setattr(fetcher, "_sleep", lambda s: None)
        monkeypatch.setattr(fetcher.requests, "get", lambda *a, **kw: mock_resp)
        df = fetcher.fetch_oi_latest("ETHUSDT", 2)
        assert "timestamp" in df.columns
        assert "open_interest" in df.columns
        assert len(df) == 2
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_live_fetcher_fr_oi.py -q`
Expected: FAIL

- [ ] **Step 3: Add live fetcher functions to `live/fetcher.py`**

```python
BYBIT_FR_URL = "https://api.bybit.com/v5/market/funding/history"
BYBIT_OI_URL = "https://api.bybit.com/v5/market/open-interest"


def fetch_funding_rate_latest(symbol: str, n: int = 100) -> pd.DataFrame:
    """Fetch n most recent funding rate records. Returns chronological DataFrame."""
    body = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BYBIT_FR_URL,
                params={"category": "linear", "symbol": symbol, "limit": n},
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            _sleep(_BACKOFF_BASE_SEC * (2 ** attempt))

    if body.get("retCode", 0) != 0:
        raise ValueError(f"Bybit FR API error: {body.get('retMsg')}")

    rows = body["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "funding_rate"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["fundingRateTimestamp"].astype("int64"), unit="ms", utc=True)
    df["funding_rate"] = df["fundingRate"].astype(float)
    return df[["timestamp", "funding_rate"]].sort_values("timestamp").reset_index(drop=True)


def fetch_oi_latest(symbol: str, n: int = 30) -> pd.DataFrame:
    """Fetch n most recent hourly OI snapshots. Returns chronological DataFrame."""
    body = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BYBIT_OI_URL,
                params={"category": "linear", "symbol": symbol, "intervalTime": "1h", "limit": n},
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            if attempt == MAX_RETRIES - 1:
                raise
            _sleep(_BACKOFF_BASE_SEC * (2 ** attempt))

    if body.get("retCode", 0) != 0:
        raise ValueError(f"Bybit OI API error: {body.get('retMsg')}")

    rows = body["result"]["list"]
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open_interest"])
    df = pd.DataFrame(rows)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    df["open_interest"] = df["openInterest"].astype(float)
    return df[["timestamp", "open_interest"]].sort_values("timestamp").reset_index(drop=True)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_live_fetcher_fr_oi.py -q`
Expected: 2 passed

- [ ] **Step 5: Modify `live/pipeline.py` to fetch and pass FR/OI**

Add constants at top:

```python
FR_LOOKBACK = 100   # 30d * 3/day = 90, plus buffer
OI_LOOKBACK = 30    # 24h change needs 25 rows, plus buffer
```

In `compute_signal()`, after the BTC ref_df block, add:

```python
# Funding Rate + Open Interest (Phase 6)
fr_df = None
oi_df = None
try:
    from live.fetcher import fetch_funding_rate_latest, fetch_oi_latest
    fr_df = fetch_funding_rate_latest(symbol, FR_LOOKBACK)
    oi_df = fetch_oi_latest(symbol, OI_LOOKBACK)
except Exception as e:
    logger.warning(f"[{symbol}] FR/OI fetch failed (non-fatal): {e}")
```

Then change the `indicators.compute()` call:

```python
df = indicators.compute(df, daily_df, ref_df=ref_df, fr_df=fr_df, oi_df=oi_df)
```

- [ ] **Step 6: Run all tests**

Run: `pytest -q`
Expected: All pass

- [ ] **Step 7: Commit**

```bash
git add live/fetcher.py live/pipeline.py tests/test_live_fetcher_fr_oi.py
git commit -m "feat: live pipeline fetches FR/OI for Phase 6 features"
```

---

## Task 8: Deploy + Verify

- [ ] **Step 1: Push to GitHub**

```bash
git push
```

- [ ] **Step 2: Wait for CI green**

Check GitHub Actions — all tests must pass.

- [ ] **Step 3: Deploy to VM**

```powershell
.\deploy_oracle.ps1
```

- [ ] **Step 4: Verify heartbeat**

```powershell
ssh ubuntu@140.238.37.45 "sudo journalctl -u bybit-ml -n 15 --no-pager"
```

Expected: heartbeat runs without error, `[ETHUSDT] prob=X.XXXX signal=False/True` appears.
The prob value may differ from before (new features change model output). This is expected.

- [ ] **Step 5: Commit deployment note**

```bash
git commit --allow-empty -m "deploy: Phase 6 FR/OI features live on VM"
git push
```
