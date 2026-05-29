# Phase 2 Feature Engineering Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a feature engineering pipeline that reads Phase 1 Parquet files and outputs a labeled feature matrix (X + Y) ready for ML model training, with a statistical validation report.

**Architecture:** Four single-responsibility modules (`labels`, `indicators`, `validator`, `builder`) under `features/`, orchestrated by `build_features.py`. Triple barrier labels use numpy vectorized 2D windows. Technical indicators use pandas-ta as the engine.

**Tech Stack:** pandas, pandas-ta, numpy (stride_tricks), scipy (pointbiserialr), pyarrow

---

## File Structure

| File | Action | Responsibility |
|------|--------|----------------|
| `config.py` | Modify | Add `STORAGE_FEATURES` path |
| `requirements.txt` | Modify | Add pandas-ta, scipy |
| `features/__init__.py` | Create | Empty module init |
| `features/labels.py` | Create | Vectorized triple barrier labeling |
| `features/indicators.py` | Create | pandas-ta multi-window indicators + daily features |
| `features/validator.py` | Create | Label distribution, correlation, JSON report |
| `features/builder.py` | Create | Pipeline orchestrator |
| `build_features.py` | Create | CLI entry point |
| `tests/test_labels.py` | Create | 5 label behavior tests |
| `tests/test_indicators.py` | Create | 7 column-existence tests |
| `tests/test_validator.py` | Create | 3 validator output tests |
| `tests/test_builder.py` | Create | 3 end-to-end data quality tests |

---

### Task 1: Setup — config, requirements, features package

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`
- Create: `features/__init__.py`

- [ ] **Step 1: Add STORAGE_FEATURES to config.py**

In `config.py`, add after the `STORAGE_EXCEL` line:

```python
STORAGE_FEATURES = BASE_DIR / "storage" / "features"
```

- [ ] **Step 2: Update requirements.txt**

Replace the contents of `requirements.txt` with:

```
pybit>=5.0.0
pandas>=2.0.0
pyarrow>=14.0.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
pytest>=7.0.0
numpy>=1.26.0,<2
pandas-ta>=0.3.14b
scipy>=1.10.0
```

- [ ] **Step 3: Install new dependencies**

```bash
pip install pandas-ta scipy
```

Expected: installs without errors.

- [ ] **Step 4: Create features/__init__.py**

Create `features/__init__.py` as an empty file.

- [ ] **Step 5: Commit**

```bash
git add config.py requirements.txt features/__init__.py
git commit -m "feat: Phase 2 setup — STORAGE_FEATURES, pandas-ta, scipy"
```

---

### Task 2: features/labels.py — Triple Barrier Method

**Files:**
- Create: `features/labels.py`
- Create: `tests/test_labels.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_labels.py`:

```python
import numpy as np
import pandas as pd

HORIZON = 24


def _make_df(n, closes, highs, lows):
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes.copy(),
        "high": highs,
        "low": lows,
        "close": closes.copy(),
        "volume": np.ones(n),
        "turnover": np.ones(n) * 100.0,
        "atr_14": np.ones(n) * 1.0,
        "atr_24": np.ones(n) * 1.0,
    })


class TestLabels:
    def test_tp_hit_within_horizon_label_1(self):
        """Bar 5 high >= TP (+2%) → entry bar 0 gets label=1."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[5] = 102.0          # bar 5: high hits TP (100 * 1.02 = 102.0)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 1.0

    def test_sl_hit_within_horizon_label_0(self):
        """Bar 3 low <= SL (-1%) → entry bar 0 gets label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        lows[3] = 98.9            # bar 3: low hits SL (100 * 0.99 = 99.0)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_time_limit_no_hit_label_0(self):
        """No barrier hit within 24 bars → time limit → label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)   # never reaches 102
        lows = np.full(n, 99.5)     # never reaches 99
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_intra_bar_collision_sl_wins(self):
        """Same bar hits both TP and SL → conservative pessimism → label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[2] = 102.1           # bar 2 hits TP
        lows[2] = 98.9             # bar 2 also hits SL (same bar collision)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_tail_nans_last_horizon_rows(self):
        """Last HORIZON rows must have NaN for both target columns."""
        from features.labels import compute
        n = HORIZON + 10
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[-HORIZON:].isna().all()
        assert result["target_atr"].iloc[-HORIZON:].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_labels.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'features.labels'`

- [ ] **Step 3: Implement features/labels.py**

Create `features/labels.py`:

```python
# features/labels.py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZON = 24
TP_PCT_FIXED = 0.02
SL_PCT_FIXED = 0.01
ATR_TP_MULT = 3.0
ATR_SL_MULT = 1.5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values

    df["target_fixed"] = _barrier_labels(
        highs, lows,
        tp=closes * (1.0 + TP_PCT_FIXED),
        sl=closes * (1.0 - SL_PCT_FIXED),
        n=len(df),
    )
    df["target_atr"] = _barrier_labels(
        highs, lows,
        tp=closes + ATR_TP_MULT * atrs,
        sl=closes - ATR_SL_MULT * atrs,
        n=len(df),
    )
    return df


def _barrier_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    n: int,
) -> np.ndarray:
    n_valid = n - HORIZON
    labels = np.full(n, np.nan)
    if n_valid <= 0:
        return labels

    # future_highs[i] = highs[i+1 : i+1+HORIZON], shape (n_valid, HORIZON)
    future_highs = sliding_window_view(highs[1:], HORIZON)
    future_lows  = sliding_window_view(lows[1:],  HORIZON)

    tp_hit = future_highs >= tp[:n_valid, np.newaxis]  # (n_valid, HORIZON)
    sl_hit = future_lows  <= sl[:n_valid, np.newaxis]  # (n_valid, HORIZON)

    # First hit index; HORIZON means "never hit"
    tp_first = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), HORIZON)
    sl_first = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), HORIZON)

    # Conservative pessimism: SL wins ties (same-bar collision → SL)
    wins_tp = (sl_first > tp_first) & (tp_first < HORIZON)
    labels[:n_valid] = np.where(wins_tp, 1.0, 0.0)
    return labels
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_labels.py -v
```

Expected: 5 passed

- [ ] **Step 5: Commit**

```bash
git add features/labels.py tests/test_labels.py
git commit -m "feat: add vectorized triple barrier labeling with conservative pessimism"
```

---

### Task 3: features/indicators.py — Multi-Window Technical Features

**Files:**
- Create: `features/indicators.py`
- Create: `tests/test_indicators.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_indicators.py`:

```python
import numpy as np
import pandas as pd


def _make_hourly(n=300):
    np.random.seed(42)
    price = 100.0 + np.cumsum(np.random.randn(n) * 0.5)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC"),
        "open": price, "high": price + 0.5, "low": price - 0.5, "close": price,
        "volume": np.random.uniform(10, 100, n),
        "turnover": np.random.uniform(1000, 10000, n),
    })


def _make_daily(n=300):
    np.random.seed(99)
    price = 100.0 + np.cumsum(np.random.randn(n) * 1.0)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="1D", tz="UTC"),
        "open": price, "high": price + 2, "low": price - 2, "close": price,
        "volume": np.random.uniform(100, 1000, n),
        "turnover": np.random.uniform(10000, 100000, n),
    })


class TestIndicators:
    def test_rsi_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["rsi_7", "rsi_14", "rsi_24"]:
            assert col in df.columns, f"Missing: {col}"

    def test_ppo_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["ppo", "ppo_signal", "ppo_hist"]:
            assert col in df.columns, f"Missing: {col}"

    def test_atr_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["atr_14", "atr_24"]:
            assert col in df.columns, f"Missing: {col}"

    def test_bband_width_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["bband_width_20", "bband_width_50"]:
            assert col in df.columns, f"Missing: {col}"

    def test_ma_bias_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["ma_bias_20", "ma_bias_50", "ma_bias_200"]:
            assert col in df.columns, f"Missing: {col}"

    def test_volume_ratio_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["vol_ratio_12", "vol_ratio_24", "turnover_ratio_12", "turnover_ratio_24"]:
            assert col in df.columns, f"Missing: {col}"

    def test_daily_feature_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["daily_rsi_14", "daily_atr_14",
                    "daily_ma_bias_20", "daily_ma_bias_50", "daily_ma_bias_200"]:
            assert col in df.columns, f"Missing: {col}"
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_indicators.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'features.indicators'`

- [ ] **Step 3: Implement features/indicators.py**

Create `features/indicators.py`:

```python
# features/indicators.py
import pandas as pd
import pandas_ta as ta


def compute(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()

    # RSI: [7, 14, 24]
    for period in [7, 14, 24]:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

    # PPO replaces MACD — output is percentage-based, no absolute-value trap
    # pandas-ta returns columns in order: PPO, PPOs (signal), PPOh (histogram)
    ppo = ta.ppo(df["close"], fast=12, slow=26, signal=9)
    df["ppo"]        = ppo.iloc[:, 0]
    df["ppo_signal"] = ppo.iloc[:, 1]
    df["ppo_hist"]   = ppo.iloc[:, 2]

    # ATR: [14, 24]
    for period in [14, 24]:
        df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)

    # Bollinger Band width = (upper - lower) / middle
    # pandas-ta bbands columns order: BBL, BBM, BBU, BBB, BBP
    for length, std in [(20, 2.0), (50, 2.5)]:
        bb = ta.bbands(df["close"], length=length, std=std)
        lower, middle, upper = bb.iloc[:, 0], bb.iloc[:, 1], bb.iloc[:, 2]
        df[f"bband_width_{length}"] = (upper - lower) / middle

    # MA Bias = (close - SMA_N) / SMA_N
    for period in [20, 50, 200]:
        sma = ta.sma(df["close"], length=period)
        df[f"ma_bias_{period}"] = (df["close"] - sma) / sma

    # Volume / Turnover ratios
    for period in [12, 24]:
        df[f"vol_ratio_{period}"]      = df["volume"]   / df["volume"].rolling(period).mean()
        df[f"turnover_ratio_{period}"] = df["turnover"] / df["turnover"].rolling(period).mean()

    # Daily indicators — computed independently, then merged with look-ahead prevention
    df = _attach_daily_features(df, daily_df)
    return df


def _attach_daily_features(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    daily = daily_df.copy()

    daily["daily_rsi_14"] = ta.rsi(daily["close"], length=14)
    daily["daily_atr_14"] = ta.atr(daily["high"], daily["low"], daily["close"], length=14)
    for period in [20, 50, 200]:
        sma = ta.sma(daily["close"], length=period)
        daily[f"daily_ma_bias_{period}"] = (daily["close"] - sma) / sma

    # Shift forward by 1 day to prevent look-ahead bias (same logic as Phase 1)
    daily["date_available"] = daily["timestamp"] + pd.Timedelta(days=1)

    feat_cols = [
        "date_available", "daily_rsi_14", "daily_atr_14",
        "daily_ma_bias_20", "daily_ma_bias_50", "daily_ma_bias_200",
    ]
    merged = pd.merge_asof(
        hourly_df.sort_values("timestamp"),
        daily[feat_cols].sort_values("date_available"),
        left_on="timestamp",
        right_on="date_available",
        direction="backward",
    )
    return merged.drop(columns=["date_available"], errors="ignore")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_indicators.py -v
```

Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add features/indicators.py tests/test_indicators.py
git commit -m "feat: add multi-window technical indicators via pandas-ta with daily feature alignment"
```

---

### Task 4: features/validator.py — Statistical Validation

**Files:**
- Create: `features/validator.py`
- Create: `tests/test_validator.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_validator.py`:

```python
import json
import numpy as np
import pandas as pd
from pathlib import Path


def _make_balanced_df(n=200):
    np.random.seed(7)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC"),
        "rsi_14": np.random.uniform(20, 80, n),
        "ma_bias_50": np.random.uniform(-0.05, 0.05, n),
        "target_fixed": np.random.randint(0, 2, n).astype(float),
        "target_atr":   np.random.randint(0, 2, n).astype(float),
    })


def _make_imbalanced_df(n=200, positive_rate=0.05):
    df = _make_balanced_df(n)
    labels = np.zeros(n)
    labels[:int(n * positive_rate)] = 1.0
    df["target_fixed"] = labels
    df["target_atr"]   = labels
    return df


class TestValidator:
    def test_report_creates_json_file(self, tmp_path):
        """report() saves a readable JSON file with required top-level keys."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_balanced_df(), out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "metadata" in data
        assert "feature_columns" in data["metadata"]
        assert "target_columns" in data["metadata"]

    def test_report_contains_class_balance(self, tmp_path):
        """class_balance section includes an entry for target_fixed."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_balanced_df(), out)
        data = json.loads(out.read_text())
        assert "class_balance" in data
        assert "target_fixed" in data["class_balance"]

    def test_imbalanced_labels_trigger_warning(self, tmp_path):
        """5% positive rate is < 20% threshold → warning field is non-null."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_imbalanced_df(positive_rate=0.05), out)
        data = json.loads(out.read_text())
        assert data["class_balance"]["target_fixed"]["warning"] is not None
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_validator.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'features.validator'`

- [ ] **Step 3: Implement features/validator.py**

Create `features/validator.py`:

```python
# features/validator.py
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pointbiserialr

logger = logging.getLogger(__name__)

TARGET_COLS = ["target_fixed", "target_atr"]
_EXCLUDE = {
    "timestamp", "open", "high", "low", "close", "volume", "turnover",
    "daily_open", "daily_high", "daily_low", "daily_close",
    "daily_volume", "daily_turnover",
}


def report(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in _EXCLUDE and c not in TARGET_COLS]

    result = {
        "metadata": {
            "total_rows": len(df),
            "feature_columns": feature_cols,
            "target_columns": TARGET_COLS,
        },
        "class_balance": {},
        "correlations_with_target_fixed": {},
        "correlations_with_target_atr": {},
        "high_collinearity_warnings": [],
    }

    # 1. Label distribution
    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        rate = float(df[target].mean())
        warning = None
        if rate < 0.20 or rate > 0.80:
            warning = f"Severe imbalance: {rate:.1%} positive"
            logger.warning(f"[{target}] {warning}")
        else:
            logger.info(f"[{target}] positive rate: {rate:.1%}")
        result["class_balance"][target] = {"positive_rate": round(rate, 4), "warning": warning}

    # 2. Feature-target correlation (Point-Biserial)
    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        y = df[target].values
        corrs = {}
        for feat in feature_cols:
            try:
                r, _ = pointbiserialr(df[feat].values, y)
                corrs[feat] = round(float(r), 4)
            except Exception:
                corrs[feat] = None
        result[f"correlations_with_{target}"] = dict(
            sorted(corrs.items(), key=lambda x: abs(x[1] or 0), reverse=True)
        )

    # 3. High inter-feature collinearity (|r| > 0.95)
    if len(feature_cols) > 1:
        cm = df[feature_cols].corr()
        warnings = []
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols):
                if j <= i:
                    continue
                r = cm.loc[f1, f2]
                if abs(r) > 0.95:
                    warnings.append([f1, f2, round(float(r), 4)])
                    logger.warning(f"High collinearity: {f1} ↔ {f2} r={r:.3f}")
        result["high_collinearity_warnings"] = warnings

    # 4. NaN / inf sanity check
    all_cols = feature_cols + [t for t in TARGET_COLS if t in df.columns]
    nan_count = int(df[all_cols].isna().sum().sum())
    inf_count = int(np.isinf(df[feature_cols].select_dtypes(include=[np.number]).values).sum())
    if nan_count > 0 or inf_count > 0:
        logger.error(f"Data quality: {nan_count} NaNs, {inf_count} infs still present!")
    else:
        logger.info("Data quality: 0 NaN, 0 inf ✓")

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Validation report: {output_path}")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_validator.py -v
```

Expected: 3 passed

- [ ] **Step 5: Commit**

```bash
git add features/validator.py tests/test_validator.py
git commit -m "feat: add statistical validator with label distribution, correlation, and JSON report"
```

---

### Task 5: features/builder.py + build_features.py — Orchestrator

**Files:**
- Create: `features/builder.py`
- Create: `build_features.py`
- Create: `tests/test_builder.py`

- [ ] **Step 1: Write the failing tests**

Create `tests/test_builder.py`:

```python
import numpy as np
import pandas as pd
from pathlib import Path


def _write_raw_parquets(raw_dir: Path, symbol: str) -> None:
    """Write synthetic 1H and 1D Parquet files large enough for all warm-ups.

    n_h = 6000 hours (250 days). n_d = 250 days.
    After daily MA200 warm-up (4800 h) + hourly MA200 (200 h) + tail (24 h),
    approximately 976 valid rows remain.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    n_h = 6000
    h_price = 100.0 + np.cumsum(np.random.randn(n_h) * 0.5)
    hourly = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n_h, freq="1h", tz="UTC"),
        "open": h_price, "high": h_price + 0.5, "low": h_price - 0.5,
        "close": h_price,
        "volume": np.random.uniform(10, 100, n_h),
        "turnover": np.random.uniform(1000, 10000, n_h),
    })
    hourly.to_parquet(raw_dir / f"{symbol}_1h.parquet", index=False)

    n_d = 250
    d_price = 100.0 + np.cumsum(np.random.randn(n_d) * 1.0)
    daily = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n_d, freq="1D", tz="UTC"),
        "open": d_price, "high": d_price + 2, "low": d_price - 2,
        "close": d_price,
        "volume": np.random.uniform(100, 1000, n_d),
        "turnover": np.random.uniform(10000, 100000, n_d),
    })
    daily.to_parquet(raw_dir / f"{symbol}_1d.parquet", index=False)


class TestBuilder:
    def test_output_has_no_nans(self, tmp_path):
        """Feature Parquet must contain zero NaN values after the full pipeline."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        bad = df.isna().sum()
        assert bad.sum() == 0, f"NaNs found:\n{bad[bad > 0]}"

    def test_output_has_no_infs(self, tmp_path):
        """Feature Parquet must contain zero inf values."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        numeric = df.select_dtypes(include=[float, int])
        assert not (numeric.values == float("inf")).any()
        assert not (numeric.values == float("-inf")).any()

    def test_output_has_binary_target_columns(self, tmp_path):
        """target_fixed and target_atr must exist and contain only 0.0 and 1.0."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        assert "target_fixed" in df.columns
        assert "target_atr" in df.columns
        assert df["target_fixed"].isin([0.0, 1.0]).all()
        assert df["target_atr"].isin([0.0, 1.0]).all()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_builder.py -v
```

Expected: FAIL — `ModuleNotFoundError: No module named 'features.builder'`

- [ ] **Step 3: Implement features/builder.py**

Create `features/builder.py`:

```python
# features/builder.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import config
from data import cleaner
from features import indicators, labels, validator

logger = logging.getLogger(__name__)


def build(
    symbol: str,
    raw_dir: Path = None,
    features_dir: Path = None,
) -> None:
    raw_dir      = Path(raw_dir)      if raw_dir      is not None else config.STORAGE_RAW
    features_dir = Path(features_dir) if features_dir is not None else config.STORAGE_FEATURES
    features_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 1 Parquet
    hourly_df = pd.read_parquet(raw_dir / f"{symbol}_1h.parquet")
    daily_df  = pd.read_parquet(raw_dir / f"{symbol}_1d.parquet")

    # 2. Dual-timeframe alignment (Phase 1 function, prevents look-ahead bias)
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)

    # 3. Compute technical indicators — produces head NaNs from rolling windows
    df = indicators.compute(df, daily_df)

    # 4. Compute triple barrier labels — produces tail NaNs for last HORIZON rows
    df = labels.compute(df)

    # 5. Clean: replace inf first, then drop NaN (order matters)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6. Statistical validation (operates on clean data)
    validator.report(df, features_dir / "validation_report.json")

    # 7. Save feature matrix
    out_path = features_dir / f"{symbol}_features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Features saved: {out_path} ({len(df):,} rows)")
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_builder.py -v
```

Expected: 3 passed

- [ ] **Step 5: Run full test suite**

```bash
python -m pytest -v
```

Expected: 39 passed (21 Phase 1 + 18 Phase 2)

- [ ] **Step 6: Create build_features.py**

Create `build_features.py`:

```python
import logging
import config
from features import builder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    for symbol in config.SYMBOLS:
        builder.build(symbol)
```

- [ ] **Step 7: Commit**

```bash
git add features/builder.py build_features.py tests/test_builder.py
git commit -m "feat: add feature builder orchestrator and build_features entry point"
```
