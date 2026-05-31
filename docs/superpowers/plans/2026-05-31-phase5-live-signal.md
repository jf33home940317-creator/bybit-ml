# Phase 5 Live Signal Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立每小時心跳迴圈，從 Bybit 公開 API 抓取即時 K 線，對 ETHUSDT 模型進行集成推論，以 JSON 追蹤虛擬持倉，並透過 Discord Webhook 推播交易訊號。

**Architecture:** `live/` 套件封裝五個單一職責模組（fetcher / pipeline / state / ledger / notifier）；`run_live.py` 以 `schedule` 每小時 :01 分觸發 `heartbeat()`。State 以 JSON 原子寫入（tmp → rename）實現 Crash Recovery。Pipeline 繞過 `build()` 的檔案 I/O，直接呼叫 `cleaner.align_daily_to_hourly` + `indicators.compute`。

**Tech Stack:** requests, schedule (新增), discord.py (現有 Discord bot 的 webhook URL), pybit (現有), pandas, numpy, joblib, xgboost (均已安裝)

---

## File Map

| 動作 | 路徑 | 責任 |
|------|------|------|
| 修改 | `config.py` | 新增 `STORAGE_LIVE`、`DISCORD_WEBHOOK_URL` |
| 修改 | `requirements.txt` | 新增 `schedule>=1.2.0` |
| 建立 | `live/__init__.py` | 空白套件初始化 |
| 建立 | `live/fetcher.py` | Bybit V5 公開 API 抓最新 N 根 K 線 |
| 建立 | `live/pipeline.py` | 特徵重建 + 集成推論，回傳 signal dict |
| 建立 | `live/state.py` | `active_positions.json` 原子讀寫 |
| 建立 | `live/ledger.py` | `paper_trading_ledger.json` 讀寫 |
| 建立 | `live/notifier.py` | Discord Webhook POST |
| 建立 | `run_live.py` | `schedule` 心跳迴圈主程式 |
| 建立 | `tests/test_live_fetcher.py` | fetcher 單元測試（mock HTTP） |
| 建立 | `tests/test_live_state.py` | state / ledger 單元測試（tmp_path） |

`backtest/`、`features/`、`data/`、`models/` 均**不修改**（OCP）。

---

### Task 1: Config 擴充 + 套件安裝

**Files:**
- Modify: `config.py`
- Modify: `requirements.txt`

- [ ] **Step 1: 修改 config.py，新增 Live 相關設定**

在 `config.py` 的 `STORAGE_BACKTEST` 行之後加入：

```python
STORAGE_LIVE = BASE_DIR / "storage" / "live"
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
LIVE_TARGET = "target_atr"
LIVE_SYMBOLS = ["ETHUSDT"]   # BTCUSDT excluded (negative Sharpe in Phase 4.2)
```

- [ ] **Step 2: 在 requirements.txt 新增 schedule**

在 `requirements.txt` 末尾加入：

```
schedule>=1.2.0
```

- [ ] **Step 3: 安裝 schedule**

```
pip install schedule
python -c "import schedule; print(schedule.__version__)"
```

預期：印出版本號（如 `1.2.2`）

- [ ] **Step 4: 建立 live/__init__.py**

建立空白檔案 `live/__init__.py`（內容為空）。

- [ ] **Step 5: 語法確認**

```
python -c "import config; print(config.STORAGE_LIVE); print(config.DISCORD_WEBHOOK_URL)"
```

預期：印出 `...\storage\live` 和空字串（或已設定的 Webhook URL）。

- [ ] **Step 6: Commit**

```bash
git add config.py requirements.txt live/__init__.py
git commit -m "feat: add STORAGE_LIVE config and schedule dependency for Phase 5"
```

---

### Task 2: TDD — 測試 + 實作 live/fetcher.py

**Files:**
- Create: `tests/test_live_fetcher.py`
- Create: `live/fetcher.py`

- [ ] **Step 1: 寫 tests/test_live_fetcher.py（4 個測試，先全部 FAIL）**

```python
# tests/test_live_fetcher.py
from unittest.mock import patch, MagicMock
import pandas as pd
import pytest

# Bybit V5 kline response: newest first, columns = [startTime, open, high, low, close, volume, turnover]
_MOCK_ROWS_NEWEST_FIRST = [
    ["1716819600000", "3015", "3025", "3005", "3020", "120", "360000"],
    ["1716816000000", "3000", "3010", "2990", "3005", "100", "300000"],
]

def _mock_get(rows_newest_first):
    mock = MagicMock()
    mock.json.return_value = {"result": {"list": rows_newest_first}}
    mock.raise_for_status.return_value = None
    return mock


class TestFetchLatest:

    def test_returns_seven_columns(self):
        with patch("live.fetcher.requests.get", return_value=_mock_get(_MOCK_ROWS_NEWEST_FIRST)):
            from live.fetcher import fetch_latest
            df = fetch_latest("ETHUSDT", "60", 2)
        expected = {"timestamp", "open", "high", "low", "close", "volume", "turnover"}
        assert expected.issubset(df.columns)

    def test_rows_in_chronological_order(self):
        with patch("live.fetcher.requests.get", return_value=_mock_get(_MOCK_ROWS_NEWEST_FIRST)):
            from live.fetcher import fetch_latest
            df = fetch_latest("ETHUSDT", "60", 2)
        assert df["timestamp"].iloc[0] < df["timestamp"].iloc[1]

    def test_numeric_columns_are_float(self):
        with patch("live.fetcher.requests.get", return_value=_mock_get(_MOCK_ROWS_NEWEST_FIRST)):
            from live.fetcher import fetch_latest
            df = fetch_latest("ETHUSDT", "60", 2)
        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            assert df[col].dtype == float, f"{col} should be float"

    def test_timestamp_is_utc_aware(self):
        with patch("live.fetcher.requests.get", return_value=_mock_get(_MOCK_ROWS_NEWEST_FIRST)):
            from live.fetcher import fetch_latest
            df = fetch_latest("ETHUSDT", "60", 2)
        assert df["timestamp"].dt.tz is not None, "timestamp must be UTC-aware"
```

- [ ] **Step 2: 確認 4 個測試均 FAIL（ImportError）**

```
pytest tests/test_live_fetcher.py -v 2>&1 | head -20
```

預期：`ModuleNotFoundError: No module named 'live.fetcher'`

- [ ] **Step 3: 實作 live/fetcher.py**

```python
# live/fetcher.py
import requests
import pandas as pd

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


def fetch_latest(symbol: str, interval: str, n: int = 300) -> pd.DataFrame:
    """
    Fetch n most recent candles from Bybit V5 public endpoint (no API key needed).
    Returns chronological DataFrame with UTC-aware timestamps.
    """
    resp = requests.get(
        BYBIT_KLINE_URL,
        params={"symbol": symbol, "interval": interval, "limit": n},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()["result"]["list"]   # Bybit returns newest first
    rows = list(reversed(rows))            # convert to chronological
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df
```

- [ ] **Step 4: 確認 4 個測試全部 PASS**

```
pytest tests/test_live_fetcher.py -v
```

預期：4 tests PASSED

- [ ] **Step 5: 確認完整測試套件無回歸**

```
pytest tests/ -v 2>&1 | tail -5
```

預期：所有測試 PASSED（含原有 86 個）

- [ ] **Step 6: Commit**

```bash
git add tests/test_live_fetcher.py live/fetcher.py
git commit -m "feat: add live/fetcher.py with Bybit V5 public API (TDD)"
```

---

### Task 3: TDD — 測試 + 實作 live/state.py 和 live/ledger.py

**Files:**
- Create: `tests/test_live_state.py`
- Create: `live/state.py`
- Create: `live/ledger.py`

- [ ] **Step 1: 寫 tests/test_live_state.py（8 個測試，先全部 FAIL）**

```python
# tests/test_live_state.py
import json
import pytest
import pandas as pd
from pathlib import Path


# ─── state.py 測試 ────────────────────────────────────────────────────────────

class TestLoadState:

    def test_returns_empty_when_no_file(self, tmp_path):
        from live.state import load_state
        result = load_state(state_file=tmp_path / "active_positions.json")
        assert result == {"positions": []}

    def test_load_returns_saved_data(self, tmp_path):
        from live.state import save_state, load_state
        sf = tmp_path / "active_positions.json"
        data = {"positions": [{"symbol": "ETHUSDT", "exit_time": "2030-01-01T00:00:00+00:00"}]}
        save_state(data, state_file=sf)
        assert load_state(state_file=sf) == data


class TestExpirePositions:

    def test_removes_past_positions(self):
        from live.state import expire_closed_positions
        now = pd.Timestamp.utcnow()
        past   = (now - pd.Timedelta(hours=1)).isoformat()
        future = (now + pd.Timedelta(hours=1)).isoformat()
        s = {"positions": [{"symbol": "A", "exit_time": past},
                            {"symbol": "B", "exit_time": future}]}
        new_s, expired = expire_closed_positions(s, now)
        assert len(new_s["positions"]) == 1
        assert new_s["positions"][0]["symbol"] == "B"
        assert len(expired) == 1
        assert expired[0]["symbol"] == "A"

    def test_keeps_all_when_none_expired(self):
        from live.state import expire_closed_positions
        now    = pd.Timestamp.utcnow()
        future = (now + pd.Timedelta(hours=1)).isoformat()
        s = {"positions": [{"symbol": "A", "exit_time": future}]}
        new_s, expired = expire_closed_positions(s, now)
        assert len(new_s["positions"]) == 1
        assert expired == []


class TestCountAndAdd:

    def test_count_active(self):
        from live.state import count_active
        s = {"positions": [{"symbol": "A"}, {"symbol": "B"}]}
        assert count_active(s) == 2

    def test_add_position_appends(self):
        from live.state import add_position
        s   = {"positions": []}
        pos = {"symbol": "ETHUSDT", "entry_price": 3000.0}
        new_s = add_position(s, pos)
        assert len(new_s["positions"]) == 1
        assert new_s["positions"][0]["entry_price"] == 3000.0


# ─── ledger.py 測試 ───────────────────────────────────────────────────────────

class TestLedger:

    def test_append_entry_creates_file(self, tmp_path):
        from live.ledger import append_entry, load_ledger
        lf = tmp_path / "ledger.json"
        append_entry({"symbol": "ETHUSDT", "status": "open"}, ledger_file=lf)
        records = load_ledger(ledger_file=lf)
        assert len(records) == 1
        assert records[0]["symbol"] == "ETHUSDT"

    def test_multiple_entries_accumulate(self, tmp_path):
        from live.ledger import append_entry, load_ledger
        lf = tmp_path / "ledger.json"
        append_entry({"id": 1}, ledger_file=lf)
        append_entry({"id": 2}, ledger_file=lf)
        records = load_ledger(ledger_file=lf)
        assert len(records) == 2

    def test_load_returns_empty_list_when_no_file(self, tmp_path):
        from live.ledger import load_ledger
        records = load_ledger(ledger_file=tmp_path / "nonexistent.json")
        assert records == []
```

- [ ] **Step 2: 確認 8 個測試均 FAIL**

```
pytest tests/test_live_state.py -v 2>&1 | head -20
```

預期：`ModuleNotFoundError: No module named 'live.state'`

- [ ] **Step 3: 實作 live/state.py**

```python
# live/state.py
import json
import os
from pathlib import Path
import pandas as pd
import config

STATE_FILE = config.STORAGE_LIVE / "active_positions.json"


def load_state(state_file: Path = None) -> dict:
    f = Path(state_file) if state_file else STATE_FILE
    if not f.exists():
        return {"positions": []}
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def save_state(state: dict, state_file: Path = None) -> None:
    f = Path(state_file) if state_file else STATE_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    tmp = f.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2)
    os.replace(tmp, f)   # atomic rename — Crash Recovery


def count_active(state: dict) -> int:
    return len(state["positions"])


def add_position(state: dict, position: dict) -> dict:
    state["positions"].append(position)
    return state


def expire_closed_positions(state: dict, now: pd.Timestamp) -> tuple:
    """Remove positions whose exit_time <= now. Returns (new_state, expired_list)."""
    expired = [p for p in state["positions"] if pd.Timestamp(p["exit_time"]) <= now]
    kept    = [p for p in state["positions"] if pd.Timestamp(p["exit_time"]) >  now]
    return {"positions": kept}, expired
```

- [ ] **Step 4: 實作 live/ledger.py**

```python
# live/ledger.py
import json
from pathlib import Path
import config

LEDGER_FILE = config.STORAGE_LIVE / "paper_trading_ledger.json"


def load_ledger(ledger_file: Path = None) -> list:
    f = Path(ledger_file) if ledger_file else LEDGER_FILE
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def append_entry(trade: dict, ledger_file: Path = None) -> None:
    f = Path(ledger_file) if ledger_file else LEDGER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    records = load_ledger(ledger_file=f)
    records.append(trade)
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
```

- [ ] **Step 5: 確認 8 個測試全部 PASS**

```
pytest tests/test_live_state.py -v
```

預期：8 tests PASSED

- [ ] **Step 6: 完整測試套件確認無回歸**

```
pytest tests/ -v 2>&1 | tail -5
```

預期：全部 PASSED

- [ ] **Step 7: Commit**

```bash
git add tests/test_live_state.py live/state.py live/ledger.py
git commit -m "feat: add live/state.py and live/ledger.py with JSON crash-recovery (TDD)"
```

---

### Task 4: 實作 live/pipeline.py

**Files:**
- Create: `live/pipeline.py`

> 注意：pipeline 呼叫真實模型 + 特徵管線，不適合單元測試（需要真實資料）。本 Task 以 smoke test 代替 TDD。

- [ ] **Step 1: 實作 live/pipeline.py**

```python
# live/pipeline.py
import json
import logging
import joblib
import numpy as np
import pandas as pd

import config
from data import cleaner
from features import indicators
from live.fetcher import fetch_latest

logger = logging.getLogger(__name__)

HOURLY_LOOKBACK = 300   # SMA_200 warmup (200) + buffer
DAILY_LOOKBACK  = 220   # daily_ma_bias_200 warmup (200) + buffer


def load_assets(symbol: str, target: str) -> tuple:
    """Load feature_cols list and 5 fold models from storage."""
    report_path = config.STORAGE_FEATURES / f"{symbol}_validation_report.json"
    with open(report_path, encoding="utf-8") as f:
        feature_cols = json.load(f)["metadata"]["feature_columns"]
    fold_models = [
        joblib.load(config.STORAGE_MODELS / f"{symbol}_{target}_fold{k}.pkl")
        for k in range(1, 6)
    ]
    return feature_cols, fold_models


def compute_signal(
    symbol: str,
    feature_cols: list,
    fold_models: list,
    optimal_threshold: float,
    hourly_df: pd.DataFrame = None,
    daily_df: pd.DataFrame = None,
    btc_hourly_df: pd.DataFrame = None,
) -> dict:
    """
    Fetch live data (or accept injected DataFrames for testing), compute features,
    run ensemble inference, and return signal dict.

    Returns:
        {
          "symbol":      str,
          "timestamp":   str (ISO 8601 UTC),
          "close":       float,
          "atr_14":      float,
          "probability": float,
          "signal":      bool,
        }
    """
    if hourly_df is None:
        hourly_df = fetch_latest(symbol, "60", HOURLY_LOOKBACK)
    if daily_df is None:
        daily_df = fetch_latest(symbol, "D", DAILY_LOOKBACK)

    # cross_roc_24 = BTC ROC_24，ETHUSDT 模型需要 BTCUSDT hourly 作為 ref_df
    ref_df = None
    if symbol != "BTCUSDT":
        if btc_hourly_df is None:
            btc_hourly_df = fetch_latest("BTCUSDT", "60", HOURLY_LOOKBACK)
        ref_df = btc_hourly_df[["timestamp", "close"]].rename(columns={"close": "ref_close"})

    # Feature pipeline（繞過 build() 的檔案 I/O，直接呼叫底層函式）
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)
    df = indicators.compute(df, daily_df, ref_df=ref_df)
    df = df.dropna(subset=feature_cols)

    if df.empty:
        raise ValueError(f"[{symbol}] No valid rows after feature computation — not enough history?")

    last_row = df.iloc[[-1]]           # keep as DataFrame (shape 1×N) for predict_proba
    X = last_row[feature_cols]

    proba = float(np.mean([m.predict_proba(X)[0, 1] for m in fold_models]))

    return {
        "symbol":      symbol,
        "timestamp":   last_row["timestamp"].iloc[0].isoformat(),
        "close":       float(last_row["close"].iloc[0]),
        "atr_14":      float(last_row["atr_14"].iloc[0]),
        "probability": round(proba, 4),
        "signal":      proba >= optimal_threshold,
    }
```

- [ ] **Step 2: 語法確認**

```
python -m py_compile live/pipeline.py && echo "OK"
```

預期：`OK`

- [ ] **Step 3: Smoke test — 確認可匯入並呼叫 load_assets**

```python
python -c "
from live.pipeline import load_assets
feature_cols, fold_models = load_assets('ETHUSDT', 'target_atr')
print(f'feature_cols: {len(feature_cols)} columns')
print(f'fold_models:  {len(fold_models)} models')
assert len(feature_cols) == 27
assert len(fold_models) == 5
print('OK')
"
```

預期：
```
feature_cols: 27 columns
fold_models:  5 models
OK
```

- [ ] **Step 4: Commit**

```bash
git add live/pipeline.py
git commit -m "feat: add live/pipeline.py for feature reconstruction and ensemble inference"
```

---

### Task 5: 實作 live/notifier.py

**Files:**
- Create: `live/notifier.py`

- [ ] **Step 1: 實作 live/notifier.py**

```python
# live/notifier.py
import logging
import requests
import config

logger = logging.getLogger(__name__)


def send(message: str) -> None:
    """POST message to Discord Webhook. No-op if DISCORD_WEBHOOK_URL not configured."""
    url = config.DISCORD_WEBHOOK_URL
    if not url:
        logger.warning("DISCORD_WEBHOOK_URL not set — skipping notification")
        return
    try:
        resp = requests.post(url, json={"content": message}, timeout=5)
        resp.raise_for_status()
    except Exception as e:
        logger.error(f"Discord notification failed: {e}")
```

- [ ] **Step 2: 語法確認**

```
python -m py_compile live/notifier.py && echo "OK"
```

預期：`OK`

- [ ] **Step 3: Commit**

```bash
git add live/notifier.py
git commit -m "feat: add live/notifier.py Discord Webhook sender"
```

---

### Task 6: 實作 run_live.py 心跳迴圈

**Files:**
- Create: `run_live.py`

- [ ] **Step 1: 實作 run_live.py**

```python
# run_live.py
import json
import logging
import schedule
import time
import pandas as pd

import config
from live import pipeline, state, ledger, notifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_CONCURRENT = 3
HOLDING_BARS   = 24       # 24 小時後 timeout
RISK_PCT       = 0.02
INITIAL_EQUITY = 1_000_000

_threshold_cache: dict = {}


def _load_threshold(symbol: str, target: str) -> float:
    key = f"{symbol}_{target}"
    if key not in _threshold_cache:
        path = config.STORAGE_BACKTEST / f"{symbol}_{target}_threshold_scan.json"
        with open(path, encoding="utf-8") as f:
            _threshold_cache[key] = json.load(f)["optimal_threshold"]
    return _threshold_cache[key]


def _sl_tp(close: float, atr_14: float) -> tuple:
    """target_atr: SL = close − 1.5×ATR, TP = close + 2.0×ATR"""
    return close - 1.5 * atr_14, close + 2.0 * atr_14


def heartbeat() -> None:
    now = pd.Timestamp.utcnow()
    logger.info(f"[heartbeat] {now.isoformat()}")

    current_state = state.load_state()

    # ── 1. 到期平倉 ──────────────────────────────────────────────────
    current_state, expired = state.expire_closed_positions(current_state, now)
    for pos in expired:
        ledger.append_entry({**pos, "outcome": "timeout", "exit_time_actual": now.isoformat()})
        notifier.send(
            f"[BYBIT_ML] 📋 **{pos['symbol']} 倉位到期**\n"
            f"進場：{pos['entry_price']:.4f} @ {pos['entry_time']}\n"
            f"結果：Timeout（{HOLDING_BARS} 小時）"
        )
        logger.info(f"Expired: {pos['symbol']} @ {pos['entry_price']}")

    # ── 2. 檢查訊號 ──────────────────────────────────────────────────
    for symbol in config.LIVE_SYMBOLS:
        n_active = state.count_active(current_state)
        if n_active >= MAX_CONCURRENT:
            logger.info(f"[{symbol}] Concurrent limit ({n_active}/{MAX_CONCURRENT}), skip")
            continue

        target    = config.LIVE_TARGET
        threshold = _load_threshold(symbol, target)
        feature_cols, fold_models = pipeline.load_assets(symbol, target)

        try:
            result = pipeline.compute_signal(symbol, feature_cols, fold_models, threshold)
        except Exception as e:
            logger.error(f"[{symbol}] Signal failed: {e}")
            continue

        logger.info(f"[{symbol}] prob={result['probability']:.4f} signal={result['signal']}")

        if not result["signal"]:
            continue

        sl, tp     = _sl_tp(result["close"], result["atr_14"])
        sl_dist    = result["close"] - sl
        pos_qty    = (INITIAL_EQUITY * RISK_PCT) / sl_dist if sl_dist > 0 else 0
        pos_usd    = pos_qty * result["close"]
        exit_time  = (pd.Timestamp(result["timestamp"]) + pd.Timedelta(hours=HOLDING_BARS)).isoformat()

        position = {
            "symbol":       symbol,
            "target":       target,
            "entry_time":   result["timestamp"],
            "entry_price":  result["close"],
            "sl_price":     round(sl, 4),
            "tp_price":     round(tp, 4),
            "atr_14":       result["atr_14"],
            "probability":  result["probability"],
            "position_usd": round(pos_usd, 2),
            "exit_time":    exit_time,
        }

        current_state = state.add_position(current_state, position)
        ledger.append_entry({**position, "status": "open"})
        notifier.send(
            f"[BYBIT_ML] 🚀 **{symbol} 買入訊號**\n"
            f"機率：{result['probability']:.4f} > {threshold}\n"
            f"進場價：{result['close']:,.4f}\n"
            f"SL：{sl:,.4f}  |  TP：{tp:,.4f}\n"
            f"虛擬部位：${pos_usd:,.0f} USD\n"
            f"預計出場：{exit_time}"
        )
        logger.info(f"[{symbol}] Signal! prob={result['probability']:.4f}, pos=${pos_usd:,.0f}")

    state.save_state(current_state)
    logger.info("[heartbeat] Done")


def main() -> None:
    logger.info("Live signal daemon starting — running once immediately...")
    heartbeat()
    schedule.every().hour.at(":01").do(heartbeat)
    logger.info("Scheduled: every hour at :01. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: 語法確認**

```
python -m py_compile run_live.py && echo "OK"
```

預期：`OK`

- [ ] **Step 3: 完整測試套件確認無回歸**

```
pytest tests/ -v 2>&1 | tail -5
```

預期：全部 PASSED

- [ ] **Step 4: Commit**

```bash
git add run_live.py
git commit -m "feat: add run_live.py heartbeat loop with schedule (Phase 5 entry point)"
```

---

### Task 7: 端對端 Smoke Test（真實 API）

**Files:**
- 執行：`run_live.py`（單次 heartbeat）
- 確認：`storage/live/` 的輸出

> 此 Task 需要網路連線（呼叫 Bybit 公開 API）。

- [ ] **Step 1: 設定 Discord Webhook URL（若尚未設定）**

在 `.env` 檔案（或環境變數）中加入：

```
DISCORD_WEBHOOK_URL=https://discord.com/api/webhooks/YOUR_WEBHOOK_ID/YOUR_WEBHOOK_TOKEN
```

取得方式：Discord 伺服器 → 頻道設定 → 整合 → Webhook → 複製 URL

若暫不設定，notifier 會 log warning 但不影響其他邏輯。

- [ ] **Step 2: 執行一次性 heartbeat（不啟動 schedule 迴圈）**

```python
python -c "
import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
from run_live import heartbeat
heartbeat()
"
```

預期 log（格式）：
```
2026-05-31 HH:MM:SS INFO [heartbeat] 2026-05-31T...
2026-05-31 HH:MM:SS INFO [ETHUSDT] prob=0.XXXX signal=True/False
2026-05-31 HH:MM:SS INFO [heartbeat] Done
```

- [ ] **Step 3: 確認 storage/live/ 輸出**

```python
python -c "
import json
from pathlib import Path

state_f  = Path('storage/live/active_positions.json')
ledger_f = Path('storage/live/paper_trading_ledger.json')

print('active_positions.json exists:', state_f.exists())
print('paper_trading_ledger.json exists:', ledger_f.exists())

if state_f.exists():
    s = json.loads(state_f.read_text())
    print(f'Active positions: {len(s[\"positions\"])}')

if ledger_f.exists():
    l = json.loads(ledger_f.read_text())
    print(f'Ledger entries: {len(l)}')
"
```

- [ ] **Step 4: 最終完整測試套件**

```
pytest tests/ -v 2>&1 | tail -5
```

預期：全部 PASSED

- [ ] **Step 5: Commit**

```bash
git add docs/superpowers/specs/2026-05-31-phase5-live-signal-design.md
git add docs/superpowers/plans/2026-05-31-phase5-live-signal.md
git commit -m "docs: add Phase 5 live signal design spec and implementation plan"
```

---

## 架構備忘

```
heartbeat() 每小時 :01 執行
    ├─ state.expire_closed_positions()  → 到期平倉
    ├─ for symbol in LIVE_SYMBOLS:
    │   ├─ count_active() >= 3 → skip
    │   ├─ pipeline.compute_signal()   → ETH hourly(300) + daily(220) + BTC hourly(300)
    │   │   └─ indicators.compute(df, daily_df, ref_df=btc_ref)  [cross_roc_24 需要 BTC]
    │   ├─ prob >= 0.75 → 計算 DRC 部位大小
    │   ├─ state.add_position()
    │   ├─ ledger.append_entry()
    │   └─ notifier.send()             → Discord Webhook
    └─ state.save_state()              → 原子寫入 active_positions.json
```

**未來擴充路徑（不在本 Phase 範圍）：**
- 加入 BTCUSDT（待特徵改善後）
- 真實下單：將 `ledger.append_entry()` 換成 `bybit.place_order()`
- SL/TP 即時監控：加入 WebSocket 訂閱替代 timeout 機制
