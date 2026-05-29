# Phase 1 資料抓取與時間對齊 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 建立獨立的 BYBIT_ML 專案，從 Bybit 抓取 BTCUSDT/ETHUSDT 4.5 年雙週期（1H+1D）歷史資料，清洗對齊（無 look-ahead bias），存為 Parquet + Excel，並支援每日增量更新。

**Architecture:** 四個職責單一的模組（fetcher / cleaner / exporter / scheduler），各自可獨立測試。fetcher 做 forward pagination 分頁抓取；cleaner 做去重、補缺、日線向後位移一天再 merge_asof；exporter 做 Parquet append（舊前新後 keep=last）與 Excel 重寫；scheduler 從最後時間戳往回 3 根做重疊覆寫。

**Tech Stack:** Python 3.12, pybit>=5.0.0, pandas>=2.0.0, pyarrow>=14.0.0, openpyxl>=3.1.0, python-dotenv, pytest>=7.0.0, numpy<2

---

## File Map

| 檔案 | 動作 | 職責 |
|------|------|------|
| `config.py` | 建立 | 全域設定（幣對、路徑、重試、overlap） |
| `pytest.ini` | 建立 | 設定 pythonpath，讓 tests 找到 data/ |
| `requirements.txt` | 建立 | 套件版本 |
| `.gitignore` | 建立 | 排除 .env、storage/、__pycache__ |
| `.env.example` | 建立 | API 金鑰範本 |
| `data/__init__.py` | 建立 | 空檔，讓 data/ 成為 package |
| `data/fetcher.py` | 建立 | Bybit API forward pagination，含 turnover |
| `data/cleaner.py` | 建立 | 去重、補缺、look-ahead bias 防護 |
| `data/exporter.py` | 建立 | Parquet append + Excel 重寫 |
| `main.py` | 建立 | 歷史全量拉取入口 |
| `scheduler.py` | 建立 | 每日增量更新（overlap 3 根） |
| `tests/__init__.py` | 建立 | 空檔 |
| `tests/test_fetcher.py` | 建立 | Mock API，測試分頁、retry、欄位 |
| `tests/test_cleaner.py` | 建立 | 去重、補缺、look-ahead bias 正確性 |
| `tests/test_exporter.py` | 建立 | Parquet append、Excel 工作表結構 |

---

## Task 1：專案初始化

**Files:**
- Create: `config.py`
- Create: `pytest.ini`
- Create: `requirements.txt`
- Create: `.gitignore`
- Create: `.env.example`
- Create: `data/__init__.py`
- Create: `tests/__init__.py`

- [ ] **Step 1：建立目錄結構**

```bash
cd /e/93050207/python/BYBIT_ML
mkdir -p data tests storage/raw storage/excel
```

- [ ] **Step 2：建立 `requirements.txt`**

```
pybit>=5.0.0
pandas>=2.0.0
pyarrow>=14.0.0
openpyxl>=3.1.0
python-dotenv>=1.0.0
pytest>=7.0.0
numpy<2
```

- [ ] **Step 3：建立 `pytest.ini`**

```ini
[pytest]
pythonpath = .
```

- [ ] **Step 4：建立 `.gitignore`**

```
.env
storage/
__pycache__/
*.pyc
.pytest_cache/
logs/
```

- [ ] **Step 5：建立 `.env.example`**

```
BYBIT_API_KEY=your_api_key_here
BYBIT_API_SECRET=your_api_secret_here
```

- [ ] **Step 6：建立 `config.py`**

```python
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["60", "D"]

INTERVAL_LABELS = {
    "60": "1h",
    "D": "1d",
}

HISTORY_START = "2022-01-01"

BASE_DIR = Path(__file__).parent
STORAGE_RAW = BASE_DIR / "storage" / "raw"
STORAGE_EXCEL = BASE_DIR / "storage" / "excel"

OVERLAP_HOURS = 3
OVERLAP_DAYS = 3
RATE_LIMIT_SLEEP = 0.2
MAX_RETRIES = 3
```

- [ ] **Step 7：建立空的 `data/__init__.py` 與 `tests/__init__.py`**

```bash
touch data/__init__.py tests/__init__.py
```

（Windows 可用 `echo. > data/__init__.py && echo. > tests/__init__.py`）

- [ ] **Step 8：安裝套件**

```bash
pip install -r requirements.txt
```

Expected：無 error，所有套件安裝完成。

- [ ] **Step 9：Commit**

```bash
git add config.py pytest.ini requirements.txt .gitignore .env.example data/__init__.py tests/__init__.py
git commit -m "chore: project setup for Phase 1 data pipeline"
```

---

## Task 2：fetcher.py（TDD）

**Files:**
- Create: `tests/test_fetcher.py`
- Create: `data/fetcher.py`

- [ ] **Step 1：寫失敗的測試 `tests/test_fetcher.py`**

```python
# tests/test_fetcher.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import timezone


def _make_rows(base_ms: int, count: int, interval_ms: int = 3_600_000):
    """建立 count 根 K 線，newest first（模擬 Bybit 回傳順序）"""
    rows = []
    for i in range(count - 1, -1, -1):
        ts = base_ms + i * interval_ms
        rows.append([str(ts), "100.0", "101.0", "99.0", "100.5", "10.0", "1005.0"])
    return rows


BASE_MS = 1_640_995_200_000  # 2022-01-01 00:00 UTC


class TestFetchOhlcv:
    def test_returns_correct_columns(self):
        """回傳包含 7 個正確欄位的 DataFrame"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            {"result": {"list": _make_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        assert list(df.columns) == [
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ]

    def test_timestamp_is_utc(self):
        """timestamp 欄位為 UTC 時區"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            {"result": {"list": _make_rows(BASE_MS, 1)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        assert df["timestamp"].dt.tz == timezone.utc

    def test_numeric_columns_are_float64(self):
        """數值欄位為 float64"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            {"result": {"list": _make_rows(BASE_MS, 2)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        for col in ["open", "high", "low", "close", "volume", "turnover"]:
            assert df[col].dtype == "float64", f"{col} 應為 float64"

    def test_pagination_combines_batches(self):
        """分頁時合併多批資料，總筆數正確"""
        from data.fetcher import fetch_ohlcv

        batch1 = _make_rows(BASE_MS, 3)
        batch2 = _make_rows(BASE_MS + 3 * 3_600_000, 2)

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            {"result": {"list": batch1}},
            {"result": {"list": batch2}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-12-31")

        assert len(df) == 5

    def test_data_in_chronological_order(self):
        """回傳資料為時間正序（由舊到新）"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            {"result": {"list": _make_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_empty_response_returns_empty_dataframe(self):
        """API 無資料時回傳空 DataFrame（含正確欄位）"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.return_value = {"result": {"list": []}}
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        assert df.empty
        assert list(df.columns) == [
            "timestamp", "open", "high", "low", "close", "volume", "turnover"
        ]

    def test_retry_on_api_error(self):
        """API 錯誤時自動重試，最終成功"""
        from data.fetcher import fetch_ohlcv

        mock_session = MagicMock()
        mock_session.get_kline.side_effect = [
            Exception("network error"),
            Exception("network error"),
            {"result": {"list": _make_rows(BASE_MS, 1)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            with patch("data.fetcher.time.sleep"):
                df = fetch_ohlcv("BTCUSDT", "60", "2022-01-01", "2022-01-02")

        assert len(df) == 1
```

- [ ] **Step 2：確認測試失敗**

```bash
cd /e/93050207/python/BYBIT_ML
pytest tests/test_fetcher.py -v
```

Expected：`ImportError: cannot import name 'fetch_ohlcv' from 'data.fetcher'`（或 ModuleNotFoundError）

- [ ] **Step 3：實作 `data/fetcher.py`**

```python
# data/fetcher.py
import time
import logging
from datetime import datetime, timezone

import pandas as pd
from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    session = HTTP(testnet=False, api_key=config.API_KEY, api_secret=config.API_SECRET)
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches = []
    current_start = start_dt

    while current_start < end_dt:
        rows = _fetch_batch(session, symbol, interval, _to_ms(current_start))
        if not rows:
            break

        rows = list(reversed(rows))  # Bybit returns newest first; reverse to chronological
        batch_df = _parse_rows(rows)
        batch_df = batch_df[batch_df["timestamp"] <= end_dt]

        if batch_df.empty:
            break

        all_batches.append(batch_df)
        last_ts_ms = int(batch_df["timestamp"].iloc[-1].timestamp() * 1000)
        current_start = _from_ms(last_ts_ms + 1)
        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(all_batches, ignore_index=True)


def _fetch_batch(session, symbol: str, interval: str, start_ms: int) -> list:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval,
                start=start_ms,
                limit=200,
            )
            return resp["result"]["list"]
        except Exception as exc:
            if attempt == config.MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"API error (attempt {attempt + 1}/{config.MAX_RETRIES}), retry in {wait}s: {exc}")
            time.sleep(wait)
    return []


def _parse_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype(int), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df


def _parse_date(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
```

- [ ] **Step 4：確認測試全通過**

```bash
pytest tests/test_fetcher.py -v
```

Expected：
```
tests/test_fetcher.py::TestFetchOhlcv::test_returns_correct_columns PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_timestamp_is_utc PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_numeric_columns_are_float64 PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_pagination_combines_batches PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_data_in_chronological_order PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_empty_response_returns_empty_dataframe PASSED
tests/test_fetcher.py::TestFetchOhlcv::test_retry_on_api_error PASSED
7 passed
```

- [ ] **Step 5：Commit**

```bash
git add data/fetcher.py tests/test_fetcher.py
git commit -m "feat: add fetcher with forward pagination and retry"
```

---

## Task 3：cleaner.py（TDD）

**Files:**
- Create: `tests/test_cleaner.py`
- Create: `data/cleaner.py`

- [ ] **Step 1：寫失敗的測試 `tests/test_cleaner.py`**

```python
# tests/test_cleaner.py
import pytest
import pandas as pd
from datetime import timezone


def _make_hourly(start: str, periods: int, close: float = 100.5) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":  [100.0] * periods,
        "high":  [101.0] * periods,
        "low":   [99.0]  * periods,
        "close": [close] * periods,
        "volume":   [10.0]   * periods,
        "turnover": [1005.0] * periods,
    })


def _make_daily(start: str, periods: int, close: float = 102.0) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1D", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":  [100.0] * periods,
        "high":  [105.0] * periods,
        "low":   [95.0]  * periods,
        "close": [close] * periods,
        "volume":   [1000.0]   * periods,
        "turnover": [102000.0] * periods,
    })


class TestClean:
    def test_removes_duplicates_keeps_last(self):
        """重複 timestamp 保留最後一筆（最新、已收盤版本）"""
        from data.cleaner import clean

        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        df = pd.DataFrame({
            "timestamp": [ts, ts],
            "open":  [100.0, 200.0],
            "high":  [101.0, 201.0],
            "low":   [99.0,  199.0],
            "close": [100.5, 200.5],
            "volume":   [5.0,  20.0],
            "turnover": [502.5, 2005.0],
        })

        result = clean(df, "60")

        assert len(result) == 1
        assert result.iloc[0]["close"] == 200.5

    def test_fills_missing_ohlc_with_forward_fill(self):
        """缺失 K 線的 OHLC 用 forward fill 補（延續上根收盤價）"""
        from data.cleaner import clean

        ts1 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        ts3 = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
        df = pd.DataFrame({
            "timestamp": [ts1, ts3],
            "open":  [100.0, 102.0],
            "high":  [101.0, 103.0],
            "low":   [99.0,  101.0],
            "close": [100.5, 102.5],
            "volume":   [10.0, 12.0],
            "turnover": [1005.0, 1225.0],
        })

        result = clean(df, "60")
        missing = result[result["timestamp"] == pd.Timestamp("2024-01-01 01:00:00", tz="UTC")]

        assert len(missing) == 1
        assert missing.iloc[0]["close"] == 100.5

    def test_fills_missing_volume_with_zero(self):
        """缺失 K 線的 volume/turnover 填 0（無交易發生）"""
        from data.cleaner import clean

        ts1 = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")
        ts3 = pd.Timestamp("2024-01-01 02:00:00", tz="UTC")
        df = pd.DataFrame({
            "timestamp": [ts1, ts3],
            "open":  [100.0, 102.0],
            "high":  [101.0, 103.0],
            "low":   [99.0,  101.0],
            "close": [100.5, 102.5],
            "volume":   [10.0, 12.0],
            "turnover": [1005.0, 1225.0],
        })

        result = clean(df, "60")
        missing = result[result["timestamp"] == pd.Timestamp("2024-01-01 01:00:00", tz="UTC")]

        assert missing.iloc[0]["volume"] == 0.0
        assert missing.iloc[0]["turnover"] == 0.0

    def test_timestamp_dtype_is_utc(self):
        """清洗後 timestamp 為 UTC DatetimeTZ 型別"""
        from data.cleaner import clean

        result = clean(_make_hourly("2024-01-01", 5), "60")

        assert result["timestamp"].dt.tz == timezone.utc

    def test_result_sorted_by_timestamp(self):
        """清洗後資料按 timestamp 升序排列"""
        from data.cleaner import clean

        result = clean(_make_hourly("2024-01-01", 5), "60")
        timestamps = result["timestamp"].tolist()

        assert timestamps == sorted(timestamps)


class TestAlignDailyToHourly:
    def test_daily_not_available_same_day(self):
        """當天日線特徵不能貼到當天小時線（look-ahead bias 防護）"""
        from data.cleaner import align_daily_to_hourly

        # 日線 2024-01-01，close=102.0；date_available=2024-01-02
        daily_df = _make_daily("2024-01-01", periods=1, close=102.0)
        # 小時線只覆蓋 2024-01-01
        hourly_df = _make_hourly("2024-01-01 00:00", periods=8)

        result = align_daily_to_hourly(hourly_df, daily_df)

        # 2024-01-01 的所有小時線不應有日線資料
        assert result["daily_close"].isna().all()

    def test_daily_available_from_next_day(self):
        """日線特徵從次日 00:00 起才貼到小時線"""
        from data.cleaner import align_daily_to_hourly

        daily_df = _make_daily("2024-01-01", periods=1, close=102.0)
        # 小時線覆蓋 2024-01-01 + 2024-01-02（48 小時）
        hourly_df = _make_hourly("2024-01-01 00:00", periods=48)

        result = align_daily_to_hourly(hourly_df, daily_df)

        jan2 = result[result["timestamp"] >= pd.Timestamp("2024-01-02 00:00:00", tz="UTC")]
        assert (jan2["daily_close"] == 102.0).all()

    def test_output_has_daily_prefix_columns(self):
        """對齊後 DataFrame 含 daily_open/high/low/close/volume/turnover 欄位"""
        from data.cleaner import align_daily_to_hourly

        daily_df = _make_daily("2024-01-01", periods=3)
        hourly_df = _make_hourly("2024-01-01 00:00", periods=72)

        result = align_daily_to_hourly(hourly_df, daily_df)

        for col in ["daily_open", "daily_high", "daily_low", "daily_close",
                    "daily_volume", "daily_turnover"]:
            assert col in result.columns, f"缺少欄位: {col}"
```

- [ ] **Step 2：確認測試失敗**

```bash
pytest tests/test_cleaner.py -v
```

Expected：`ImportError: cannot import name 'clean' from 'data.cleaner'`

- [ ] **Step 3：實作 `data/cleaner.py`**

```python
# data/cleaner.py
import logging

import pandas as pd

logger = logging.getLogger(__name__)

INTERVAL_TO_FREQ = {
    "60": "1h",
    "D":  "1D",
}
OHLC_COLS   = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "turnover"]
DAILY_COLS  = ["open", "high", "low", "close", "volume", "turnover"]


def clean(df: pd.DataFrame, interval: str, label: str = "") -> pd.DataFrame:
    original_len = len(df)

    # 1. 去重，保留最新版本（已收盤資料）
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    dupes_removed = original_len - len(df)

    # 2. 確保 timestamp 有 UTC 時區
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")

    # 3. 補缺失 K 線
    freq = INTERVAL_TO_FREQ[interval]
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=freq,
        tz="UTC",
    )
    df_indexed = df.set_index("timestamp").reindex(full_index)
    df_indexed.index.name = "timestamp"
    filled = int(df_indexed[OHLC_COLS].isna().any(axis=1).sum())

    df_indexed[OHLC_COLS]   = df_indexed[OHLC_COLS].ffill()
    df_indexed[VOLUME_COLS] = df_indexed[VOLUME_COLS].fillna(0.0)

    df = df_indexed.reset_index()

    tag = label or interval
    logger.info(f"[{tag}] 總筆數: {len(df)} | 補缺: {filled} 根 | 重複移除: {dupes_removed}")
    return df


def align_daily_to_hourly(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    daily = daily_df.copy()

    # 日線向後位移一天，防止 look-ahead bias
    daily["date_available"] = daily["timestamp"] + pd.Timedelta(days=1)

    # 日線欄位加 daily_ 前綴，避免與小時線欄位衝突
    daily = daily.rename(columns={col: f"daily_{col}" for col in DAILY_COLS})

    keep_cols = ["date_available"] + [f"daily_{c}" for c in DAILY_COLS]

    merged = pd.merge_asof(
        hourly_df.sort_values("timestamp"),
        daily[keep_cols].sort_values("date_available"),
        left_on="timestamp",
        right_on="date_available",
        direction="backward",
    )
    return merged.drop(columns=["date_available"], errors="ignore")
```

- [ ] **Step 4：確認測試全通過**

```bash
pytest tests/test_cleaner.py -v
```

Expected：
```
tests/test_cleaner.py::TestClean::test_removes_duplicates_keeps_last PASSED
tests/test_cleaner.py::TestClean::test_fills_missing_ohlc_with_forward_fill PASSED
tests/test_cleaner.py::TestClean::test_fills_missing_volume_with_zero PASSED
tests/test_cleaner.py::TestClean::test_timestamp_dtype_is_utc PASSED
tests/test_cleaner.py::TestClean::test_result_sorted_by_timestamp PASSED
tests/test_cleaner.py::TestAlignDailyToHourly::test_daily_not_available_same_day PASSED
tests/test_cleaner.py::TestAlignDailyToHourly::test_daily_available_from_next_day PASSED
tests/test_cleaner.py::TestAlignDailyToHourly::test_output_has_daily_prefix_columns PASSED
8 passed
```

- [ ] **Step 5：Commit**

```bash
git add data/cleaner.py tests/test_cleaner.py
git commit -m "feat: add cleaner with dedup, gap-fill, and look-ahead bias prevention"
```

---

## Task 4：exporter.py（TDD）

**Files:**
- Create: `tests/test_exporter.py`
- Create: `data/exporter.py`

- [ ] **Step 1：寫失敗的測試 `tests/test_exporter.py`**

```python
# tests/test_exporter.py
import pytest
import pandas as pd
from pathlib import Path


def _make_df(start: str = "2024-01-01", periods: int = 5) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":  [100.0] * periods,
        "high":  [101.0] * periods,
        "low":   [99.0]  * periods,
        "close": [100.5] * periods,
        "volume":   [10.0]   * periods,
        "turnover": [1005.0] * periods,
    })


class TestAppendParquet:
    def test_creates_file_if_not_exists(self, tmp_path):
        """Parquet 不存在時建立新檔"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        append_parquet(path, _make_df())

        assert path.exists()
        assert len(pd.read_parquet(path)) == 5

    def test_appends_non_overlapping_rows(self, tmp_path):
        """無重疊時 append 後總筆數 = 舊 + 新"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        df1 = _make_df("2024-01-01 00:00", periods=5)
        df2 = _make_df("2024-01-01 05:00", periods=5)

        append_parquet(path, df1)
        append_parquet(path, df2)

        assert len(pd.read_parquet(path)) == 10

    def test_overlap_keeps_latest_version(self, tmp_path):
        """重疊 timestamp 保留最新版本（已收盤價格覆蓋舊的未收盤價格）"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

        df_old = pd.DataFrame({
            "timestamp": [ts],
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5],      # 未收盤時抓到的臨時價格
            "volume": [5.0], "turnover": [502.5],
        })
        df_new = pd.DataFrame({
            "timestamp": [ts],
            "open": [100.0], "high": [102.0], "low": [98.0],
            "close": [101.5],      # 收盤後的最終價格
            "volume": [10.0], "turnover": [1015.0],
        })

        append_parquet(path, df_old)
        append_parquet(path, df_new)

        result = pd.read_parquet(path)
        assert len(result) == 1
        assert result.iloc[0]["close"] == 101.5

    def test_result_sorted_by_timestamp(self, tmp_path):
        """儲存後 Parquet 按 timestamp 升序排列"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        append_parquet(path, _make_df("2024-01-01", periods=5))

        result = pd.read_parquet(path)
        diffs = result["timestamp"].diff().dropna()
        assert (diffs > pd.Timedelta(0)).all()


class TestWriteExcel:
    def test_creates_two_sheets(self, tmp_path):
        """Excel 包含 1H 和 1D 兩個工作表"""
        from data.exporter import write_excel

        path = tmp_path / "BTCUSDT.xlsx"
        write_excel(path, _make_df(periods=10), _make_df(periods=3))

        xl = pd.ExcelFile(path)
        assert "1H" in xl.sheet_names
        assert "1D" in xl.sheet_names

    def test_sheet_row_counts_match_data(self, tmp_path):
        """Excel 各工作表行數與輸入資料一致"""
        from data.exporter import write_excel

        path = tmp_path / "BTCUSDT.xlsx"
        write_excel(path, _make_df(periods=10), _make_df(periods=3))

        assert len(pd.read_excel(path, sheet_name="1H")) == 10
        assert len(pd.read_excel(path, sheet_name="1D")) == 3
```

- [ ] **Step 2：確認測試失敗**

```bash
pytest tests/test_exporter.py -v
```

Expected：`ImportError: cannot import name 'append_parquet' from 'data.exporter'`

- [ ] **Step 3：實作 `data/exporter.py`**

```python
# data/exporter.py
from pathlib import Path

import pandas as pd


def append_parquet(path: Path, new_df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old_df = pd.read_parquet(path)
        # 舊資料在前，新資料在後，keep="last" 保留新版本（已收盤價）
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df.sort_values("timestamp").reset_index(drop=True)

    combined.to_parquet(path, index=False)


def write_excel(
    path: Path,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        hourly_df.to_excel(writer, sheet_name="1H", index=False)
        daily_df.to_excel(writer, sheet_name="1D", index=False)

        for sheet_name in ("1H", "1D"):
            writer.sheets[sheet_name].freeze_panes = "A2"
```

- [ ] **Step 4：確認測試全通過**

```bash
pytest tests/test_exporter.py -v
```

Expected：
```
tests/test_exporter.py::TestAppendParquet::test_creates_file_if_not_exists PASSED
tests/test_exporter.py::TestAppendParquet::test_appends_non_overlapping_rows PASSED
tests/test_exporter.py::TestAppendParquet::test_overlap_keeps_latest_version PASSED
tests/test_exporter.py::TestAppendParquet::test_result_sorted_by_timestamp PASSED
tests/test_exporter.py::TestWriteExcel::test_creates_two_sheets PASSED
tests/test_exporter.py::TestWriteExcel::test_sheet_row_counts_match_data PASSED
6 passed
```

- [ ] **Step 5：Commit**

```bash
git add data/exporter.py tests/test_exporter.py
git commit -m "feat: add exporter with parquet append and excel writer"
```

---

## Task 5：main.py（歷史全量拉取）

**Files:**
- Create: `main.py`

- [ ] **Step 1：確認全部測試仍通過**

```bash
pytest tests/ -v
```

Expected：21 passed（7+8+6）

- [ ] **Step 2：建立 `main.py`**

```python
# main.py
import logging
from datetime import datetime, timezone

import config
from data import fetcher, cleaner, exporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_full_fetch() -> None:
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    config.STORAGE_RAW.mkdir(parents=True, exist_ok=True)
    config.STORAGE_EXCEL.mkdir(parents=True, exist_ok=True)

    for symbol in config.SYMBOLS:
        dfs: dict = {}

        for interval in config.INTERVALS:
            label = config.INTERVAL_LABELS[interval]
            logger.info(f"抓取 {symbol} {label} | {config.HISTORY_START} → {end_date}")

            df = fetcher.fetch_ohlcv(symbol, interval, config.HISTORY_START, end_date)
            df = cleaner.clean(df, interval, label=f"{symbol} {label}")

            parquet_path = config.STORAGE_RAW / f"{symbol}_{label}.parquet"
            exporter.append_parquet(parquet_path, df)
            logger.info(f"  → 儲存 {parquet_path}（{len(df):,} 筆）")

            dfs[interval] = df

        excel_path = config.STORAGE_EXCEL / f"{symbol}.xlsx"
        exporter.write_excel(excel_path, dfs["60"], dfs["D"])
        logger.info(f"  → Excel 儲存 {excel_path}")


if __name__ == "__main__":
    run_full_fetch()
```

- [ ] **Step 3：複製 `.env` 並確認 API 金鑰**

將現有 `E:\93050207\python\BYBIT\.env` 複製到 `E:\93050207\python\BYBIT_ML\.env`。

- [ ] **Step 4：Dry-run 確認 import 無誤**

```bash
cd /e/93050207/python/BYBIT_ML
python -c "import main; print('import OK')"
```

Expected：`import OK`（不實際執行 API 呼叫）

- [ ] **Step 5：Commit**

```bash
git add main.py
git commit -m "feat: add main.py for full historical fetch"
```

---

## Task 6：scheduler.py（每日增量更新）

**Files:**
- Create: `scheduler.py`

- [ ] **Step 1：建立 `scheduler.py`**

```python
# scheduler.py
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from data import fetcher, cleaner, exporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_last_timestamp(parquet_path) -> pd.Timestamp:
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    return df["timestamp"].max()


def _overlap_delta(interval: str) -> timedelta:
    return (
        timedelta(hours=config.OVERLAP_HOURS)
        if interval == "60"
        else timedelta(days=config.OVERLAP_DAYS)
    )


def incremental_update(symbol: str, interval: str) -> None:
    label = config.INTERVAL_LABELS[interval]
    parquet_path = config.STORAGE_RAW / f"{symbol}_{label}.parquet"

    if not parquet_path.exists():
        logger.error(
            f"{parquet_path} 不存在，請先執行 main.py 進行歷史全量拉取"
        )
        return

    last_ts = _get_last_timestamp(parquet_path)
    start = last_ts - _overlap_delta(interval)   # 往回推，覆蓋未收盤 K 線
    end = datetime.now(timezone.utc)

    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"增量更新 {symbol} {label} | {start_str} → {end_str}")
    df = fetcher.fetch_ohlcv(symbol, interval, start_str, end_str)
    if df.empty:
        logger.info(f"  → 無新資料")
        return

    df = cleaner.clean(df, interval, label=f"{symbol} {label}")
    exporter.append_parquet(parquet_path, df)
    logger.info(f"  → 更新完成（處理 {len(df)} 筆，重疊覆寫護城河啟動）")


def run() -> None:
    for symbol in config.SYMBOLS:
        dfs: dict = {}

        for interval in config.INTERVALS:
            incremental_update(symbol, interval)
            label = config.INTERVAL_LABELS[interval]
            dfs[interval] = pd.read_parquet(
                config.STORAGE_RAW / f"{symbol}_{label}.parquet"
            )

        excel_path = config.STORAGE_EXCEL / f"{symbol}.xlsx"
        exporter.write_excel(excel_path, dfs["60"], dfs["D"])
        logger.info(f"  → Excel 更新 {excel_path}")


if __name__ == "__main__":
    run()
```

- [ ] **Step 2：Dry-run 確認 import 無誤**

```bash
python -c "import scheduler; print('import OK')"
```

Expected：`import OK`

- [ ] **Step 3：確認全部測試仍通過**

```bash
pytest tests/ -v
```

Expected：21 passed

- [ ] **Step 4：Commit**

```bash
git add scheduler.py
git commit -m "feat: add scheduler with overlap-fetch incremental update"
```

---

## 執行說明

### 首次歷史全量拉取（約 20-30 分鐘，BTCUSDT + ETHUSDT 各兩週期）

```bash
python main.py
```

### 每日增量更新

```bash
python scheduler.py
```

### Windows 工作排程器設定

1. 開啟「工作排程器」→ 建立基本工作
2. 觸發程序：每天 UTC 01:00（台灣時間 09:00）
3. 動作：啟動程式
   - 程式：`python`
   - 引數：`E:\93050207\python\BYBIT_ML\scheduler.py`
   - 起始位置：`E:\93050207\python\BYBIT_ML`
