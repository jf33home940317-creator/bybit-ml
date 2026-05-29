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
