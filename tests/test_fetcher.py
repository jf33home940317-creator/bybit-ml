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
