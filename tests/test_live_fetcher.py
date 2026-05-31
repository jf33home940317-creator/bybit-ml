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
