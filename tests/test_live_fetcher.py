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
        assert set(df.columns) == {"timestamp", "open", "high", "low", "close", "volume", "turnover"}

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
        assert str(df["timestamp"].dt.tz) == "UTC", "timestamp must be UTC"

    def test_raises_on_api_error(self):
        mock = MagicMock()
        mock.json.return_value = {"retCode": 10001, "retMsg": "Params Error"}
        mock.raise_for_status.return_value = None
        with patch("live.fetcher.requests.get", return_value=mock):
            from live.fetcher import fetch_latest
            with pytest.raises(ValueError, match="Bybit API error 10001"):
                fetch_latest("BADXYZ", "60", 2)

    def test_retries_on_transient_network_error(self, monkeypatch):
        """fetch_latest must retry transient network failures and succeed if a
        subsequent attempt works. Otherwise a single dropped TCP connection
        skips the heartbeat's signal evaluation."""
        from live import fetcher
        import requests as _req

        calls = {"n": 0}
        success_response = _mock_get(_MOCK_ROWS_NEWEST_FIRST)

        def flaky_get(*args, **kwargs):
            calls["n"] += 1
            if calls["n"] < 3:
                raise _req.exceptions.ConnectionError("simulated network blip")
            return success_response

        monkeypatch.setattr(fetcher, "_sleep", lambda s: None)  # speed up
        monkeypatch.setattr(fetcher.requests, "get", flaky_get)

        df = fetcher.fetch_latest("ETHUSDT", "60", 2)
        assert calls["n"] == 3
        assert len(df) == 2

    def test_gives_up_after_max_retries(self, monkeypatch):
        from live import fetcher
        import requests as _req

        calls = {"n": 0}
        def always_fail(*args, **kwargs):
            calls["n"] += 1
            raise _req.exceptions.ConnectionError("persistent failure")

        monkeypatch.setattr(fetcher, "_sleep", lambda s: None)
        monkeypatch.setattr(fetcher.requests, "get", always_fail)

        with pytest.raises(_req.exceptions.ConnectionError):
            fetcher.fetch_latest("ETHUSDT", "60", 2)
        assert calls["n"] == fetcher.MAX_RETRIES
