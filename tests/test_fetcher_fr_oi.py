# tests/test_fetcher_fr_oi.py
import pytest
from unittest.mock import MagicMock, patch
import pandas as pd
from datetime import timezone

BASE_MS = 1_640_995_200_000  # 2022-01-01 00:00 UTC


def _make_fr_rows(base_ms: int, count: int, interval_ms: int = 28_800_000):
    """Return ``count`` funding-rate rows newest-first (simulates Bybit response).

    Keys: symbol, fundingRate (string), fundingRateTimestamp (string ms).
    The most recent timestamp is base_ms + (count-1)*interval_ms.
    """
    rows = []
    for i in range(count - 1, -1, -1):
        ts = base_ms + i * interval_ms
        rows.append(
            {
                "symbol": "BTCUSDT",
                "fundingRate": "0.0001",
                "fundingRateTimestamp": str(ts),
            }
        )
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# fetch_funding_rate
# ─────────────────────────────────────────────────────────────────────────────

class TestFetchFundingRate:
    def test_returns_timestamp_and_funding_rate_columns(self):
        from data.fetcher import fetch_funding_rate

        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("BTCUSDT", "2022-01-01", "2022-01-02")

        assert list(df.columns) == ["timestamp", "funding_rate"]

    def test_funding_rate_is_float(self):
        from data.fetcher import fetch_funding_rate

        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 2)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("BTCUSDT", "2022-01-01", "2022-01-02")

        assert df["funding_rate"].dtype == "float64"

    def test_chronological_order(self):
        from data.fetcher import fetch_funding_rate

        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 3)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("BTCUSDT", "2022-01-01", "2022-01-02")

        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)

    def test_timestamp_is_utc(self):
        from data.fetcher import fetch_funding_rate

        mock_session = MagicMock()
        mock_session.get_funding_rate_history.side_effect = [
            {"result": {"list": _make_fr_rows(BASE_MS, 1)}},
            {"result": {"list": []}},
        ]
        with patch("data.fetcher.HTTP", return_value=mock_session):
            df = fetch_funding_rate("BTCUSDT", "2022-01-01", "2022-01-02")

        assert df["timestamp"].dt.tz == timezone.utc


# ─────────────────────────────────────────────────────────────────────────────
# fetch_open_interest
# ─────────────────────────────────────────────────────────────────────────────

def _make_oi_rows(base_ms: int, count: int, interval_ms: int = 3_600_000):
    """Return ``count`` open-interest rows (order unspecified, matches REST result.list).

    Keys: openInterest (string), timestamp (string ms).
    """
    rows = []
    for i in range(count):
        ts = base_ms + i * interval_ms
        rows.append({"openInterest": "12345.678", "timestamp": str(ts)})
    return rows


class TestFetchOpenInterest:
    def _mock_response(self, rows):
        mock_resp = MagicMock()
        mock_resp.json.return_value = {
            "retCode": 0,
            "result": {"list": rows},
        }
        mock_resp.raise_for_status = MagicMock()
        return mock_resp

    def test_returns_timestamp_and_oi_columns(self):
        from data.fetcher import fetch_open_interest

        with patch("requests.get", return_value=self._mock_response(_make_oi_rows(BASE_MS, 3))) as mock_get:
            # Second call returns empty to stop pagination
            mock_get.side_effect = [
                self._mock_response(_make_oi_rows(BASE_MS, 3)),
                self._mock_response([]),
            ]
            df = fetch_open_interest("BTCUSDT", "2022-01-01", "2022-01-02")

        assert list(df.columns) == ["timestamp", "open_interest"]

    def test_oi_is_float(self):
        from data.fetcher import fetch_open_interest

        mock_resp = self._mock_response(_make_oi_rows(BASE_MS, 2))
        with patch("requests.get", side_effect=[
            self._mock_response(_make_oi_rows(BASE_MS, 2)),
            self._mock_response([]),
        ]):
            df = fetch_open_interest("BTCUSDT", "2022-01-01", "2022-01-02")

        assert df["open_interest"].dtype == "float64"

    def test_chronological_order(self):
        from data.fetcher import fetch_open_interest

        with patch("requests.get", side_effect=[
            self._mock_response(_make_oi_rows(BASE_MS, 3)),
            self._mock_response([]),
        ]):
            df = fetch_open_interest("BTCUSDT", "2022-01-01", "2022-01-02")

        timestamps = df["timestamp"].tolist()
        assert timestamps == sorted(timestamps)
