import numpy as np
import pandas as pd
import pytest


class TestAttachFundingFeatures:

    def _make_hourly(self, n=100):
        ts = pd.date_range("2022-01-01", periods=n, freq="h", tz="UTC")
        return pd.DataFrame({"timestamp": ts, "close": np.random.uniform(2000, 3000, n)})

    def _make_fr(self, n=40):
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
        before = result[result["timestamp"] < pd.Timestamp("2022-01-03T08:00:00+00:00")]
        assert before["funding_rate"].isna().all()

    def test_none_fr_produces_nan_columns(self):
        from features.indicators import _attach_funding_features
        hourly = self._make_hourly(50)
        result = _attach_funding_features(hourly, None)
        assert result["funding_rate"].isna().all()
        assert result["funding_rate_ma_24"].isna().all()
        assert result["funding_zscore_30d"].isna().all()


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

    def test_none_oi_produces_nan_columns(self):
        from features.indicators import _attach_oi_features
        hourly = self._make_hourly(50)
        result = _attach_oi_features(hourly, None)
        assert result["oi_change_1h"].isna().all()
