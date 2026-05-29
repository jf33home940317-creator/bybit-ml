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

    def test_ppo_signal_and_hist_not_swapped(self):
        """ppo_hist must equal ppo - ppo_signal (the histogram identity)."""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["ppo", "ppo_signal", "ppo_hist"])
        expected = df["ppo"] - df["ppo_signal"]
        np.testing.assert_allclose(
            df["ppo_hist"].values, expected.values, rtol=1e-5,
            err_msg="ppo_signal and ppo_hist appear to be swapped",
        )

    def test_rsi_bounded_0_to_100(self):
        """RSI must stay within [0, 100]."""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["rsi_7", "rsi_14", "rsi_24"])
        for col in ["rsi_7", "rsi_14", "rsi_24"]:
            assert df[col].between(0, 100).all(), f"{col} out of [0, 100]"

    def test_bband_width_is_non_negative(self):
        """Bollinger Band width must be >= 0."""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["bband_width_20", "bband_width_50"])
        assert (df["bband_width_20"] >= 0).all()
        assert (df["bband_width_50"] >= 0).all()
