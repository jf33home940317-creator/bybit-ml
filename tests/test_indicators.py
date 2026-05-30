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
        for col in ["rsi_14", "rsi_50"]:
            assert col in df.columns, f"Missing: {col}"

    def test_ppo_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["ppo", "ppo_signal", "ppo_hist"]:
            assert col in df.columns, f"Missing: {col}"

    def test_atr_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["atr_14", "atr_72"]:
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
        for col in ["turnover_ratio_24"]:
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
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["rsi_14", "rsi_50"])
        for col in ["rsi_14", "rsi_50"]:
            assert df[col].between(0, 100).all(), f"{col} out of [0, 100]"

    def test_bband_width_is_non_negative(self):
        """Bollinger Band width must be >= 0."""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["bband_width_20", "bband_width_50"])
        assert (df["bband_width_20"] >= 0).all()
        assert (df["bband_width_50"] >= 0).all()

    def test_natr_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["natr_14", "natr_72", "daily_natr_14"]:
            assert col in df.columns, f"Missing: {col}"

    def test_natr_is_positive(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily()).dropna(subset=["natr_14", "natr_72"])
        assert (df["natr_14"] > 0).all()
        assert (df["natr_72"] > 0).all()

    def test_roc_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["roc_4", "roc_12", "roc_24"]:
            assert col in df.columns, f"Missing: {col}"

    def test_time_feature_columns_exist(self):
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "is_weekend"]:
            assert col in df.columns, f"Missing: {col}"

    def test_time_features_bounded(self):
        """sin/cos 值應在 [-1, 1]，is_weekend 應為 0 或 1"""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        for col in ["hour_sin", "hour_cos", "dow_sin", "dow_cos"]:
            assert df[col].between(-1.0, 1.0).all(), f"{col} out of [-1, 1]"
        assert df["is_weekend"].isin([0, 1]).all()

    def test_cross_asset_columns_with_ref_df(self):
        """提供 ref_df 時，應產生 cross_ratio 與 cross_roc_24"""
        from features.indicators import compute
        hourly = _make_hourly()
        ref_df = hourly[["timestamp", "close"]].copy().rename(columns={"close": "ref_close"})
        df = compute(hourly, _make_daily(), ref_df=ref_df)
        assert "cross_ratio" in df.columns
        assert "cross_roc_24" in df.columns
        # cross_ratio 應約等於 1（同一份 close 資料）
        valid = df["cross_ratio"].dropna()
        assert (valid > 0).all()

    def test_cross_asset_absent_without_ref_df(self):
        """不提供 ref_df 時，不應有 cross_ratio 與 cross_roc_24"""
        from features.indicators import compute
        df = compute(_make_hourly(), _make_daily())
        assert "cross_ratio" not in df.columns
        assert "cross_roc_24" not in df.columns
