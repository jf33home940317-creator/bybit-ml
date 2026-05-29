import numpy as np
import pandas as pd
from pathlib import Path


def _write_raw_parquets(raw_dir: Path, symbol: str) -> None:
    """Write synthetic 1H and 1D Parquet files large enough for all warm-ups.

    n_h = 6000 hours (~250 days). n_d = 250 days.
    After daily MA200 warm-up + hourly MA200 + tail (24 h), ~976 valid rows remain.
    """
    raw_dir.mkdir(parents=True, exist_ok=True)
    np.random.seed(42)

    n_h = 6000
    h_price = 100.0 + np.cumsum(np.random.randn(n_h) * 0.5)
    hourly = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n_h, freq="1h", tz="UTC"),
        "open": h_price, "high": h_price + 0.5, "low": h_price - 0.5,
        "close": h_price,
        "volume": np.random.uniform(10, 100, n_h),
        "turnover": np.random.uniform(1000, 10000, n_h),
    })
    hourly.to_parquet(raw_dir / f"{symbol}_1h.parquet", index=False)

    n_d = 250
    d_price = 100.0 + np.cumsum(np.random.randn(n_d) * 1.0)
    daily = pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n_d, freq="1D", tz="UTC"),
        "open": d_price, "high": d_price + 2, "low": d_price - 2,
        "close": d_price,
        "volume": np.random.uniform(100, 1000, n_d),
        "turnover": np.random.uniform(10000, 100000, n_d),
    })
    daily.to_parquet(raw_dir / f"{symbol}_1d.parquet", index=False)


class TestBuilder:
    def test_output_has_no_nans(self, tmp_path):
        """Feature Parquet must contain zero NaN values after the full pipeline."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        assert len(df) >= 500, f"Expected ≥500 rows, got {len(df)}"
        bad = df.isna().sum()
        assert bad.sum() == 0, f"NaNs found:\n{bad[bad > 0]}"

    def test_output_has_no_infs(self, tmp_path):
        """Feature Parquet must contain zero inf values."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        assert len(df) >= 500, f"Expected ≥500 rows, got {len(df)}"
        numeric = df.select_dtypes(include=[float, int])
        assert not (numeric.values == float("inf")).any()
        assert not (numeric.values == float("-inf")).any()

    def test_output_has_binary_target_columns(self, tmp_path):
        """target_fixed and target_atr must exist and contain only 0.0 and 1.0."""
        from features.builder import build
        raw_dir = tmp_path / "raw"
        feat_dir = tmp_path / "features"
        _write_raw_parquets(raw_dir, "BTCUSDT")
        build("BTCUSDT", raw_dir=raw_dir, features_dir=feat_dir)
        df = pd.read_parquet(feat_dir / "BTCUSDT_features.parquet")
        assert len(df) >= 500, f"Expected ≥500 rows, got {len(df)}"
        assert "target_fixed" in df.columns
        assert "target_atr" in df.columns
        assert df["target_fixed"].isin([0.0, 1.0]).all()
        assert df["target_atr"].isin([0.0, 1.0]).all()
