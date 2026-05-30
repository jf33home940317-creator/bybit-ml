import json
import numpy as np
import pandas as pd
import pytest
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


# ---------------------------------------------------------------------------
# models.builder tests
# ---------------------------------------------------------------------------

@pytest.fixture
def synthetic_features_dir(tmp_path):
    """建立合成 features.parquet 與 validation_report.json"""
    symbol = "TESTUSDT"
    n_rows = 500   # 足夠小（快速訓練）但足夠大（5 fold 能跑）
    n_features = 5
    feature_cols = [f"f{i}" for i in range(n_features)]

    np.random.seed(42)
    X = np.random.randn(n_rows, n_features)
    # 可學習 signal：target = (f0 > 0)
    target_fixed = (X[:, 0] > 0).astype(int)
    target_atr   = (X[:, 1] > 0).astype(int)

    df = pd.DataFrame(X, columns=feature_cols)
    df["target_fixed"] = target_fixed
    df["target_atr"]   = target_atr

    features_dir = tmp_path / "features"
    features_dir.mkdir()
    df.to_parquet(features_dir / f"{symbol}_features.parquet", index=False)

    validation_report = {
        "metadata": {
            "symbol": symbol,
            "total_rows": n_rows,
            "feature_columns": feature_cols,
            "target_columns": ["target_fixed", "target_atr"],
        }
    }
    (features_dir / f"{symbol}_validation_report.json").write_text(
        json.dumps(validation_report)
    )
    return features_dir, symbol


class TestModelBuilder:
    """End-to-end tests for models.builder.build()."""

    SYMBOL = "TESTUSDT"
    TARGET = "target_fixed"

    def _run_build(self, synthetic_features_dir, tmp_path):
        from models.builder import build
        features_dir, symbol = synthetic_features_dir
        models_dir = tmp_path / "models"
        build(symbol, self.TARGET, features_dir=features_dir, models_dir=models_dir)
        return models_dir

    def test_fold_pkl_files_exist(self, synthetic_features_dir, tmp_path):
        """確認 5 個 fold pkl 存在。"""
        models_dir = self._run_build(synthetic_features_dir, tmp_path)
        for fold_idx in range(1, 6):
            pkl = models_dir / f"{self.SYMBOL}_{self.TARGET}_fold{fold_idx}.pkl"
            assert pkl.exists(), f"Missing fold pkl: {pkl.name}"

    def test_final_pkl_exists(self, synthetic_features_dir, tmp_path):
        """確認 final model pkl 存在。"""
        models_dir = self._run_build(synthetic_features_dir, tmp_path)
        final_pkl = models_dir / f"{self.SYMBOL}_{self.TARGET}_final.pkl"
        assert final_pkl.exists(), f"Missing final pkl: {final_pkl.name}"

    def test_training_report_exists_and_valid(self, synthetic_features_dir, tmp_path):
        """確認 JSON 報告存在且含必要欄位。"""
        models_dir = self._run_build(synthetic_features_dir, tmp_path)
        report_path = models_dir / f"{self.SYMBOL}_{self.TARGET}_training_report.json"
        assert report_path.exists(), "Training report JSON not found"
        report = json.loads(report_path.read_text())
        for key in ("symbol", "target", "n_folds", "mean_best_iteration",
                    "folds", "aggregate", "feature_importance"):
            assert key in report, f"Missing key in report: {key}"

    def test_training_report_fold_count(self, synthetic_features_dir, tmp_path):
        """確認 JSON 中 n_folds == 5 且 folds 列表長度 == 5。"""
        models_dir = self._run_build(synthetic_features_dir, tmp_path)
        report_path = models_dir / f"{self.SYMBOL}_{self.TARGET}_training_report.json"
        report = json.loads(report_path.read_text())
        assert report["n_folds"] == 5, f"Expected n_folds=5, got {report['n_folds']}"
        assert len(report["folds"]) == 5, (
            f"Expected 5 folds in list, got {len(report['folds'])}"
        )

    def test_feature_importance_png_exists(self, synthetic_features_dir, tmp_path):
        """確認 feature importance PNG 存在。"""
        models_dir = self._run_build(synthetic_features_dir, tmp_path)
        png = models_dir / f"{self.SYMBOL}_{self.TARGET}_feature_importance.png"
        assert png.exists(), f"Missing feature importance PNG: {png.name}"
