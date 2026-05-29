import json
import numpy as np
import pandas as pd
from pathlib import Path


def _make_balanced_df(n=200):
    np.random.seed(7)
    return pd.DataFrame({
        "timestamp": pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC"),
        "rsi_14": np.random.uniform(20, 80, n),
        "ma_bias_50": np.random.uniform(-0.05, 0.05, n),
        "target_fixed": np.random.randint(0, 2, n).astype(float),
        "target_atr":   np.random.randint(0, 2, n).astype(float),
    })


def _make_imbalanced_df(n=200, positive_rate=0.05):
    df = _make_balanced_df(n)
    labels = np.zeros(n)
    labels[:int(n * positive_rate)] = 1.0
    df["target_fixed"] = labels
    df["target_atr"]   = labels
    return df


class TestValidator:
    def test_report_creates_json_file(self, tmp_path):
        """report() saves a readable JSON file with required top-level keys."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_balanced_df(), out)
        assert out.exists()
        data = json.loads(out.read_text())
        assert "metadata" in data
        assert "feature_columns" in data["metadata"]
        assert "target_columns" in data["metadata"]

    def test_report_contains_class_balance(self, tmp_path):
        """class_balance section includes an entry for target_fixed."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_balanced_df(), out)
        data = json.loads(out.read_text())
        assert "class_balance" in data
        assert "target_fixed" in data["class_balance"]

    def test_imbalanced_labels_trigger_warning(self, tmp_path):
        """5% positive rate is < 20% threshold → warning field is non-null."""
        from features.validator import report
        out = tmp_path / "report.json"
        report(_make_imbalanced_df(positive_rate=0.05), out)
        data = json.loads(out.read_text())
        assert data["class_balance"]["target_fixed"]["warning"] is not None
