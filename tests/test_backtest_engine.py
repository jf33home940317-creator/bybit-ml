import numpy as np
import pandas as pd
import pytest
from backtest.engine import generate_oof_probabilities
from models.splitter import purged_walk_forward_split


class _ConstantModel:
    """Mock model that always returns a fixed probability for all inputs."""
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _make_df(n: int = 200, close: float = 1000.0) -> pd.DataFrame:
    """Minimal DataFrame with all columns required by the backtest engine."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01', periods=n, freq='1h'),
        'open':   close,
        'high':   close * 1.005,
        'low':    close * 0.995,
        'close':  close,
        'volume': 1.0,
        'atr_14': close * 0.01,
    })


class TestGenerateOofProbabilities:

    def test_oof_no_future_leak(self):
        """Training-period rows (never in any val set) must be NaN."""
        n = 600
        df = _make_df(n)
        feature_cols = []
        fold_models = [_ConstantModel(0.7)] * 5

        proba = generate_oof_probabilities(df, feature_cols, fold_models)

        # With n=600: fold_size=100, first val starts at index 124.
        # Indices [0, 124) were NEVER in a validation set.
        assert proba.iloc[:124].isna().all(), "Training-period rows must be NaN"

    def test_oof_val_coverage(self):
        """Number of non-NaN values must equal the sum of all val-set sizes."""
        n = 600
        df = _make_df(n)
        feature_cols = []
        fold_models = [_ConstantModel(0.7)] * 5

        proba = generate_oof_probabilities(df, feature_cols, fold_models)

        expected = sum(len(v) for _, v in purged_walk_forward_split(n))
        assert proba.notna().sum() == expected  # 5 × 76 = 380 for n=600
