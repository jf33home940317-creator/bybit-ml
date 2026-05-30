import numpy as np
import pandas as pd
import pytest
from xgboost import XGBClassifier

from models.trainer import train_fold, train_final


# ---------------------------------------------------------------------------
# Shared synthetic data helpers
# ---------------------------------------------------------------------------

def _make_data(n_rows: int = 100, n_features: int = 5, seed: int = 0):
    """Generate synthetic data with a learnable signal.

    The label is determined by the sign of the first feature so that
    the model can meaningfully improve beyond the very first tree, making
    best_iteration reliably > 0.
    """
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(
        rng.standard_normal((n_rows, n_features)),
        columns=[f"f{i}" for i in range(n_features)],
    )
    # Label = 1 when f0 > 0, giving the model a clear signal to learn
    y = pd.Series((X["f0"] > 0).astype(int), name="label")
    return X, y


def _split(X: pd.DataFrame, y: pd.Series, train_frac: float = 0.8):
    split = int(len(X) * train_frac)
    return X.iloc[:split], y.iloc[:split], X.iloc[split:], y.iloc[split:]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_train_fold_return_types():
    """train_fold() must return (XGBClassifier, int)."""
    X, y = _make_data()
    X_tr, y_tr, X_ev, y_ev = _split(X, y)
    result = train_fold(X_tr, y_tr, X_ev, y_ev)

    assert isinstance(result, tuple) and len(result) == 2, (
        "train_fold() should return a 2-tuple"
    )
    model, best_iter = result
    assert isinstance(model, XGBClassifier), (
        f"First element should be XGBClassifier, got {type(model)}"
    )
    assert isinstance(best_iter, int), (
        f"Second element should be int, got {type(best_iter)}"
    )


def test_best_iteration_positive():
    """best_iteration returned by train_fold() must be > 0."""
    X, y = _make_data()
    X_tr, y_tr, X_ev, y_ev = _split(X, y)
    _, best_iter = train_fold(X_tr, y_tr, X_ev, y_ev)

    assert best_iter > 0, f"best_iteration should be > 0, got {best_iter}"


def test_scale_pos_weight_imbalanced():
    """train_fold() should not raise on imbalanced 75:25 labels."""
    rng = np.random.default_rng(42)
    n = 100
    X = pd.DataFrame(
        rng.standard_normal((n, 5)),
        columns=[f"f{i}" for i in range(5)],
    )
    # Give the model a learnable signal: label = 1 when f0 > 0.5
    # This also naturally produces an ~50:50 split, so we force the imbalance
    # by overriding the last quarter of labels to 1 and the rest to 0,
    # but keep f0 values as the signal.
    y_signal = (X["f0"] > 0).astype(int)
    # Force 75:25 imbalance while preserving signal ordering (sort by f0)
    sorted_idx = X["f0"].argsort().values
    y_values = np.zeros(n, dtype=int)
    y_values[sorted_idx[75:]] = 1  # top 25 by f0 get label=1
    y = pd.Series(y_values, name="label")

    X_tr, y_tr, X_ev, y_ev = _split(X, y)

    # Should complete without exception; scale_pos_weight is auto-computed
    model, best_iter = train_fold(X_tr, y_tr, X_ev, y_ev)
    assert isinstance(model, XGBClassifier)
    assert best_iter >= 0  # 0-indexed; just verify it ran without error


def test_train_final_return_type():
    """train_final() must return an XGBClassifier."""
    X, y = _make_data()
    model = train_final(X, y, n_estimators=50)

    assert isinstance(model, XGBClassifier), (
        f"train_final() should return XGBClassifier, got {type(model)}"
    )


def test_train_final_n_estimators():
    """train_final() n_estimators must match the value passed in."""
    X, y = _make_data()
    n = 37
    model = train_final(X, y, n_estimators=n)

    assert model.n_estimators == n, (
        f"Expected n_estimators={n}, got {model.n_estimators}"
    )
