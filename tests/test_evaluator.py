import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from xgboost import XGBClassifier


@pytest.fixture
def trained_model_and_data():
    """Create a trained XGBClassifier with learnable signal for testing."""
    np.random.seed(42)
    X = pd.DataFrame(np.random.randn(200, 5), columns=[f"f{i}" for i in range(5)])
    y = pd.Series((X["f0"] > 0).astype(int))
    model = XGBClassifier(n_estimators=50, random_state=42, eval_metric="logloss")
    model.fit(X, y)
    return model, X, y


class TestEvaluateFold:
    def test_returns_required_keys(self, trained_model_and_data):
        """Verify that evaluate_fold returns dict with all required keys."""
        from models.evaluator import evaluate_fold

        model, X, y = trained_model_and_data

        result = evaluate_fold(model, X, y)

        required_keys = {"precision", "recall", "f1", "roc_auc", "positive_rate_val"}
        assert set(result.keys()) == required_keys

    def test_metrics_in_valid_range(self, trained_model_and_data):
        """Verify that all metrics are in valid range [0.0, 1.0]."""
        from models.evaluator import evaluate_fold

        model, X, y = trained_model_and_data

        result = evaluate_fold(model, X, y)

        for key in ["precision", "recall", "f1", "roc_auc", "positive_rate_val"]:
            assert 0.0 <= result[key] <= 1.0, f"{key} out of valid range: {result[key]}"

    def test_perfect_prediction(self):
        """Verify metrics for perfect predictions."""
        from models.evaluator import evaluate_fold

        # Create mock model with perfect predictions
        model = MagicMock(spec=XGBClassifier)
        # Perfect prediction: probability 1.0 for positive, 0.0 for negative
        y_val = pd.Series([1, 1, 0, 0])
        X_val = pd.DataFrame(np.zeros((4, 1)))

        # Set up mock to return perfect probabilities
        model.predict_proba.return_value = np.array([
            [0.0, 1.0],  # positive, prob=1.0
            [0.0, 1.0],  # positive, prob=1.0
            [1.0, 0.0],  # negative, prob=0.0
            [1.0, 0.0],  # negative, prob=0.0
        ])

        result = evaluate_fold(model, X_val, y_val)

        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0

    def test_positive_rate_val(self, trained_model_and_data):
        """Verify that positive_rate_val equals y_val.mean()."""
        from models.evaluator import evaluate_fold

        model, X, y = trained_model_and_data

        result = evaluate_fold(model, X, y)

        expected_positive_rate = round(y.mean(), 4)
        assert result["positive_rate_val"] == expected_positive_rate

    def test_metrics_rounded_to_4_decimals(self):
        """Verify that all metrics are rounded to 4 decimal places."""
        from models.evaluator import evaluate_fold

        # Create mock model with specific predictions
        model = MagicMock(spec=XGBClassifier)
        y_val = pd.Series([1, 0, 1, 0, 1])
        X_val = pd.DataFrame(np.zeros((5, 1)))

        # Set up mock predictions
        model.predict_proba.return_value = np.array([
            [0.0, 0.9],
            [0.2, 0.8],
            [0.3, 0.7],
            [0.6, 0.4],
            [0.1, 0.9],
        ])

        result = evaluate_fold(model, X_val, y_val)

        for key in ["precision", "recall", "f1", "roc_auc", "positive_rate_val"]:
            # Check that value has at most 4 decimal places
            assert isinstance(result[key], float)
            # Verify rounding by checking decimal places
            decimal_str = str(result[key]).split('.')[-1]
            assert len(decimal_str) <= 4, f"{key} has more than 4 decimal places: {result[key]}"
