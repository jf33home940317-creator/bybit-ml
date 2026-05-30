from xgboost import XGBClassifier
import pandas as pd
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score


def evaluate_fold(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    """Evaluate model on validation set for one fold.

    Uses fixed threshold of 0.5 for binary classification.

    Returns dict with metrics:
    - precision: Precision score
    - recall: Recall score
    - f1: F1 score
    - roc_auc: ROC AUC score
    - positive_rate_val: Proportion of positive examples in validation set

    All metrics rounded to 4 decimal places.
    """
    # Get probability predictions
    y_pred_proba = model.predict_proba(X_val)[:, 1]

    # Convert to binary predictions using threshold 0.5
    y_pred = (y_pred_proba >= 0.5).astype(int)

    # Calculate metrics with zero_division=0 to handle edge cases
    precision = precision_score(y_val, y_pred, zero_division=0)
    recall = recall_score(y_val, y_pred, zero_division=0)
    f1 = f1_score(y_val, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_val, y_pred_proba)
    positive_rate_val = y_val.mean()

    # Round all metrics to 4 decimal places
    return {
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "positive_rate_val": round(positive_rate_val, 4),
    }
