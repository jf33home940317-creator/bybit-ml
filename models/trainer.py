from xgboost import XGBClassifier
import pandas as pd
import numpy as np


def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[XGBClassifier, int]:
    """Train an XGBoost model on one fold with early stopping.

    Returns (trained_model, best_iteration).
    """
    pos_count = (y_train == 1).sum()
    neg_count = (y_train == 0).sum()
    scale_pos_weight = float(neg_count) / float(pos_count)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=2000,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        early_stopping_rounds=100,
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=[(X_eval, y_eval)],
        verbose=False,
    )

    best_iteration = model.best_iteration
    if best_iteration is None:
        best_iteration = model.n_estimators

    return model, int(best_iteration)


def train_final(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,
) -> XGBClassifier:
    """Train final XGBoost model on full data without early stopping.

    Uses fixed n_estimators (typically the average best_iteration from CV folds).
    """
    pos_count = (y == 1).sum()
    neg_count = (y == 0).sum()
    scale_pos_weight = float(neg_count) / float(pos_count)

    model = XGBClassifier(
        objective="binary:logistic",
        n_estimators=n_estimators,
        learning_rate=0.01,
        max_depth=4,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
    )

    model.fit(X, y, verbose=False)

    return model
