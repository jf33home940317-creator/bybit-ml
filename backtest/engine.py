import numpy as np
import pandas as pd
from models.splitter import purged_walk_forward_split


def generate_oof_probabilities(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
) -> pd.Series:
    """Generate out-of-fold probabilities; training rows remain NaN."""
    proba = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, val_idx), model in zip(purged_walk_forward_split(len(df)), fold_models):
        X_val = df.iloc[val_idx][feature_cols]
        proba.iloc[val_idx] = model.predict_proba(X_val)[:, 1]
    return proba
