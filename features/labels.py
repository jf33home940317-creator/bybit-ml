# features/labels.py
import numpy as np
import pandas as pd
from numpy.lib.stride_tricks import sliding_window_view

HORIZON = 24
TP_PCT_FIXED = 0.02
SL_PCT_FIXED = 0.01
ATR_TP_MULT = 3.0
ATR_SL_MULT = 1.5


def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values

    df["target_fixed"] = _barrier_labels(
        highs, lows,
        tp=closes * (1.0 + TP_PCT_FIXED),
        sl=closes * (1.0 - SL_PCT_FIXED),
        direction="long",
    )
    df["target_atr"] = _barrier_labels(
        highs, lows,
        tp=closes + ATR_TP_MULT * atrs,
        sl=closes - ATR_SL_MULT * atrs,
        direction="long",
    )
    df["target_atr_short"] = _barrier_labels(
        highs, lows,
        tp=closes - ATR_TP_MULT * atrs,
        sl=closes + ATR_SL_MULT * atrs,
        direction="short",
    )
    return df


def _barrier_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    direction: str = "long",
) -> np.ndarray:
    n = len(highs)
    n_valid = n - HORIZON
    labels = np.full(n, np.nan)
    if n_valid <= 0:
        return labels

    # future_highs[i] = highs[i+1 : i+1+HORIZON], shape (n_valid, HORIZON)
    future_highs = sliding_window_view(highs[1:], HORIZON)
    future_lows  = sliding_window_view(lows[1:],  HORIZON)

    if direction == "long":
        tp_hit = future_highs >= tp[:n_valid, np.newaxis]
        sl_hit = future_lows  <= sl[:n_valid, np.newaxis]
    elif direction == "short":
        tp_hit = future_lows  <= tp[:n_valid, np.newaxis]
        sl_hit = future_highs >= sl[:n_valid, np.newaxis]
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    # First hit index; HORIZON means "never hit"
    tp_first = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), HORIZON)
    sl_first = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), HORIZON)

    # SL wins ties regardless of direction (conservative pessimism)
    wins_tp = (sl_first > tp_first) & (tp_first < HORIZON)
    labels[:n_valid] = np.where(wins_tp, 1.0, 0.0)
    return labels
