# live/pipeline.py
import json
import logging
import joblib
import numpy as np
import pandas as pd

import config
from data import cleaner
from features import indicators
from live.fetcher import fetch_latest

logger = logging.getLogger(__name__)

HOURLY_LOOKBACK = 300   # SMA_200 warmup (200) + buffer
DAILY_LOOKBACK  = 220   # daily_ma_bias_200 warmup (200) + buffer


def load_assets(symbol: str, target: str) -> tuple[list[str], list]:
    """Load feature_cols list and 5 fold models from storage."""
    report_path = config.STORAGE_FEATURES / f"{symbol}_validation_report.json"
    with open(report_path, encoding="utf-8") as f:
        feature_cols = json.load(f)["metadata"]["feature_columns"]
    fold_models = [
        joblib.load(config.STORAGE_MODELS / f"{symbol}_{target}_fold{k}.pkl")
        for k in range(1, 6)
    ]
    return feature_cols, fold_models


def compute_signal(
    symbol: str,
    feature_cols: list,
    fold_models: list,
    optimal_threshold: float,
    hourly_df: pd.DataFrame = None,
    daily_df: pd.DataFrame = None,
    btc_hourly_df: pd.DataFrame = None,
) -> dict:
    """
    Fetch live data (or accept injected DataFrames for testing), compute features,
    run ensemble inference, and return signal dict.

    Args for offline testing / injection:
        hourly_df:     Inject hourly OHLCV (avoids live fetch). Defaults to None (fetch live).
        daily_df:      Inject daily OHLCV (avoids live fetch). Defaults to None (fetch live).
        btc_hourly_df: Inject BTC hourly data for cross_roc_24 feature. Defaults to None
                       (fetched live when symbol != 'BTCUSDT'). Must inject for fully
                       offline testing.

    Returns:
        {
          "symbol":      str,
          "timestamp":   str (ISO 8601 UTC),
          "close":       float,
          "atr_14":      float,
          "probability": float,
          "signal":      bool,
        }
    """
    if hourly_df is None:
        hourly_df = fetch_latest(symbol, "60", HOURLY_LOOKBACK)
    if daily_df is None:
        daily_df = fetch_latest(symbol, "D", DAILY_LOOKBACK)

    # cross_roc_24 = BTC ROC_24，ETHUSDT 模型需要 BTCUSDT hourly 作為 ref_df
    ref_df = None
    if symbol != "BTCUSDT":
        if btc_hourly_df is None:
            btc_hourly_df = fetch_latest("BTCUSDT", "60", HOURLY_LOOKBACK)
        ref_df = btc_hourly_df[["timestamp", "close"]].rename(columns={"close": "ref_close"})

    # Feature pipeline（繞過 build() 的檔案 I/O，直接呼叫底層函式）
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)
    df = indicators.compute(df, daily_df, ref_df=ref_df)
    df = df.dropna(subset=feature_cols + ["atr_14"])

    if df.empty:
        raise ValueError(f"[{symbol}] No valid rows after feature computation — not enough history?")

    last_row = df.iloc[[-1]]           # keep as DataFrame (shape 1×N) for predict_proba
    X = last_row[feature_cols]

    if not fold_models:
        raise ValueError(f"[{symbol}] fold_models is empty — no models to ensemble")
    proba = float(np.mean([m.predict_proba(X)[0, 1] for m in fold_models]))

    return {
        "symbol":      symbol,
        "timestamp":   last_row["timestamp"].iloc[0].isoformat(),
        "close":       float(last_row["close"].iloc[0]),
        "atr_14":      float(last_row["atr_14"].iloc[0]),
        "probability": round(proba, 4),
        "signal":      bool(proba >= optimal_threshold),
    }
