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

HOURLY_LOOKBACK   = 300   # SMA_200 warmup (200) + buffer
DAILY_LOOKBACK    = 220   # daily_ma_bias_200 warmup (200) + buffer
MAX_HOURLY_AGE_HR = 1.5   # last closed bar must be < this many hours old
MAX_DAILY_AGE_HR  = 25    # last closed daily bar must be < this many hours old


def _drop_partial_bars(df: pd.DataFrame, period_hours: float) -> pd.DataFrame:
    """Drop rows whose period (timestamp + period_hours) has not yet fully elapsed.

    Bybit's V5 kline endpoint returns the currently-forming candle as the last row.
    Inference on a partial bar is out-of-distribution vs the model's training data.
    """
    now = pd.Timestamp.now("UTC")
    period = pd.Timedelta(hours=period_hours)
    return df[df["timestamp"] + period <= now].reset_index(drop=True)


def _check_freshness(df: pd.DataFrame, period_hours: float, max_age_hours: float) -> None:
    """Raise if the most recent closed bar's close-time is older than max_age_hours.

    Guards against acting on stale data when the upstream API is degraded.
    """
    if df.empty:
        raise ValueError("No closed bars available — cannot check freshness")
    now = pd.Timestamp.now("UTC")
    last_close_time = df["timestamp"].iloc[-1] + pd.Timedelta(hours=period_hours)
    age = now - last_close_time
    if age > pd.Timedelta(hours=max_age_hours):
        raise ValueError(
            f"Data stale: last closed bar at {last_close_time.isoformat()}, "
            f"age={age} > max={max_age_hours}h"
        )


def load_assets(symbol: str, target: str) -> tuple[list[str], list]:
    """Load feature_cols list and all fold models from storage.

    Fold count is derived from glob (not hardcoded), so retraining with a
    different number of folds doesn't silently load a stale subset.
    """
    report_path = config.STORAGE_FEATURES / f"{symbol}_validation_report.json"
    with open(report_path, encoding="utf-8") as f:
        feature_cols = json.load(f)["metadata"]["feature_columns"]

    fold_paths = sorted(config.STORAGE_MODELS.glob(f"{symbol}_{target}_fold*.pkl"))
    if not fold_paths:
        raise FileNotFoundError(
            f"No fold models found for {symbol}_{target} in {config.STORAGE_MODELS}"
        )
    fold_models = [joblib.load(p) for p in fold_paths]
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
    import time as _time
    _time.sleep(0.3)
    if daily_df is None:
        daily_df = fetch_latest(symbol, "D", DAILY_LOOKBACK)
    _time.sleep(0.3)

    # Drop the partial forming bar that Bybit includes as the last row, then
    # confirm the latest closed bar is fresh enough to act on.
    hourly_df = _drop_partial_bars(hourly_df, period_hours=1)
    daily_df  = _drop_partial_bars(daily_df,  period_hours=24)
    _check_freshness(hourly_df, period_hours=1,  max_age_hours=MAX_HOURLY_AGE_HR)
    _check_freshness(daily_df,  period_hours=24, max_age_hours=MAX_DAILY_AGE_HR)

    # cross_roc_24 = BTC ROC_24，ETHUSDT 模型需要 BTCUSDT hourly 作為 ref_df
    ref_df = None
    if symbol != "BTCUSDT":
        if btc_hourly_df is None:
            _time.sleep(0.3)
            btc_hourly_df = fetch_latest("BTCUSDT", "60", HOURLY_LOOKBACK)
        btc_hourly_df = _drop_partial_bars(btc_hourly_df, period_hours=1)
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
