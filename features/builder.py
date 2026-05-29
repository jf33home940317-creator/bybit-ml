# features/builder.py
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import config
from data import cleaner
from features import indicators, labels, validator

logger = logging.getLogger(__name__)


def build(
    symbol: str,
    raw_dir: Path = None,
    features_dir: Path = None,
) -> None:
    raw_dir      = Path(raw_dir)      if raw_dir      is not None else config.STORAGE_RAW
    features_dir = Path(features_dir) if features_dir is not None else config.STORAGE_FEATURES
    features_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Phase 1 Parquet
    hourly_df = pd.read_parquet(raw_dir / f"{symbol}_1h.parquet")
    daily_df  = pd.read_parquet(raw_dir / f"{symbol}_1d.parquet")

    # 2. Dual-timeframe alignment (Phase 1 function, prevents look-ahead bias)
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)

    # 3. Compute technical indicators — produces head NaNs; also adds atr_14 needed by labels
    df = indicators.compute(df, daily_df)

    # 4. Compute triple barrier labels — produces tail NaNs for last HORIZON rows
    df = labels.compute(df)

    # 5. Clean: replace inf first, then drop NaN (order matters)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    # 6. Statistical validation (operates on clean data)
    validator.report(df, features_dir / "validation_report.json", symbol=symbol)

    # 7. Save feature matrix
    out_path = features_dir / f"{symbol}_features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Features saved: {out_path} ({len(df):,} rows)")
