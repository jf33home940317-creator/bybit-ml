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
    all_hourly: dict = None,
) -> None:
    raw_dir      = Path(raw_dir)      if raw_dir      is not None else config.STORAGE_RAW
    features_dir = Path(features_dir) if features_dir is not None else config.STORAGE_FEATURES
    features_dir.mkdir(parents=True, exist_ok=True)

    for _path in (raw_dir / f"{symbol}_1h.parquet", raw_dir / f"{symbol}_1d.parquet"):
        if not _path.exists():
            raise FileNotFoundError(
                f"[{symbol}] Raw data not found: {_path}. "
                "Run the Phase 1 data pipeline first."
            )

    # 1. Load Phase 1 Parquet
    hourly_df = pd.read_parquet(raw_dir / f"{symbol}_1h.parquet")
    daily_df  = pd.read_parquet(raw_dir / f"{symbol}_1d.parquet")

    # Load FR/OI if available (Phase 6 — gracefully skip if not yet fetched)
    fr_df = None
    oi_df = None
    fr_path = raw_dir / f"{symbol}_funding_rate.parquet"
    oi_path = raw_dir / f"{symbol}_open_interest.parquet"
    if fr_path.exists():
        fr_df = pd.read_parquet(fr_path)
        logger.info(f"[{symbol}] Loaded funding rate: {len(fr_df):,} rows")
    if oi_path.exists():
        oi_df = pd.read_parquet(oi_path)
        logger.info(f"[{symbol}] Loaded open interest: {len(oi_df):,} rows")

    # 2. Dual-timeframe alignment (Phase 1 function, prevents look-ahead bias)
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)

    # 2.5. Determine reference dataframe for cross-asset features
    ref_df = None
    if all_hourly:
        all_symbols = config.SYMBOLS
        other_symbols = [s for s in all_symbols if s != symbol]
        if other_symbols:
            ref_symbol = other_symbols[0]
            ref_raw = all_hourly[ref_symbol]
            ref_df = ref_raw[["timestamp", "close"]].rename(columns={"close": "ref_close"})

    # 3. Compute technical indicators — produces head NaNs; also adds atr_14 needed by labels
    df = indicators.compute(df, daily_df, ref_df=ref_df, fr_df=fr_df, oi_df=oi_df)

    # 3.1 ETH: drop cross_ratio (ETH/BTC is self-referential noise; keep cross_roc_24 as pure BTC momentum)
    if symbol == "ETHUSDT" and "cross_ratio" in df.columns:
        df = df.drop(columns=["cross_ratio"])

    # 4. Compute triple barrier labels — produces tail NaNs for last HORIZON rows
    df = labels.compute(df)

    # 5. Clean: replace inf first, then drop NaN (order matters)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop columns that are entirely NaN (e.g. funding_rate / OI columns when
    # no fr_df / oi_df was supplied) so they do not cause all rows to be removed.
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    df = df.reset_index(drop=True)

    # 6. Statistical validation (operates on clean data)
    validator.report(df, features_dir / f"{symbol}_validation_report.json", symbol=symbol)

    # 7. Save feature matrix
    out_path = features_dir / f"{symbol}_features.parquet"
    df.to_parquet(out_path, index=False)
    logger.info(f"Features saved: {out_path} ({len(df):,} rows)")
