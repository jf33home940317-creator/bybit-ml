# data/cleaner.py
import logging

import pandas as pd

from config import INTERVAL_TO_FREQ

logger = logging.getLogger(__name__)

OHLC_COLS   = ["open", "high", "low", "close"]
VOLUME_COLS = ["volume", "turnover"]
DAILY_COLS  = ["open", "high", "low", "close", "volume", "turnover"]


def clean(df: pd.DataFrame, interval: str, label: str = "") -> pd.DataFrame:
    original_len = len(df)

    # 1. 去重，保留最新版本（已收盤資料）
    df = df.drop_duplicates(subset=["timestamp"], keep="last")
    df = df.sort_values("timestamp").reset_index(drop=True)
    dupes_removed = original_len - len(df)

    # Guard against empty DataFrame
    if df.empty:
        return df

    # 2. 確保 timestamp 有 UTC 時區
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize("UTC")
    elif str(df["timestamp"].dt.tz) != "UTC":
        df["timestamp"] = df["timestamp"].dt.tz_convert("UTC")

    # 3. 補缺失 K 線
    freq = INTERVAL_TO_FREQ[interval]
    full_index = pd.date_range(
        start=df["timestamp"].min(),
        end=df["timestamp"].max(),
        freq=freq,
        tz="UTC",
    )
    df_indexed = df.set_index("timestamp").reindex(full_index)
    df_indexed.index.name = "timestamp"
    filled = int(df_indexed[OHLC_COLS].isna().any(axis=1).sum())

    df_indexed[OHLC_COLS]   = df_indexed[OHLC_COLS].ffill()
    df_indexed[VOLUME_COLS] = df_indexed[VOLUME_COLS].fillna(0.0)

    df = df_indexed.reset_index()

    tag = label or interval
    logger.info(f"[{tag}] 總筆數: {len(df)} | 補缺: {filled} 根 | 重複移除: {dupes_removed}")
    return df


def align_daily_to_hourly(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    # Guard against empty daily_df
    if daily_df.empty:
        result = hourly_df.copy()
        for col in [f"daily_{c}" for c in DAILY_COLS]:
            result[col] = float("nan")
        return result

    daily = daily_df.copy()

    # 日線向後位移一天，防止 look-ahead bias
    daily["date_available"] = daily["timestamp"] + pd.Timedelta(days=1)

    # 日線欄位加 daily_ 前綴，避免與小時線欄位衝突
    daily = daily.rename(columns={col: f"daily_{col}" for col in DAILY_COLS})

    keep_cols = ["date_available"] + [f"daily_{c}" for c in DAILY_COLS]

    hourly_sorted = hourly_df.sort_values("timestamp").copy()
    daily_sorted = daily[keep_cols].sort_values("date_available").copy()
    # Normalize precision: Parquet reads as ms, in-memory ops may produce us
    hourly_sorted["timestamp"] = hourly_sorted["timestamp"].astype("datetime64[us, UTC]")
    daily_sorted["date_available"] = daily_sorted["date_available"].astype("datetime64[us, UTC]")

    merged = pd.merge_asof(
        hourly_sorted,
        daily_sorted,
        left_on="timestamp",
        right_on="date_available",
        direction="backward",
    )
    return merged.drop(columns=["date_available"], errors="ignore")
