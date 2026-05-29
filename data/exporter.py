# data/exporter.py
import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)


def append_parquet(path: Path, new_df: pd.DataFrame) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if path.exists():
        old_df = pd.read_parquet(path)
        # 舊資料在前，新資料在後，keep="last" 保留新版本（已收盤價）
        combined = pd.concat([old_df, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df.sort_values("timestamp").reset_index(drop=True)

    combined.to_parquet(path, index=False)
    logger.info(f"Parquet saved: {path} ({len(combined):,} rows)")


def write_excel(
    path: Path,
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    # Convert timezone-aware timestamps to timezone-naive for Excel compatibility
    hourly_df_tz_naive = hourly_df.copy()
    if hourly_df_tz_naive["timestamp"].dt.tz is not None:
        hourly_df_tz_naive["timestamp"] = hourly_df_tz_naive["timestamp"].dt.tz_convert(None)

    daily_df_tz_naive = daily_df.copy()
    if daily_df_tz_naive["timestamp"].dt.tz is not None:
        daily_df_tz_naive["timestamp"] = daily_df_tz_naive["timestamp"].dt.tz_convert(None)

    with pd.ExcelWriter(path, engine="openpyxl") as writer:
        hourly_df_tz_naive.to_excel(writer, sheet_name="1H", index=False)
        daily_df_tz_naive.to_excel(writer, sheet_name="1D", index=False)

        for sheet_name, fmt in (("1H", "YYYY-MM-DD HH:MM:SS"), ("1D", "YYYY-MM-DD")):
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            for cell in ws["A"][1:]:  # column A = timestamp, skip header
                cell.number_format = fmt

    logger.info(f"Excel saved: {path}")
