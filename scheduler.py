import logging
from datetime import datetime, timedelta, timezone

import pandas as pd

import config
from data import fetcher, cleaner, exporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def _get_last_timestamp(parquet_path) -> pd.Timestamp:
    df = pd.read_parquet(parquet_path, columns=["timestamp"])
    return df["timestamp"].max()


def _overlap_delta(interval: str) -> timedelta:
    return (
        timedelta(hours=config.OVERLAP_HOURS)
        if interval == "60"
        else timedelta(days=config.OVERLAP_DAYS)
    )


def incremental_update(symbol: str, interval: str) -> None:
    label = config.INTERVAL_LABELS[interval]
    parquet_path = config.STORAGE_RAW / f"{symbol}_{label}.parquet"

    if not parquet_path.exists():
        logger.error(
            f"{parquet_path} 不存在，請先執行 main.py 進行歷史全量拉取"
        )
        return

    last_ts = _get_last_timestamp(parquet_path)
    start = last_ts - _overlap_delta(interval)   # 往回推，覆蓋未收盤 K 線
    end = datetime.now(timezone.utc)

    start_str = start.strftime("%Y-%m-%d %H:%M:%S")
    end_str   = end.strftime("%Y-%m-%d %H:%M:%S")

    logger.info(f"增量更新 {symbol} {label} | {start_str} → {end_str}")
    df = fetcher.fetch_ohlcv(symbol, interval, start_str, end_str)
    if df.empty:
        logger.info(f"  → 無新資料")
        return

    df = cleaner.clean(df, interval, label=f"{symbol} {label}")
    exporter.append_parquet(parquet_path, df)
    logger.info(f"  → 更新完成（處理 {len(df)} 筆，重疊覆寫護城河啟動）")


def run() -> None:
    for symbol in config.SYMBOLS:
        dfs: dict = {}
        skip_symbol = False

        for interval in config.INTERVALS:
            incremental_update(symbol, interval)
            label = config.INTERVAL_LABELS[interval]
            parquet_path = config.STORAGE_RAW / f"{symbol}_{label}.parquet"

            if not parquet_path.exists():
                logger.error(f"跳過 {symbol} Excel 更新：{parquet_path} 不存在")
                skip_symbol = True
                break

            dfs[interval] = pd.read_parquet(parquet_path)

        if skip_symbol:
            continue

        excel_path = config.STORAGE_EXCEL / f"{symbol}.xlsx"
        exporter.write_excel(excel_path, dfs["60"], dfs["D"])
        logger.info(f"  → Excel 更新 {excel_path}")


if __name__ == "__main__":
    run()
