import logging
from datetime import datetime, timezone

import config
from data import fetcher, cleaner, exporter

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def run_full_fetch() -> None:
    end_date = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
    config.STORAGE_RAW.mkdir(parents=True, exist_ok=True)
    config.STORAGE_EXCEL.mkdir(parents=True, exist_ok=True)

    for symbol in config.SYMBOLS:
        dfs: dict = {}

        for interval in config.INTERVALS:
            label = config.INTERVAL_LABELS[interval]
            logger.info(f"抓取 {symbol} {label} | {config.HISTORY_START} → {end_date}")

            df = fetcher.fetch_ohlcv(symbol, interval, config.HISTORY_START, end_date)
            df = cleaner.clean(df, interval, label=f"{symbol} {label}")

            parquet_path = config.STORAGE_RAW / f"{symbol}_{label}.parquet"
            exporter.append_parquet(parquet_path, df)
            logger.info(f"  → 儲存 {parquet_path}（{len(df):,} 筆）")

            dfs[interval] = df

        excel_path = config.STORAGE_EXCEL / f"{symbol}.xlsx"
        exporter.write_excel(excel_path, dfs["60"], dfs["D"])
        logger.info(f"  → Excel 儲存 {excel_path}")


if __name__ == "__main__":
    run_full_fetch()
