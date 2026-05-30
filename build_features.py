import logging
import pandas as pd
import config
from features import builder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    # Pre-load all hourly data for cross-asset features
    all_hourly = {}
    for symbol in config.SYMBOLS:
        path = config.STORAGE_RAW / f"{symbol}_1h.parquet"
        if path.exists():
            all_hourly[symbol] = pd.read_parquet(path)

    for symbol in config.SYMBOLS:
        try:
            builder.build(symbol, all_hourly=all_hourly)
        except Exception as exc:
            logging.error("[%s] build failed: %s", symbol, exc, exc_info=True)
