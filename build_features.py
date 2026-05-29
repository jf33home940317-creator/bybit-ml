import logging
import config
from features import builder

logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

if __name__ == "__main__":
    for symbol in config.SYMBOLS:
        try:
            builder.build(symbol)
        except Exception as exc:
            logging.error("[%s] build failed: %s", symbol, exc, exc_info=True)
