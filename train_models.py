import logging
import config
from models import builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    for symbol in config.SYMBOLS:
        for target in ["target_fixed", "target_atr", "target_atr_short"]:
            builder.build(symbol, target)
