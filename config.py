import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["60", "D"]

INTERVAL_LABELS = {
    "60": "1h",
    "D": "1d",
}

HISTORY_START = "2022-01-01"

BASE_DIR = Path(__file__).parent
STORAGE_RAW = BASE_DIR / "storage" / "raw"
STORAGE_EXCEL = BASE_DIR / "storage" / "excel"

OVERLAP_HOURS = 3   # overlap candles re-fetched to overwrite unclosed candles from previous run
OVERLAP_DAYS = 3
RATE_LIMIT_SLEEP = 0.2
MAX_RETRIES = 3
