import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("BYBIT_API_KEY", "")
API_SECRET = os.getenv("BYBIT_API_SECRET", "")

SYMBOLS = ["BTCUSDT", "ETHUSDT"]
INTERVALS = ["60", "D"]

INTERVAL_TO_FREQ = {
    "60": "1h",
    "D":  "1D",
}

INTERVAL_LABELS = {
    "60": "1h",
    "D": "1d",
}

HISTORY_START = "2022-01-01"

BASE_DIR = Path(__file__).parent
STORAGE_RAW = BASE_DIR / "storage" / "raw"
STORAGE_EXCEL = BASE_DIR / "storage" / "excel"
STORAGE_FEATURES = BASE_DIR / "storage" / "features"
STORAGE_MODELS = BASE_DIR / "storage" / "models"
STORAGE_BACKTEST = BASE_DIR / "storage" / "backtest"
STORAGE_LIVE = BASE_DIR / "storage" / "live"

# ── Live Trading ──────────────────────────────────────────────────────────────
DISCORD_WEBHOOK_URL = os.getenv("DISCORD_WEBHOOK_URL", "")
LIVE_TARGET  = "target_atr"       # must match model training label
LIVE_SYMBOLS = ["ETHUSDT"]        # BTCUSDT excluded (negative Sharpe in Phase 4.2)

# Live execution constants — shared by run_live.py and show_results.py so the
# displayed simulation matches what the daemon actually does.
INITIAL_EQUITY = 1_000_000.0
RISK_PCT       = 0.02
HOLDING_BARS   = 24       # 24h timeout if SL/TP not hit
MAX_CONCURRENT = 3
FEE_PCT        = 0.2      # 0.2% round-trip, matches backtest fee=0.002

# Risk guards — any breach pauses new signal generation
MAX_DRAWDOWN_PCT       = -15.0   # halt if equity drops > 15% from peak
MAX_CONSECUTIVE_LOSSES = 5       # halt after 5 consecutive losing trades
MAX_DAILY_LOSS_PCT     = -5.0    # halt if today's realized loss > 5% of equity

OVERLAP_HOURS = 3   # overlap candles re-fetched to overwrite unclosed candles from previous run
OVERLAP_DAYS = 3
RATE_LIMIT_SLEEP = 0.2
MAX_RETRIES = 3
