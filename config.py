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

# Record-only mode — the daemon scores every hour and appends to prob_history.csv
# but never opens a position.
#
# The 2026-09-02 diagnosis (see diag_feature_parity.py / diag_oof_vs_ensemble.py /
# diag_auc_ci.py) found the model has no demonstrated edge:
#   * honest out-of-fold AUC 0.5293; the 0.66 figure was the ensemble scoring rows
#     it had trained on, and CV mean in the training report is 0.5376
#   * live AUC 0.4298 with a 95% block-bootstrap CI of [0.3542, 0.5091] — 0.50 is
#     inside the interval, so it is noise, not a reverse indicator
#   * backtest Sharpe 1.50 came from picking the best of 31 thresholds on 84 trades
# The live pipeline itself is correct (27/27 features match the training path), so
# there is nothing to repair. Scoring continues so the call can be revisited if the
# regime changes; trading does not.
RECORD_ONLY = os.getenv("BYBIT_ML_RECORD_ONLY", "1") == "1"

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

# Shadow threshold — signals between SHADOW and optimal_threshold are logged
# to the ledger (status="shadow") but NOT executed. Used to compare whether
# a lower threshold would have been profitable in live conditions.
SHADOW_THRESHOLD       = 0.70

OVERLAP_HOURS = 3   # overlap candles re-fetched to overwrite unclosed candles from previous run
OVERLAP_DAYS = 3
RATE_LIMIT_SLEEP = 0.2
MAX_RETRIES = 3
