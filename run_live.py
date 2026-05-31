# run_live.py
import json
import logging
import schedule
import time
import pandas as pd

import config
from live import pipeline, state, ledger, notifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

MAX_CONCURRENT = 3
HOLDING_BARS   = 24       # 24 小時後 timeout
RISK_PCT       = 0.02
INITIAL_EQUITY = 1_000_000

_threshold_cache: dict = {}


def _load_threshold(symbol: str, target: str) -> float:
    key = f"{symbol}_{target}"
    if key not in _threshold_cache:
        path = config.STORAGE_BACKTEST / f"{symbol}_{target}_threshold_scan.json"
        with open(path, encoding="utf-8") as f:
            _threshold_cache[key] = json.load(f)["optimal_threshold"]
    return _threshold_cache[key]


def _sl_tp(close: float, atr_14: float) -> tuple:
    """target_atr: SL = close − 1.5×ATR, TP = close + 2.0×ATR"""
    return close - 1.5 * atr_14, close + 2.0 * atr_14


def heartbeat() -> None:
    now = pd.Timestamp.now("UTC")
    logger.info(f"[heartbeat] {now.isoformat()}")

    current_state = state.load_state()

    # ── 1. 到期平倉 ──────────────────────────────────────────────────
    current_state, expired = state.expire_closed_positions(current_state, now)
    for pos in expired:
        ledger.append_entry({**pos, "outcome": "timeout", "exit_time_actual": now.isoformat()})
        notifier.send(
            f"[BYBIT_ML] 📋 **{pos['symbol']} 倉位到期**\n"
            f"進場：{pos['entry_price']:.4f} @ {pos['entry_time']}\n"
            f"結果：Timeout（{HOLDING_BARS} 小時）"
        )
        logger.info(f"Expired: {pos['symbol']} @ {pos['entry_price']}")

    # ── 2. 檢查訊號 ──────────────────────────────────────────────────
    for symbol in config.LIVE_SYMBOLS:
        n_active = state.count_active(current_state)
        if n_active >= MAX_CONCURRENT:
            logger.info(f"[{symbol}] Concurrent limit ({n_active}/{MAX_CONCURRENT}), skip")
            continue

        target    = config.LIVE_TARGET
        threshold = _load_threshold(symbol, target)

        try:
            feature_cols, fold_models = pipeline.load_assets(symbol, target)
            result = pipeline.compute_signal(symbol, feature_cols, fold_models, threshold)
        except Exception as e:
            logger.error(f"[{symbol}] Signal failed: {e}")
            continue

        logger.info(f"[{symbol}] prob={result['probability']:.4f} signal={result['signal']}")

        if not result["signal"]:
            continue

        sl, tp  = _sl_tp(result["close"], result["atr_14"])
        sl_dist = result["close"] - sl
        if sl_dist <= 0:
            logger.warning(f"[{symbol}] Skipping signal: sl_dist={sl_dist:.6f} (atr_14={result['atr_14']:.6f})")
            continue
        pos_qty   = (INITIAL_EQUITY * RISK_PCT) / sl_dist
        pos_usd   = pos_qty * result["close"]
        ts        = pd.Timestamp(result["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        exit_time = (ts + pd.Timedelta(hours=HOLDING_BARS)).isoformat()

        position = {
            "symbol":       symbol,
            "target":       target,
            "entry_time":   result["timestamp"],
            "entry_price":  result["close"],
            "sl_price":     round(sl, 4),
            "tp_price":     round(tp, 4),
            "atr_14":       result["atr_14"],
            "probability":  result["probability"],
            "position_usd": round(pos_usd, 2),
            "exit_time":    exit_time,
        }

        current_state = state.add_position(current_state, position)
        ledger.append_entry({**position, "status": "open"})
        notifier.send(
            f"[BYBIT_ML] 🚀 **{symbol} 買入訊號**\n"
            f"機率：{result['probability']:.4f} > {threshold}\n"
            f"進場價：{result['close']:,.4f}\n"
            f"SL：{sl:,.4f}  |  TP：{tp:,.4f}\n"
            f"虛擬部位：${pos_usd:,.0f} USD\n"
            f"預計出場：{exit_time}"
        )
        logger.info(f"[{symbol}] Signal! prob={result['probability']:.4f}, pos=${pos_usd:,.0f}")

    state.save_state(current_state)
    logger.info("[heartbeat] Done")


def main() -> None:
    if not config.DISCORD_WEBHOOK_URL:
        logger.warning("DISCORD_WEBHOOK_URL not set — Discord notifications disabled")
    logger.info("Live signal daemon starting — running once immediately...")
    heartbeat()
    schedule.every().hour.at(":01").do(heartbeat)
    logger.info("Scheduled: every hour at :01. Ctrl+C to stop.")
    while True:
        schedule.run_pending()
        time.sleep(30)


if __name__ == "__main__":
    main()
