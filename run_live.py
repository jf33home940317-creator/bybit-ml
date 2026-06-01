# run_live.py
import csv
import json
import logging
import schedule
import time
from pathlib import Path
import pandas as pd

import config
from live import fetcher, pipeline, state, ledger, notifier

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_threshold_cache: dict = {}
_assets_cache: dict = {}

DISABLE_FLAG = config.BASE_DIR / ".disabled"
PROB_CSV     = config.STORAGE_LIVE / "prob_history.csv"


# ─── Pure helpers (unit-tested in tests/test_run_live_helpers.py) ────────────

def _realized_pnl_pct(entry: float, exit_price: float) -> float:
    """Return P&L percent after deducting the round-trip fee (matches backtest)."""
    raw = (exit_price - entry) / entry * 100
    return round(raw - config.FEE_PCT, 4)


def _build_daily_summary(records: list, current_state: dict, now: pd.Timestamp) -> str:
    """24h activity summary used by the daily health heartbeat."""
    cutoff = now - pd.Timedelta(hours=24)

    def _within(ts_str: str) -> bool:
        if not ts_str:
            return False
        ts = pd.Timestamp(ts_str)
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        return ts >= cutoff

    recent_signals = [r for r in records if r.get("status") == "open" and _within(r.get("entry_time"))]
    recent_closes  = [r for r in records if "outcome" in r and _within(r.get("exit_time_actual"))]

    wins   = sum(1 for r in recent_closes if r.get("outcome") == "win")
    losses = sum(1 for r in recent_closes if r.get("outcome") == "loss")

    pnl_usd_sum = sum(
        r["position_usd"] * r["pnl_pct"] / 100
        for r in recent_closes
        if r.get("pnl_pct") is not None and r.get("position_usd") is not None
    )

    return (
        f"[BYBIT_ML] 📊 **24h 健康心跳**\n"
        f"過去 24h 訊號: {len(recent_signals)} 筆\n"
        f"已結算: {wins} 贏 / {losses} 輸\n"
        f"24h 損益: ${pnl_usd_sum:+,.0f} USD\n"
        f"當前持倉: {len(current_state['positions'])} 筆"
    )


class _ErrorThrottle:
    """Rate-limit identical error alerts to one per cooldown window."""

    def __init__(self, cooldown_hours: float = 6.0) -> None:
        self._last_alert: dict = {}
        self._cooldown = pd.Timedelta(hours=cooldown_hours)

    def should_alert(self, key: str, now: pd.Timestamp) -> bool:
        last = self._last_alert.get(key)
        if last is None or now - last >= self._cooldown:
            self._last_alert[key] = now
            return True
        return False


_error_throttle = _ErrorThrottle(cooldown_hours=6)


def _is_disabled(flag_path: Path = None) -> bool:
    """Return True if the kill-switch flag file exists."""
    return (flag_path or DISABLE_FLAG).exists()


def _get_assets(symbol: str, target: str) -> tuple[list[str], list]:
    """Cached wrapper around pipeline.load_assets — models are read once at
    first heartbeat, not every hour."""
    key = f"{symbol}_{target}"
    if key not in _assets_cache:
        _assets_cache[key] = pipeline.load_assets(symbol, target)
        logger.info(f"[{symbol}] Loaded {len(_assets_cache[key][1])} fold models (cached)")
    return _assets_cache[key]


def _record_prob(
    csv_path: Path,
    timestamp: str,
    symbol: str,
    probability: float,
    signal: bool,
    close: float,
) -> None:
    """Append one row to prob_history.csv. Creates the file + header on first call."""
    csv_path = Path(csv_path)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if write_header:
            writer.writerow(["timestamp", "symbol", "probability", "signal", "close"])
        writer.writerow([timestamp, symbol, probability, signal, close])


def _check_risk_guards(records: list, now: pd.Timestamp) -> dict:
    """Evaluate three risk guards. Returns {"blocked": bool, "reason": str}.

    Guards (checked in order, first breach wins):
      1. MDD drawdown halt   — equity vs all-time peak
      2. Consecutive losses   — last N trades all losses
      3. Daily loss limit     — today's realized loss vs equity
    """
    closes = [
        r for r in records
        if "outcome" in r
        and r.get("pnl_usd") is not None
    ]

    if not closes:
        return {"blocked": False, "reason": ""}

    # ── 1. MDD drawdown halt ──
    equity = config.INITIAL_EQUITY
    peak = equity
    for r in closes:
        equity += r["pnl_usd"]
        peak = max(peak, equity)

    if peak > 0:
        dd_pct = (equity - peak) / peak * 100
        if dd_pct <= config.MAX_DRAWDOWN_PCT:
            return {
                "blocked": True,
                "reason": f"回撤熔斷: {dd_pct:.2f}% (限制 {config.MAX_DRAWDOWN_PCT}%)",
            }

    # ── 2. Consecutive losses ──
    recent_outcomes = [r["outcome"] for r in closes[-config.MAX_CONSECUTIVE_LOSSES:]]
    if (len(recent_outcomes) >= config.MAX_CONSECUTIVE_LOSSES
            and all(o == "loss" for o in recent_outcomes)):
        return {
            "blocked": True,
            "reason": f"連續 {config.MAX_CONSECUTIVE_LOSSES} 筆虧損暫停",
        }

    # ── 3. Daily loss limit ──
    today_start = now.normalize()  # midnight UTC

    def _parse_utc(ts_str: str) -> pd.Timestamp:
        t = pd.Timestamp(ts_str)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t

    daily_pnl = sum(
        r["pnl_usd"] for r in closes
        if _parse_utc(r.get("exit_time_actual", "1970-01-01T00:00:00+00:00")) >= today_start
    )
    if equity > 0:
        daily_loss_pct = daily_pnl / equity * 100
        if daily_loss_pct <= config.MAX_DAILY_LOSS_PCT:
            return {
                "blocked": True,
                "reason": f"當日虧損熔斷: {daily_loss_pct:.2f}% (限制 {config.MAX_DAILY_LOSS_PCT}%)",
            }

    return {"blocked": False, "reason": ""}


def _load_threshold(symbol: str, target: str) -> float:
    key = f"{symbol}_{target}"
    if key not in _threshold_cache:
        path = config.STORAGE_BACKTEST / f"{symbol}_{target}_threshold_scan.json"
        with open(path, encoding="utf-8") as f:
            _threshold_cache[key] = json.load(f)["optimal_threshold"]
    return _threshold_cache[key]


def _compute_current_equity() -> float:
    """Return compound equity = INITIAL_EQUITY + sum of all closed-trade USD P&L."""
    records = ledger.load_ledger()
    closes = [
        r for r in records
        if "outcome" in r
        and r.get("pnl_pct") is not None
        and r.get("position_usd") is not None
    ]
    total_pnl = sum(r["position_usd"] * r["pnl_pct"] / 100 for r in closes)
    return config.INITIAL_EQUITY + total_pnl


def _sl_tp(close: float, atr_14: float) -> tuple:
    """target_atr: SL = close − 1.5×ATR, TP = close + 3.0×ATR (matches labels.py ATR_TP_MULT)"""
    return close - 1.5 * atr_14, close + 3.0 * atr_14


def _check_barriers(positions: list, now: pd.Timestamp) -> tuple[list, list]:
    """Check open positions for SL/TP hits on the latest candle.

    Returns (remaining_positions, hit_list).
    hit_list entries: (pos, outcome, exit_price)
    SL wins ties — consistent with labels.py barrier logic.
    """
    hit = []
    remaining = []

    symbol_candles: dict = {}
    for pos in positions:
        sym = pos["symbol"]
        if sym not in symbol_candles:
            try:
                df = fetcher.fetch_latest(sym, "60", 2)
                if len(df) < 2:
                    symbol_candles[sym] = None
                else:
                    # iloc[-1] is the currently forming candle (only seconds/minutes of data)
                    # iloc[-2] is the last fully closed candle — use this for accurate high/low
                    symbol_candles[sym] = df.iloc[-2]
            except Exception as e:
                logger.warning(f"[{sym}] Could not fetch candle for barrier check: {e}")
                symbol_candles[sym] = None

    for pos in positions:
        candle = symbol_candles.get(pos["symbol"])
        if candle is None:
            remaining.append(pos)
            continue

        high = float(candle["high"])
        low  = float(candle["low"])
        tp   = pos["tp_price"]
        sl   = pos["sl_price"]

        tp_hit = high >= tp
        sl_hit = low  <= sl

        if sl_hit or tp_hit:
            # SL wins ties (same-bar collision)
            if sl_hit:
                outcome    = "loss"
                exit_price = sl
            else:
                outcome    = "win"
                exit_price = tp
            hit.append((pos, outcome, exit_price))
        else:
            remaining.append(pos)

    return remaining, hit


def _resolve_shadow_positions(now: pd.Timestamp) -> None:
    """Check shadow signals for SL/TP hits or timeout, same logic as real positions.

    Shadow entries have status="shadow" and no "outcome" key. Once resolved,
    a new entry with status="shadow_closed" + outcome/pnl is appended.
    """
    records = ledger.load_ledger()
    unresolved = [
        (i, r) for i, r in enumerate(records)
        if r.get("status") == "shadow" and "outcome" not in r
    ]
    if not unresolved:
        return

    for _, shadow in unresolved:
        exit_time = pd.Timestamp(shadow["exit_time"])
        if exit_time.tzinfo is None:
            exit_time = exit_time.tz_localize("UTC")

        # Try to fetch latest candle for barrier check
        try:
            df = fetcher.fetch_latest(shadow["symbol"], "60", 2)
            if len(df) < 2:
                continue
            candle = df.iloc[-2]  # last closed bar
        except Exception:
            continue

        high = float(candle["high"])
        low  = float(candle["low"])
        tp   = shadow["tp_price"]
        sl   = shadow["sl_price"]

        outcome = None
        exit_price = None

        # SL/TP barrier check (SL wins ties)
        if low <= sl:
            outcome = "loss"
            exit_price = sl
        elif high >= tp:
            outcome = "win"
            exit_price = tp
        elif now >= exit_time:
            # Timeout — use close of last candle
            exit_price = float(candle["close"])
            pnl_raw = (exit_price - shadow["entry_price"]) / shadow["entry_price"] * 100
            outcome = "win" if pnl_raw > 0 else "loss"

        if outcome is None:
            continue

        pnl_pct = _realized_pnl_pct(shadow["entry_price"], exit_price)
        # Hypothetical position size (same DRC formula as real trades)
        sl_dist = shadow["entry_price"] - shadow["sl_price"]
        if sl_dist > 0:
            pos_usd = (config.INITIAL_EQUITY * config.RISK_PCT / sl_dist) * shadow["entry_price"]
        else:
            pos_usd = 0.0
        pnl_usd = round(pos_usd * pnl_pct / 100, 2)

        ledger.append_entry({
            **shadow,
            "status":           "shadow_closed",
            "outcome":          outcome,
            "exit_price":       exit_price,
            "pnl_pct":          pnl_pct,
            "pnl_usd":          pnl_usd,
            "position_usd":     round(pos_usd, 2),
            "exit_time_actual": now.isoformat(),
        })
        logger.info(f"[{shadow['symbol']}] Shadow resolved: {outcome} pnl={pnl_pct:+.4f}%")


def heartbeat() -> None:
    now = pd.Timestamp.now("UTC")
    logger.info(f"[heartbeat] {now.isoformat()}")

    current_state = state.load_state()

    # ── 0. SL/TP 障礙觸發平倉(路徑相依,每小時檢查)──────────────────
    remaining, barrier_hits = _check_barriers(current_state["positions"], now)
    current_state = {"positions": remaining}
    for pos, outcome, exit_price in barrier_hits:
        pnl_pct = _realized_pnl_pct(pos["entry_price"], exit_price)
        pnl_usd = round(pos["position_usd"] * pnl_pct / 100, 2)
        ledger.append_entry({
            **pos,
            "outcome":          outcome,
            "exit_price":       exit_price,
            "pnl_pct":          pnl_pct,
            "pnl_usd":          pnl_usd,
            "exit_time_actual": now.isoformat(),
        })
        if outcome == "win":
            tag = "🎯 止盈出場"
            pnl_label = f"獲利:+${pnl_usd:,.0f} USD"
        else:
            tag = "🛡️ 止損保護"
            pnl_label = f"虧損:-${abs(pnl_usd):,.0f} USD"
        notifier.send(
            f"[BYBIT_ML] **{pos['symbol']} {tag}**\n"
            f"進場:{pos['entry_price']:.4f} @ {pos['entry_time']}\n"
            f"出場:{exit_price:.4f}  ({pnl_pct:+.4f}%)\n"
            f"{pnl_label}"
        )
        logger.info(f"Barrier closed: {pos['symbol']} outcome={outcome} exit={exit_price} pnl={pnl_pct:+.4f}% ({pnl_usd:+,.0f} USD)")

    # ── 1. 到期平倉(24h timeout,SL/TP 皆未觸發)────────────────────
    current_state, expired = state.expire_closed_positions(current_state, now)
    for pos in expired:
        exit_price = None
        try:
            exit_df = fetcher.fetch_latest(pos["symbol"], "60", 2)
            # use last fully closed bar — iloc[-1] is the still-forming candle
            exit_price = float(exit_df["close"].iloc[-2])
        except Exception as e:
            logger.warning(f"[{pos['symbol']}] Could not fetch exit price: {e}")

        pnl_pct = None
        pnl_usd = None
        outcome = "timeout"
        if exit_price is not None:
            pnl_pct  = _realized_pnl_pct(pos["entry_price"], exit_price)
            pnl_usd  = round(pos["position_usd"] * pnl_pct / 100, 2)
            outcome  = "win" if pnl_pct > 0 else "loss"

        ledger.append_entry({
            **pos,
            "outcome":          outcome,
            "exit_price":       exit_price,
            "pnl_pct":          pnl_pct,
            "pnl_usd":          pnl_usd,
            "exit_time_actual": now.isoformat(),
        })
        result_line = (f"出場:{exit_price:.4f}  P&L:{pnl_pct:+.4f}%  結果:{'✅ 漲' if outcome == 'win' else '❌ 跌'}"
                       if exit_price is not None else "出場價抓取失敗")
        notifier.send(
            f"[BYBIT_ML] 📋 **{pos['symbol']} 倉位到期**\n"
            f"進場:{pos['entry_price']:.4f} @ {pos['entry_time']}\n"
            f"{result_line}"
        )
        logger.info(f"Expired: {pos['symbol']} entry={pos['entry_price']} exit={exit_price} outcome={outcome}")

    # ── 2. 檢查訊號 ──────────────────────────────────────────────────
    disabled = _is_disabled()
    if disabled:
        logger.info("[heartbeat] Kill switch active (.disabled exists) — skipping signal generation")

    # Risk guards: check MDD / consecutive losses / daily loss
    risk_check = {"blocked": False, "reason": ""}
    if not disabled:
        try:
            all_records = ledger.load_ledger()
            risk_check = _check_risk_guards(all_records, now)
            if risk_check["blocked"]:
                logger.warning(f"[heartbeat] Risk guard triggered: {risk_check['reason']}")
                if _error_throttle.should_alert("risk_guard", now):
                    notifier.send(
                        f"[BYBIT_ML] 🚨 **風控熔斷**\n"
                        f"{risk_check['reason']}\n"
                        f"新訊號已暫停，SL/TP 監控照常。\n"
                        f"(同一警告 6h 內僅推播一次)"
                    )
        except Exception as e:
            logger.error(f"Risk guard check failed: {e}")

    for symbol in config.LIVE_SYMBOLS:
        if disabled or risk_check["blocked"]:
            break

        n_active = state.count_active(current_state)
        if n_active >= config.MAX_CONCURRENT:
            logger.info(f"[{symbol}] Concurrent limit ({n_active}/{config.MAX_CONCURRENT}), skip")
            continue

        target    = config.LIVE_TARGET
        threshold = _load_threshold(symbol, target)

        try:
            feature_cols, fold_models = _get_assets(symbol, target)
            result = pipeline.compute_signal(symbol, feature_cols, fold_models, threshold)
        except Exception as e:
            logger.error(f"[{symbol}] Signal failed: {e}")
            if _error_throttle.should_alert(f"{symbol}_signal", now):
                notifier.send(
                    f"[BYBIT_ML] ⚠️ **{symbol} 訊號計算錯誤**\n"
                    f"{type(e).__name__}: {e}\n"
                    f"(同一錯誤 6h 內僅推播一次)"
                )
            continue

        logger.info(f"[{symbol}] prob={result['probability']:.4f} signal={result['signal']}")

        # Record probability for trend analysis
        try:
            _record_prob(PROB_CSV, result["timestamp"], symbol,
                         result["probability"], result["signal"], result["close"])
        except Exception as e:
            logger.warning(f"[{symbol}] Failed to record prob: {e}")

        if not result["signal"]:
            # Shadow signal: prob between SHADOW_THRESHOLD and optimal — record but don't trade
            if result["probability"] >= config.SHADOW_THRESHOLD:
                shadow_sl, shadow_tp = _sl_tp(result["close"], result["atr_14"])
                ts_s = pd.Timestamp(result["timestamp"])
                if ts_s.tzinfo is None:
                    ts_s = ts_s.tz_localize("UTC")
                ledger.append_entry({
                    "symbol":       symbol,
                    "target":       target,
                    "entry_time":   result["timestamp"],
                    "entry_price":  result["close"],
                    "sl_price":     round(shadow_sl, 4),
                    "tp_price":     round(shadow_tp, 4),
                    "atr_14":       result["atr_14"],
                    "probability":  result["probability"],
                    "exit_time":    (ts_s + pd.Timedelta(hours=config.HOLDING_BARS)).isoformat(),
                    "status":       "shadow",
                })
                logger.info(f"[{symbol}] Shadow signal recorded: prob={result['probability']:.4f} (>={config.SHADOW_THRESHOLD}, <{threshold})")
            continue

        sl, tp  = _sl_tp(result["close"], result["atr_14"])
        sl_dist = result["close"] - sl
        if sl_dist <= 0:
            logger.warning(f"[{symbol}] Skipping signal: sl_dist={sl_dist:.6f} (atr_14={result['atr_14']:.6f})")
            continue
        current_equity = _compute_current_equity()
        pos_qty        = (current_equity * config.RISK_PCT) / sl_dist
        pos_usd        = pos_qty * result["close"]
        ts        = pd.Timestamp(result["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize("UTC")
        exit_time = (ts + pd.Timedelta(hours=config.HOLDING_BARS)).isoformat()

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
            f"機率:{result['probability']:.4f} > {threshold}\n"
            f"進場價:{result['close']:,.4f}\n"
            f"SL:{sl:,.4f}  |  TP:{tp:,.4f}\n"
            f"帳戶淨值:${current_equity:,.0f} USD\n"
            f"虛擬部位:${pos_usd:,.0f} USD\n"
            f"預計出場:{exit_time}"
        )
        logger.info(f"[{symbol}] Signal! prob={result['probability']:.4f}, pos=${pos_usd:,.0f}")

    state.save_state(current_state)

    # ── 3. 結算 shadow positions ──────────────────────────────────────
    try:
        _resolve_shadow_positions(now)
    except Exception as e:
        logger.warning(f"Shadow resolution failed (non-fatal): {e}")

    # ── 4. 每日健康心跳(每天 UTC 00:01 的 heartbeat 跑這段)─────────
    if now.hour == 0:
        try:
            records = ledger.load_ledger()
            notifier.send(_build_daily_summary(records, current_state, now))
            logger.info("[heartbeat] Daily summary sent")
        except Exception as e:
            logger.error(f"Daily summary failed: {e}")

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
