"""
SOL 回測實驗 — 用現有特徵 pipeline 跑 SOLUSDT，看 Sharpe 是否為正。
完全獨立，不影響主系統任何設定。

Usage:
    cd E:\93050207\python\BYBIT_ML
    python test_sol/run_sol_backtest.py
"""
import sys
import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import config
from data import fetcher, cleaner
from features import indicators, labels, validator
from models.splitter import purged_walk_forward_split
from models.trainer import train_fold
from backtest import engine, simulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "SOLUSDT"
TARGET = "target_atr"
OUT_DIR = Path(__file__).parent / "output"


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    end_date = pd.Timestamp.now("UTC").strftime("%Y-%m-%d %H:%M:%S")

    # ── Phase 1: 抓資料 ──────────────────────────────────────────────
    logger.info(f"[{SYMBOL}] Phase 1: Fetching data...")
    hourly_df = fetcher.fetch_ohlcv(SYMBOL, "60", config.HISTORY_START, end_date)
    hourly_df = cleaner.clean(hourly_df, "60", label=f"{SYMBOL} 1h")
    hourly_path = OUT_DIR / f"{SYMBOL}_1h.parquet"
    hourly_df.to_parquet(hourly_path, index=False)
    logger.info(f"  1h: {len(hourly_df):,} rows")

    daily_df = fetcher.fetch_ohlcv(SYMBOL, "D", config.HISTORY_START, end_date)
    daily_df = cleaner.clean(daily_df, "D", label=f"{SYMBOL} 1d")
    daily_path = OUT_DIR / f"{SYMBOL}_1d.parquet"
    daily_df.to_parquet(daily_path, index=False)
    logger.info(f"  1d: {len(daily_df):,} rows")

    # BTC hourly for cross-asset features
    btc_1h_path = config.STORAGE_RAW / "BTCUSDT_1h.parquet"
    if btc_1h_path.exists():
        btc_hourly = pd.read_parquet(btc_1h_path)
        ref_df = btc_hourly[["timestamp", "close"]].rename(columns={"close": "ref_close"})
        logger.info(f"  BTC ref: {len(ref_df):,} rows")
    else:
        ref_df = None
        logger.warning("  BTC hourly not found, skipping cross-asset features")

    # ── Phase 2: 特徵工程 ────────────────────────────────────────────
    logger.info(f"[{SYMBOL}] Phase 2: Feature engineering...")
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)
    df = indicators.compute(df, daily_df, ref_df=ref_df)

    # Drop cross_ratio for non-BTC (same as ETH pipeline)
    if "cross_ratio" in df.columns:
        df = df.drop(columns=["cross_ratio"])

    df = labels.compute(df)
    df = df.replace([np.inf, -np.inf], np.nan)
    # Drop all-NaN columns (FR/OI not provided)
    df = df.dropna(axis=1, how="all")
    df = df.dropna()
    df = df.reset_index(drop=True)

    # Validation report
    report_path = OUT_DIR / f"{SYMBOL}_validation_report.json"
    validator.report(df, report_path, symbol=SYMBOL)

    feature_cols = json.load(open(report_path))["metadata"]["feature_columns"]
    logger.info(f"  Features: {len(feature_cols)} columns, {len(df):,} rows")

    # ── Phase 3: 訓練 ────────────────────────────────────────────────
    logger.info(f"[{SYMBOL}] Phase 3: Training XGBoost 5-fold...")
    X = df[feature_cols]
    y = df[TARGET]

    fold_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        purged_walk_forward_split(len(df)), start=1
    ):
        model, best_iter = train_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx], y.iloc[val_idx],
        )
        fold_models.append(model)
        joblib.dump(model, OUT_DIR / f"{SYMBOL}_{TARGET}_fold{fold_idx}.pkl")
        logger.info(f"  Fold {fold_idx}: best_iteration={best_iter}")

    # ── Phase 4.1: 門檻掃描 ──────────────────────────────────────────
    logger.info(f"[{SYMBOL}] Phase 4.1: Threshold scan...")
    results = engine.run_threshold_scan(df, feature_cols, fold_models, TARGET)
    opt = results["optimal_threshold"]
    m = results["optimal_metrics"]

    if m is None:
        logger.error(f"[{SYMBOL}] No valid threshold found (too few trades). STOP.")
        return

    logger.info(f"  optimal_threshold={opt}, n_trades={m['n_trades']}, sharpe={m['sharpe_ratio']}")

    # Save threshold scan
    scan_path = OUT_DIR / f"{SYMBOL}_{TARGET}_threshold_scan.json"
    scan_data = {
        "optimal_threshold": results["optimal_threshold"],
        "optimal_metrics": results["optimal_metrics"],
        "threshold_scan": results["threshold_scan"],
        "total_years": results["total_years"],
    }
    with open(scan_path, "w") as f:
        json.dump(scan_data, f, indent=2)

    # ── Phase 4.2: DRC 組合回測 ──────────────────────────────────────
    logger.info(f"[{SYMBOL}] Phase 4.2: Portfolio simulation...")
    portfolio = simulator.run_portfolio_simulation(
        df, feature_cols, fold_models, TARGET,
        optimal_threshold=opt,
    )
    pm = portfolio["metrics"]

    # Save report
    report = {
        "symbol": SYMBOL,
        "target": TARGET,
        "initial_equity": portfolio["initial_equity"],
        "final_equity": portfolio["final_equity"],
        "optimal_threshold": opt,
        "total_signals": portfolio["total_signals"],
        "executed_trades": portfolio["executed_trades"],
        "skipped_signals": portfolio["skipped_signals"],
        "metrics": pm,
    }
    report_out = OUT_DIR / f"{SYMBOL}_{TARGET}_portfolio_report.json"
    with open(report_out, "w") as f:
        json.dump(report, f, indent=2)

    # ── 結果 ──────────────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  {SYMBOL} Backtest Results ({TARGET})")
    print("=" * 60)
    print(f"  {'Optimal Threshold':<25} {opt}")
    print(f"  {'Total Signals':<25} {portfolio['total_signals']}")
    print(f"  {'Executed Trades':<25} {portfolio['executed_trades']}")
    print(f"  {'Skipped (concurrent)':<25} {portfolio['skipped_signals']}")
    print(f"  {'-'*55}")
    print(f"  {'Total Return':<25} {pm.get('total_return_pct', 0):>+10.2f}%")
    print(f"  {'CAGR':<25} {pm.get('cagr_pct', 0):>10.2f}%")
    print(f"  {'Sharpe Ratio':<25} {pm.get('sharpe_ratio', 0):>10.4f}")
    print(f"  {'Max Drawdown':<25} {pm.get('max_drawdown_pct', 0):>10.2f}%")
    print(f"  {'Win Rate':<25} {pm.get('win_rate', 0):>10.2%}")
    print(f"  {'Avg Position USD':<25} ${pm.get('avg_position_usd', 0):>12,.0f}")
    print(f"  {'Avg Holding Bars':<25} {pm.get('avg_holding_bars', 0):>10.1f}h")
    print("=" * 60)

    # Compare with ETH
    eth_report = config.STORAGE_BACKTEST / "ETHUSDT_target_atr_portfolio_report.json"
    if eth_report.exists():
        eth = json.loads(eth_report.read_text())
        eth_m = eth["metrics"]
        print()
        print("  ── vs ETHUSDT ──")
        print(f"  {'Metric':<25} {'ETH':>12} {'SOL':>12}")
        print(f"  {'-'*25} {'-'*12} {'-'*12}")
        print(f"  {'Sharpe':<25} {eth_m['sharpe_ratio']:>12.4f} {pm.get('sharpe_ratio', 0):>12.4f}")
        print(f"  {'Return %':<25} {eth_m['total_return_pct']:>12.2f} {pm.get('total_return_pct', 0):>12.2f}")
        print(f"  {'MDD %':<25} {eth_m['max_drawdown_pct']:>12.2f} {pm.get('max_drawdown_pct', 0):>12.2f}")
        print(f"  {'Win Rate':<25} {eth_m['win_rate']:>12.2%} {pm.get('win_rate', 0):>12.2%}")

    if pm.get("sharpe_ratio", 0) > 0:
        print(f"\n  ✅ SOLUSDT Sharpe > 0 — 可以考慮加入 LIVE_SYMBOLS")
    else:
        print(f"\n  ❌ SOLUSDT Sharpe <= 0 — 不建議上線")

    print()


if __name__ == "__main__":
    main()
