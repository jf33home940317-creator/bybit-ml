import json
import logging
import joblib
import pandas as pd
from pathlib import Path

import config
from backtest import reporter, simulator

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

COMBINATIONS = [
    ("ETHUSDT", "target_atr"),
    ("BTCUSDT", "target_atr"),
]


def load_assets(symbol: str, target: str):
    """載入 features parquet、feature_cols、5 個 fold 模型。"""
    features_dir = config.STORAGE_FEATURES
    models_dir   = config.STORAGE_MODELS
    backtest_dir = config.STORAGE_BACKTEST

    feature_path = features_dir / f"{symbol}_features.parquet"
    report_path  = features_dir / f"{symbol}_validation_report.json"

    if not feature_path.exists():
        raise FileNotFoundError(f"Features not found: {feature_path}. Run Phase 2 first.")
    if not report_path.exists():
        raise FileNotFoundError(f"Validation report not found: {report_path}.")

    df = pd.read_parquet(feature_path)
    with open(report_path, encoding='utf-8') as f:
        feature_cols = json.load(f)["metadata"]["feature_columns"]

    fold_paths = [models_dir / f"{symbol}_{target}_fold{k}.pkl" for k in range(1, 6)]
    missing = [p for p in fold_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing fold models: {missing}. Run Phase 3 first.")
    fold_models = [joblib.load(p) for p in fold_paths]

    # 從 threshold_scan.json 讀取 optimal_threshold（不寫死）
    threshold_path = backtest_dir / f"{symbol}_{target}_threshold_scan.json"
    if not threshold_path.exists():
        raise FileNotFoundError(f"Threshold scan not found: {threshold_path}. Run Phase 4.1 first.")
    with open(threshold_path, encoding='utf-8') as f:
        optimal_threshold = json.load(f)["optimal_threshold"]

    return df, feature_cols, fold_models, optimal_threshold


def main() -> None:
    backtest_dir = config.STORAGE_BACKTEST
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for symbol, target in COMBINATIONS:
        logger.info(f"[{symbol}][{target}] 載入資產...")
        try:
            df, feature_cols, fold_models, optimal_threshold = load_assets(symbol, target)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        logger.info(f"[{symbol}][{target}] optimal_threshold={optimal_threshold}，開始模擬...")
        results = simulator.run_portfolio_simulation(
            df, feature_cols, fold_models, target,
            optimal_threshold=optimal_threshold,
        )

        m = results['metrics']
        logger.info(
            f"[{symbol}][{target}] "
            f"final_equity={results['final_equity']:,.0f} USD | "
            f"executed={results['executed_trades']} | "
            f"skipped={results['skipped_signals']} | "
            f"MDD={m.get('max_drawdown_pct', 'N/A')}% | "
            f"Sharpe={m.get('sharpe_ratio', 'N/A')}"
        )

        reporter.save_portfolio_report(results, symbol, target, backtest_dir)
        reporter.save_portfolio_equity_curve(results, symbol, target, backtest_dir)
        logger.info(f"[{symbol}][{target}] 輸出完成 → {backtest_dir}")


if __name__ == "__main__":
    main()
