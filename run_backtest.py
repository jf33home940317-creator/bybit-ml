import json
import logging
import joblib
import pandas as pd
from pathlib import Path

import config
from backtest import engine, reporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Actual data lives in the main repo, not the worktree
_MAIN_REPO = Path("E:/93050207/python/BYBIT_ML")
FEATURES_DIR = _MAIN_REPO / "storage" / "features"
MODELS_DIR   = _MAIN_REPO / "storage" / "models"


def main() -> None:
    backtest_dir = _MAIN_REPO / "storage" / "backtest"
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for symbol in config.SYMBOLS:
        feature_path = FEATURES_DIR / f"{symbol}_features.parquet"
        report_path  = FEATURES_DIR / f"{symbol}_validation_report.json"

        df = pd.read_parquet(feature_path)
        feature_cols = json.load(open(report_path))["metadata"]["feature_columns"]

        for target in ["target_fixed", "target_atr"]:
            logger.info(f"[{symbol}][{target}] Loading fold models...")
            fold_models = [
                joblib.load(MODELS_DIR / f"{symbol}_{target}_fold{k}.pkl")
                for k in range(1, 6)
            ]

            logger.info(f"[{symbol}][{target}] Running threshold scan...")
            results = engine.run_threshold_scan(df, feature_cols, fold_models, target)
            opt = results['optimal_threshold']
            m   = results['optimal_metrics']
            logger.info(
                f"[{symbol}][{target}] optimal_threshold={opt}, "
                f"n_trades={m['n_trades'] if m else 'N/A'}, "
                f"sharpe={m['sharpe_ratio'] if m else 'N/A'}"
            )

            reporter.save_threshold_scan(results, symbol, target, backtest_dir)
            reporter.save_threshold_tradeoff_chart(results, symbol, target, backtest_dir)
            reporter.save_equity_curve(results, symbol, target, backtest_dir)
            logger.info(f"[{symbol}][{target}] Saved to {backtest_dir}")


if __name__ == "__main__":
    main()
