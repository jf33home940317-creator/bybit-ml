import json
import logging
import joblib
import pandas as pd
from pathlib import Path

import config
from backtest import engine, reporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    features_dir = config.STORAGE_FEATURES
    models_dir   = config.STORAGE_MODELS
    backtest_dir = config.STORAGE_BACKTEST
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for symbol in config.SYMBOLS:
        feature_path = features_dir / f"{symbol}_features.parquet"
        report_path  = features_dir / f"{symbol}_validation_report.json"

        if not feature_path.exists():
            logger.error(f"Features not found: {feature_path}. Run Phase 2 first.")
            continue

        df = pd.read_parquet(feature_path)
        with open(report_path, encoding='utf-8') as f:
            feature_cols = json.load(f)["metadata"]["feature_columns"]

        for target in ["target_fixed", "target_atr"]:
            logger.info(f"[{symbol}][{target}] Loading fold models...")
            fold_paths = [models_dir / f"{symbol}_{target}_fold{k}.pkl" for k in range(1, 6)]
            missing = [p for p in fold_paths if not p.exists()]
            if missing:
                logger.error(f"[{symbol}][{target}] Missing models: {missing}. Run Phase 3 first.")
                continue
            fold_models = [joblib.load(p) for p in fold_paths]

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
