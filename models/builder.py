import json
import logging
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

import config
from models.splitter import purged_walk_forward_split
from models.trainer import train_fold, train_final
from models.evaluator import evaluate_fold
from models.reporter import save_report, save_feature_importance_chart, average_feature_importance

logger = logging.getLogger(__name__)


def build(
    symbol: str,
    target: str,
    features_dir: Path = None,
    models_dir: Path = None,
) -> None:
    """End-to-end model training pipeline for a given symbol and target.

    Args:
        symbol: Crypto symbol (e.g., "BTCUSDT")
        target: Target column name (e.g., "target_fixed")
        features_dir: Directory containing features parquet and validation report.
                      Defaults to config.STORAGE_FEATURES.
        models_dir: Output directory for model pickle files and reports.
                    Defaults to config.STORAGE_MODELS.
    """
    features_dir = Path(features_dir) if features_dir is not None else config.STORAGE_FEATURES
    models_dir   = Path(models_dir)   if models_dir   is not None else config.STORAGE_MODELS
    models_dir.mkdir(parents=True, exist_ok=True)

    # 1. 讀取特徵矩陣
    df = pd.read_parquet(features_dir / f"{symbol}_features.parquet")
    report_path = features_dir / f"{symbol}_validation_report.json"
    feature_cols = json.load(open(report_path))["metadata"]["feature_columns"]

    # 2. 分離 X / y
    X = df[feature_cols]
    y = df[target]

    # 3. Purged Walk-Forward CV
    fold_results = []
    fold_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        purged_walk_forward_split(len(df)), start=1
    ):
        logger.info(
            f"[{symbol}][{target}] Fold {fold_idx}: "
            f"train={len(train_idx)}, val={len(val_idx)}"
        )
        model, best_iter = train_fold(
            X.iloc[train_idx], y.iloc[train_idx],
            X.iloc[val_idx],   y.iloc[val_idx],
        )
        metrics = evaluate_fold(model, X.iloc[val_idx], y.iloc[val_idx])
        metrics["best_iteration"] = best_iter
        fold_results.append(metrics)
        fold_models.append(model)
        joblib.dump(model, models_dir / f"{symbol}_{target}_fold{fold_idx}.pkl")

    # 4. 全量重訓 final model
    mean_iters = round(np.mean([r["best_iteration"] for r in fold_results]))
    final_model = train_final(X, y, n_estimators=mean_iters)
    joblib.dump(final_model, models_dir / f"{symbol}_{target}_final.pkl")

    # 5. 平均 feature importance
    avg_importance = average_feature_importance(fold_models, feature_cols)

    # 6. 儲存報告與圖表
    save_report(fold_results, avg_importance, symbol, target, models_dir)
    save_feature_importance_chart(avg_importance, symbol, target, models_dir)
    logger.info(
        f"[{symbol}][{target}] Training complete. mean_best_iteration={mean_iters}"
    )
