import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def save_report(
    results: list[dict],
    feature_importance: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """將訓練結果序列化為 JSON

    Args:
        results: List of fold results, each containing metrics and best_iteration
        feature_importance: Dict of {feature_name: importance_score}
        symbol: Crypto symbol (e.g., "BTCUSDT")
        target: Target variable name (e.g., "target_fixed")
        output_dir: Output directory for the JSON report
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Calculate mean best iteration
    mean_best_iteration = round(np.mean([r["best_iteration"] for r in results]))

    # Calculate aggregates (mean and std) for metrics
    metrics = ["precision", "recall", "f1", "roc_auc"]
    aggregate = {}

    for metric in metrics:
        values = [r[metric] for r in results]
        mean = round(np.mean(values), 4)
        std = round(np.std(values), 4)
        aggregate[metric] = {"mean": mean, "std": std}

    # Add fold index to results (1-based)
    folds = []
    for fold_idx, result in enumerate(results):
        fold_data = {"fold": fold_idx + 1}
        fold_data.update(result)
        folds.append(fold_data)

    # Build report dict
    report = {
        "symbol": symbol,
        "target": target,
        "n_folds": len(results),
        "gap": 24,
        "mean_best_iteration": mean_best_iteration,
        "folds": folds,
        "aggregate": aggregate,
        "feature_importance": feature_importance,
    }

    # Save to JSON
    output_path = output_dir / f"{symbol}_{target}_training_report.json"
    with open(output_path, "w") as f:
        json.dump(report, f, indent=2)


def save_feature_importance_chart(
    feature_importance: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """產生 Feature Importance 橫向長條圖並儲存為 PNG

    Args:
        feature_importance: Dict of {feature_name: importance_score}
        symbol: Crypto symbol (e.g., "BTCUSDT")
        target: Target variable name (e.g., "target_fixed")
        output_dir: Output directory for the PNG chart
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort by importance in descending order (highest first)
    sorted_items = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    features = [item[0] for item in sorted_items]
    importances = [item[1] for item in sorted_items]

    # Create horizontal bar chart
    plt.figure(figsize=(10, max(6, len(features) * 0.3)))
    plt.barh(features, importances)
    plt.xlabel("Importance Score")
    plt.title(f"Feature Importance: {symbol} {target}")
    plt.tight_layout()

    # Save to PNG
    output_path = output_dir / f"{symbol}_{target}_feature_importance.png"
    plt.savefig(output_path, dpi=100)
    plt.close()


def average_feature_importance(
    fold_models: list,
    feature_cols: list[str],
) -> dict:
    """計算 fold 模型的平均 feature importance，回傳 {feature_name: avg_importance}

    Args:
        fold_models: List of trained XGBoost models (one per fold)
        feature_cols: List of feature column names (in same order as model.feature_importances_)

    Returns:
        Dict mapping feature names to their average importance across folds
    """
    importances = np.array([m.feature_importances_ for m in fold_models])
    avg = importances.mean(axis=0)
    return {col: round(float(v), 6) for col, v in zip(feature_cols, avg)}
