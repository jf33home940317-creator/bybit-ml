# features/validator.py
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import pointbiserialr

logger = logging.getLogger(__name__)

TARGET_COLS = ["target_fixed", "target_atr"]
_EXCLUDE = {
    "timestamp", "open", "high", "low", "close", "volume", "turnover",
    "daily_open", "daily_high", "daily_low", "daily_close",
    "daily_volume", "daily_turnover",
}


def report(df: pd.DataFrame, output_path: Path) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    feature_cols = [c for c in df.columns if c not in _EXCLUDE and c not in TARGET_COLS]

    result = {
        "metadata": {
            "total_rows": len(df),
            "feature_columns": feature_cols,
            "target_columns": TARGET_COLS,
        },
        "class_balance": {},
        "correlations_with_target_fixed": {},
        "correlations_with_target_atr": {},
        "high_collinearity_warnings": [],
    }

    # 1. Label distribution
    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        rate = float(df[target].mean())
        warning = None
        if rate < 0.20 or rate > 0.80:
            warning = f"Severe imbalance: {rate:.1%} positive"
            logger.warning(f"[{target}] {warning}")
        else:
            logger.info(f"[{target}] positive rate: {rate:.1%}")
        result["class_balance"][target] = {"positive_rate": round(rate, 4), "warning": warning}

    # 2. Feature-target correlation (Point-Biserial)
    for target in TARGET_COLS:
        if target not in df.columns:
            continue
        y = df[target].values
        corrs = {}
        for feat in feature_cols:
            try:
                r, _ = pointbiserialr(df[feat].values, y)
                corrs[feat] = round(float(r), 4)
            except Exception:
                corrs[feat] = None
        result[f"correlations_with_{target}"] = dict(
            sorted(corrs.items(), key=lambda x: abs(x[1] or 0), reverse=True)
        )

    # 3. High inter-feature collinearity (|r| > 0.95)
    if len(feature_cols) > 1:
        cm = df[feature_cols].corr()
        warnings = []
        for i, f1 in enumerate(feature_cols):
            for j, f2 in enumerate(feature_cols):
                if j <= i:
                    continue
                r = cm.loc[f1, f2]
                if abs(r) > 0.95:
                    warnings.append([f1, f2, round(float(r), 4)])
                    logger.warning(f"High collinearity: {f1} <-> {f2} r={r:.3f}")
        result["high_collinearity_warnings"] = warnings

    # 4. NaN / inf sanity check
    all_cols = feature_cols + [t for t in TARGET_COLS if t in df.columns]
    nan_count = int(df[all_cols].isna().sum().sum())
    inf_count = int(np.isinf(df[feature_cols].select_dtypes(include=[np.number]).values).sum())
    if nan_count > 0 or inf_count > 0:
        logger.error(f"Data quality: {nan_count} NaNs, {inf_count} infs still present!")
    else:
        logger.info("Data quality: 0 NaN, 0 inf OK")

    output_path.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info(f"Validation report: {output_path}")
