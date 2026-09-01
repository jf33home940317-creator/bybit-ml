"""Compare the three ways this repo scores the same rows.

backtest/engine.generate_oof_probabilities scores each row with ONE fold model
(the one that did not train on it). live/pipeline.compute_signal averages all
five. models/*_final.pkl trained on everything.

Result (2026-09-02, ETHUSDT/target_atr, 28,114 out-of-fold rows):

    OOF single fold      AUC 0.5293   <- the only honest number
    5-fold ensemble avg  AUC 0.6652   <- partly in-sample
    final.pkl            AUC 0.7909   <- fully in-sample

The 0.66 that circulated as "the model's CV-AUC" is the middle row: the ensemble
scoring data it had trained on. The training report's own CV mean is 0.5376.

It also disproves a plausible-sounding theory about the live daemon's zero trades:
averaging five models does NOT shrink probabilities enough to make the 0.75
threshold unreachable — on training-era data the ensemble fires 297 times at
>=0.75 versus OOF's 185. The live drought is a 2026 distribution shift (live prob
max 0.6556), not a scoring-method artifact.
"""
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from models.splitter import purged_walk_forward_split

SYMBOL, TARGET = "ETHUSDT", "target_atr"
THRESHOLDS = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80]


def say(*a):
    print(*a, flush=True)


def main():
    import json
    fc = json.loads((config.STORAGE_FEATURES / f"{SYMBOL}_validation_report.json")
                    .read_text(encoding="utf-8"))["metadata"]["feature_columns"]
    df = pd.read_parquet(config.STORAGE_FEATURES / f"{SYMBOL}_features.parquet")

    folds = [joblib.load(p) for p in
             sorted((config.STORAGE_MODELS).glob(f"{SYMBOL}_{TARGET}_fold*.pkl"))]
    final = joblib.load(config.STORAGE_MODELS / f"{SYMBOL}_{TARGET}_final.pkl")

    say("=" * 76)
    say("  OOF (backtest scoring) vs 5-fold ensemble average (live scoring)")
    say("=" * 76)
    say(f"  rows {len(df):,} | features {len(fc)} | fold models {len(folds)}")
    say(f"  window {df['timestamp'].min()} ~ {df['timestamp'].max()}")

    X = df[fc]
    y = df[TARGET].astype(int)

    # --- what the backtest scored on: one model per row, out-of-fold ---------
    oof = pd.Series(np.nan, index=df.index)
    for (_, val_idx), m in zip(purged_walk_forward_split(len(df)), folds):
        oof.iloc[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

    # --- what the daemon computes: mean of all five -------------------------
    per_fold = np.column_stack([m.predict_proba(X)[:, 1] for m in folds])
    ens = pd.Series(per_fold.mean(axis=1), index=df.index)
    fin = pd.Series(final.predict_proba(X)[:, 1], index=df.index)

    have = oof.notna()
    say(f"  out-of-fold rows {int(have.sum()):,} ({have.mean():.0%})")
    say("")
    say(f"  {'series':<26} {'mean':>8} {'std':>8} {'p99':>8} {'max':>8}")
    say("  " + "-" * 62)
    for name, s in [("OOF (single fold)", oof[have]),
                    ("ensemble avg (live)", ens[have]),
                    ("final.pkl", fin[have])]:
        say(f"  {name:<26} {s.mean():>8.4f} {s.std():>8.4f} "
            f"{s.quantile(0.99):>8.4f} {s.max():>8.4f}")

    say("")
    say("  signals produced on the same rows")
    say(f"  {'threshold':<12} {'OOF':>10} {'ensemble':>10} {'final':>10}")
    say("  " + "-" * 44)
    for t in THRESHOLDS:
        say(f"  >={t:<10.2f} {int((oof[have] >= t).sum()):>10,} "
            f"{int((ens[have] >= t).sum()):>10,} {int((fin[have] >= t).sum()):>10,}")

    from sklearn.metrics import roc_auc_score
    say("")
    say("  AUC on the out-of-fold rows (ensemble/final are PARTLY in-sample here)")
    for name, s in [("OOF (honest)", oof[have]),
                    ("ensemble avg", ens[have]),
                    ("final.pkl", fin[have])]:
        say(f"    {name:<16} {roc_auc_score(y[have], s):.4f}")

    say("")
    say("  per-fold spread on the same rows (how much averaging shrinks)")
    pf = pd.DataFrame(per_fold[have.values], columns=[f"fold{i+1}" for i in range(len(folds))])
    say(f"    per-fold std of each row: median {pf.std(axis=1).median():.4f}")
    say(f"    single-fold max prob     : {pf.max().max():.4f}")
    say(f"    row-average max prob     : {pf.mean(axis=1).max():.4f}")
    say("=" * 76)


if __name__ == "__main__":
    main()
