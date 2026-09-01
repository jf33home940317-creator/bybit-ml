"""Train-vs-live feature parity check for the ETH model.

Hypothesis tested (2026-09-02): the live daemon's 300-bar lookback
(live/pipeline.HOURLY_LOOKBACK) does not reproduce the feature values the model
was trained on.

RESULT: REJECTED. Across 555 timestamps, 25 of 27 features matched exactly;
rsi_50 and natr_72 differed by 0.15% and 0.29% (Wilder EWM warmup residue) with
correlation >= 0.9997. Rebuilt probabilities correlate 0.9996 with what the daemon
actually logged, and all three scoring paths give AUC 0.444. The live pipeline is
correct; 300 bars is enough.

Method: rebuild every feature two ways at the SAME timestamps --
  * "live"  = exactly what compute_signal() sees (300 hourly / 220 daily / 300 BTC,
              raw fetch, no gap-filling)
  * "full"  = the training path (whole history, cleaner.clean, then the same
              indicators.compute)
then diff the 27 feature columns, the ensemble probability, and the resulting
ROC-AUC against the triple-barrier ground truth.

Run on the VM so the library versions match the production daemon:
    venv/bin/python diag_feature_parity.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
from data import cleaner
from features import indicators
from features.labels import HORIZON, ATR_TP_MULT, ATR_SL_MULT, _barrier_labels
from live.pipeline import load_assets, HOURLY_LOOKBACK, DAILY_LOOKBACK

PROB_CSV = config.STORAGE_LIVE / "prob_history.csv"

KLINE = "https://api.bybit.com/v5/market/kline"
COLS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
STRIDE = 4           # sample every Nth recorded hour
WARMUP_HOURS = 1200  # extra history before the window so "live" slices are full


def say(*a):
    print(*a, flush=True)


def fetch_range(symbol, interval, start, end):
    """Page backwards from `end` until `start` is covered. Bybit caps limit at 1000."""
    frames, cursor = [], end
    while True:
        r = requests.get(KLINE, params={
            "symbol": symbol, "interval": interval, "limit": 1000,
            "end": int(cursor.timestamp() * 1000),
        }, timeout=20)
        rows = r.json()["result"]["list"]
        if not rows:
            break
        f = pd.DataFrame(rows, columns=COLS)
        f["timestamp"] = pd.to_datetime(f["timestamp"].astype("int64"), unit="ms", utc=True)
        for c in COLS[1:]:
            f[c] = f[c].astype(float)
        frames.append(f)
        oldest = f["timestamp"].min()
        if oldest <= start:
            break
        cursor = oldest - pd.Timedelta(seconds=1)
        time.sleep(0.25)
    out = pd.concat(frames, ignore_index=True)
    out = out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return out.reset_index(drop=True)


def feats(hourly, daily, btc, feature_cols):
    """The shared tail of both pipelines: align, compute indicators, drop NaN rows."""
    ref = btc[["timestamp", "close"]].rename(columns={"close": "ref_close"})
    df = cleaner.align_daily_to_hourly(hourly, daily)
    df = indicators.compute(df, daily, ref_df=ref)
    return df.dropna(subset=feature_cols + ["atr_14"])


def main():
    probs = pd.read_csv(PROB_CSV, parse_dates=["timestamp"])
    probs = probs.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    probs = probs[probs["symbol"] == "ETHUSDT"].sort_values("timestamp").reset_index(drop=True)

    t_lo, t_hi = probs["timestamp"].min(), probs["timestamp"].max()
    say("=" * 78)
    say("  ETH train/live feature parity diagnostic")
    say("=" * 78)
    say(f"  live prob window : {t_lo} ~ {t_hi}  ({len(probs):,} rows)")

    feature_cols, fold_models = load_assets("ETHUSDT", config.LIVE_TARGET)
    say(f"  features {len(feature_cols)} | fold models {len(fold_models)}")

    h_start = t_lo - pd.Timedelta(hours=WARMUP_HOURS)
    h_end = t_hi + pd.Timedelta(hours=HORIZON + 5)
    say(f"  fetching hourly {h_start.date()} ~ {h_end.date()} ...")
    eth_h = fetch_range("ETHUSDT", "60", h_start, h_end)
    btc_h = fetch_range("BTCUSDT", "60", h_start, h_end)
    eth_d = fetch_range("ETHUSDT", "D", h_start - pd.Timedelta(days=400), h_end)
    say(f"  ETH 1h {len(eth_h):,} | BTC 1h {len(btc_h):,} | ETH 1D {len(eth_d):,}")

    # ---- full-history ("training") reconstruction -------------------------
    eth_h_c = cleaner.clean(eth_h.copy(), "60", "ETH-1h")
    btc_h_c = cleaner.clean(btc_h.copy(), "60", "BTC-1h")
    eth_d_c = cleaner.clean(eth_d.copy(), "D", "ETH-1D")
    full = feats(eth_h_c, eth_d_c, btc_h_c, feature_cols).set_index("timestamp")
    say(f"  full-history feature table {len(full):,} rows")

    # ---- ground-truth labels from the full history ------------------------
    import pandas_ta as ta
    lab = eth_h_c.copy()
    lab["atr_14"] = ta.atr(lab["high"], lab["low"], lab["close"], length=14)
    lab = lab.dropna(subset=["atr_14"]).reset_index(drop=True)
    c, a = lab["close"].values, lab["atr_14"].values
    lab["actual"] = _barrier_labels(lab["high"].values, lab["low"].values,
                                    tp=c + ATR_TP_MULT * a, sl=c - ATR_SL_MULT * a)
    truth = lab.set_index("timestamp")["actual"]

    # ---- point-in-time "live" reconstruction ------------------------------
    targets = probs["timestamp"].iloc[::STRIDE].tolist()
    say(f"  rebuilding live view at {len(targets)} timestamps (stride {STRIDE}h) ...")

    recs, skipped = [], 0
    for i, T in enumerate(targets):
        if T not in full.index:
            skipped += 1
            continue
        h = eth_h[eth_h["timestamp"] <= T].tail(HOURLY_LOOKBACK)
        b = btc_h[btc_h["timestamp"] <= T].tail(HOURLY_LOOKBACK)
        # daily bars whose 24h period closed before the daemon ran (~T+1h)
        d = eth_d[eth_d["timestamp"] + pd.Timedelta(hours=24)
                  <= T + pd.Timedelta(hours=1)].tail(DAILY_LOOKBACK)
        if len(h) < HOURLY_LOOKBACK or len(d) < 30:
            skipped += 1
            continue
        try:
            lf = feats(h, d, b, feature_cols)
        except Exception as exc:
            say(f"    [{T}] live rebuild failed: {exc}")
            skipped += 1
            continue
        if lf.empty:
            skipped += 1
            continue
        row = lf.iloc[-1]
        rec = {"timestamp": T, "live_row_ts": row["timestamp"],
               "aligned": row["timestamp"] == T}
        for name in feature_cols:
            rec["live__" + name] = float(row[name])
            rec["full__" + name] = float(full.loc[T, name])
        X_live = lf.iloc[[-1]][feature_cols]
        X_full = full.loc[[T], feature_cols]
        rec["prob_live"] = float(np.mean([m.predict_proba(X_live)[0, 1] for m in fold_models]))
        rec["prob_full"] = float(np.mean([m.predict_proba(X_full)[0, 1] for m in fold_models]))
        recs.append(rec)
        if (i + 1) % 50 == 0:
            say(f"    ... {i + 1}/{len(targets)}")

    R = pd.DataFrame(recs)
    say(f"  done {len(R)} points, skipped {skipped}")
    if R.empty:
        return
    say(f"  timestamp alignment: {int(R['aligned'].sum())}/{len(R)} exact")

    # ---- per-feature divergence ------------------------------------------
    say("")
    say("  === 27 features, live(300 bars) vs full history ===")
    say(f"  {'feature':<22} {'full mean':>12} {'med diff':>12} {'p95 diff':>12} {'rel err':>10} {'corr':>8}")
    say("  " + "-" * 82)
    diag = []
    for name in feature_cols:
        lv, fv = R["live__" + name], R["full__" + name]
        d = (lv - fv).abs()
        scale = fv.abs().median()
        rel = d.median() / scale if scale > 1e-12 else (0.0 if d.median() < 1e-12 else np.inf)
        corr = lv.corr(fv) if lv.std() > 0 and fv.std() > 0 else np.nan
        diag.append({"feature": name, "med": d.median(), "rel": rel, "corr": corr})
        say(f"  {name:<22} {fv.mean():>12.6f} {d.median():>12.6f} {d.quantile(0.95):>12.6f} "
            f"{rel:>9.2%} {corr:>8.4f}")

    bad = sorted([x for x in diag
                  if x["rel"] > 0.01 or (pd.notna(x["corr"]) and x["corr"] < 0.99)],
                 key=lambda x: -x["rel"])
    say("")
    if bad:
        say(f"  [!] {len(bad)} features diverge materially (rel >1% or corr <0.99):")
        for x in bad:
            say(f"      {x['feature']:<22} rel={x['rel']:>8.2%}  corr={x['corr']:.4f}")
    else:
        say("  [OK] all 27 features match -- the 300-bar lookback causes no drift")

    # ---- probability + AUC ------------------------------------------------
    m = R.merge(probs[["timestamp", "probability"]], on="timestamp", how="left")
    m = m.join(truth, on="timestamp").dropna(subset=["actual"])
    say("")
    say("  === probability and AUC ===")
    say(f"  verifiable samples {len(m):,} | base rate {m['actual'].mean():.1%}")
    say(f"  recorded live prob : mean {m['probability'].mean():.4f}  "
        f"range {m['probability'].min():.4f}~{m['probability'].max():.4f}")
    say(f"  rebuilt  live prob : mean {m['prob_live'].mean():.4f}  "
        f"range {m['prob_live'].min():.4f}~{m['prob_live'].max():.4f}")
    say(f"  full-history  prob : mean {m['prob_full'].mean():.4f}  "
        f"range {m['prob_full'].min():.4f}~{m['prob_full'].max():.4f}")
    say(f"  recorded vs rebuilt-live corr = {m['probability'].corr(m['prob_live']):.4f}"
        "   (should be ~1.0, else the rebuild is wrong)")
    say(f"  rebuilt-live vs full    corr = {m['prob_live'].corr(m['prob_full']):.4f}")

    from sklearn.metrics import roc_auc_score
    for label, col in [("recorded live", "probability"),
                       ("rebuilt live(300)", "prob_live"),
                       ("full history", "prob_full")]:
        try:
            say(f"  AUC {label:<20} = {roc_auc_score(m['actual'], m[col]):.4f}")
        except Exception as exc:
            say(f"  AUC {label:<20} = n/a ({exc})")

    out = config.STORAGE_LIVE / "feature_parity.csv"
    R.to_csv(out, index=False)
    say("")
    say(f"  detail saved -> {out}")
    say("=" * 78)


if __name__ == "__main__":
    main()
