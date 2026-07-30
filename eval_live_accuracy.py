"""Evaluate live model accuracy: compare recorded probabilities against what
actually happened (same triple-barrier label the model was trained on).

Needs the daemon's prob log, which lives on the VM and is gitignored:

    scp -i <key> ubuntu@<vm>:/home/ubuntu/bybit_ml/storage/live/prob_history.csv \\
        backup/prob_history_eval.csv

Then:  python eval_live_accuracy.py     (and verify_eval.py to check alignment)

2026-07-30 result: AUC 0.414 over 1,385 hours — the model was anti-predictive in
that regime. No money was lost; prob never reached the 0.75 entry threshold.
"""
import sys
import io
import numpy as np
import pandas as pd
import pandas_ta as ta

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

from live.fetcher import fetch_latest
from features.labels import HORIZON, ATR_TP_MULT, ATR_SL_MULT, _barrier_labels

PROB_CSV = "backup/prob_history_eval.csv"


def load_probs():
    df = pd.read_csv(PROB_CSV, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


def fetch_actual_ohlc(start, end):
    """Fetch enough hourly bars to cover [start, end] plus ATR warmup + horizon."""
    frames = []
    cursor = pd.Timestamp(end) + pd.Timedelta(hours=HORIZON + 5)
    # Bybit returns newest-first, max 1000 per call; walk backwards.
    while True:
        import requests
        r = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={
                "symbol": "ETHUSDT", "interval": "60", "limit": 1000,
                "end": int(cursor.timestamp() * 1000),
            },
            timeout=15,
        )
        body = r.json()
        rows = body["result"]["list"]
        if not rows:
            break
        f = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        f["timestamp"] = pd.to_datetime(f["timestamp"].astype("int64"), unit="ms", utc=True)
        for c in ["open", "high", "low", "close", "volume", "turnover"]:
            f[c] = f[c].astype(float)
        frames.append(f)
        oldest = f["timestamp"].min()
        if oldest <= pd.Timestamp(start) - pd.Timedelta(hours=60):
            break
        cursor = oldest - pd.Timedelta(hours=1)
    out = pd.concat(frames, ignore_index=True)
    return out.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)


def main():
    probs = load_probs()
    print("=" * 66)
    print("  BYBIT_ML 實盤模型準確度評估")
    print("=" * 66)
    print(f"  紀錄期間 : {probs['timestamp'].min()}  ->  {probs['timestamp'].max()}")
    print(f"  紀錄筆數 : {len(probs):,} 小時")
    print()

    ohlc = fetch_actual_ohlc(probs["timestamp"].min(), probs["timestamp"].max())
    ohlc["atr_14"] = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
    ohlc = ohlc.dropna(subset=["atr_14"]).reset_index(drop=True)

    closes = ohlc["close"].values
    atrs = ohlc["atr_14"].values
    label = _barrier_labels(
        ohlc["high"].values, ohlc["low"].values,
        tp=closes + ATR_TP_MULT * atrs,
        sl=closes - ATR_SL_MULT * atrs,
    )
    ohlc["actual"] = label

    m = probs.merge(ohlc[["timestamp", "actual", "atr_14"]], on="timestamp", how="inner")
    m = m.dropna(subset=["actual"])

    print(f"  可驗證筆數 : {len(m):,}（已知 24h 後結果）")
    if len(m) == 0:
        print("  尚無足夠資料")
        return

    base = m["actual"].mean()
    print(f"  實際基準率 : {base:.1%}  <- 隨便挑一小時進場，24h 內先觸 TP 的機率")
    print()

    # --- Discriminative power ---
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = roc_auc_score(m["actual"], m["probability"])
        brier = brier_score_loss(m["actual"], m["probability"])
    except Exception:
        auc = brier = float("nan")

    print(f"  ROC-AUC    : {auc:.4f}   (0.5=亂猜, >0.6 算有鑑別力)")
    print(f"  Brier score: {brier:.4f}   (越低越好, 0.25=瞎猜)")
    print()

    # --- Calibration ---
    print("  校準度（模型說幾成 vs 實際幾成）")
    print(f"  {'prob 區間':<14} {'筆數':>7} {'模型平均':>9} {'實際勝率':>9} {'誤差':>8}")
    print("  " + "-" * 52)
    bins = [0.0, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 1.01]
    for lo, hi in zip(bins[:-1], bins[1:]):
        s = m[(m["probability"] >= lo) & (m["probability"] < hi)]
        if len(s) == 0:
            continue
        gap = s["actual"].mean() - s["probability"].mean()
        print(f"  [{lo:.2f}, {hi:.2f})   {len(s):>7,} {s['probability'].mean():>9.1%} "
              f"{s['actual'].mean():>9.1%} {gap:>+8.1%}")
    print()

    # --- What the live thresholds would have done ---
    print("  各門檻表現（若真的照這門檻進場）")
    print(f"  {'門檻':<8} {'訊號數':>8} {'佔比':>7} {'實際勝率':>9} {'vs 基準':>9}")
    print("  " + "-" * 46)
    for thr in [0.55, 0.60, 0.62, 0.65, 0.70, 0.75]:
        s = m[m["probability"] >= thr]
        if len(s) == 0:
            print(f"  >={thr:.2f}   {0:>8} {0:>6.1%} {'—':>9} {'—':>9}")
            continue
        wr = s["actual"].mean()
        print(f"  >={thr:.2f}   {len(s):>8,} {len(s)/len(m):>6.1%} {wr:>9.1%} {wr-base:>+9.1%}")
    print()

    hi = m.nlargest(10, "probability")[["timestamp", "probability", "close", "actual"]]
    print("  最高 prob 的 10 個時點")
    print(f"  {'時間 (UTC)':<20} {'prob':>7} {'ETH':>9}  結果")
    print("  " + "-" * 48)
    for _, r in hi.iterrows():
        mark = "TP 命中" if r["actual"] == 1 else "沒到 TP"
        print(f"  {r['timestamp'].strftime('%Y-%m-%d %H:%M'):<20} {r['probability']:>7.4f} "
              f"{r['close']:>9.2f}  {mark}")
    print("=" * 66)


if __name__ == "__main__":
    main()
