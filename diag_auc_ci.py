"""Is the live AUC of 0.444 actually below 0.50, or is it noise?

The triple-barrier label at hour t looks 24 hours ahead, so consecutive rows share
almost all of their outcome. Treating 2,220 hourly rows as 2,220 independent
observations would massively understate the error bar. This uses a moving-block
bootstrap with block length 24 (one full label horizon) to get an honest interval,
plus a per-month breakdown.

RESULT (2026-09-02, 2,197 live rows): point estimate 0.4298, 95% CI
[0.3542, 0.5091]. 0.50 sits inside the interval, so the model is noise, not a
reverse indicator. Monthly AUC swings 0.489 / 0.334 / 0.519 for the same reason.

Run on the VM:  venv/bin/python diag_auc_ci.py
"""
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config
import pandas_ta as ta
from data import cleaner
from features.labels import HORIZON, ATR_TP_MULT, ATR_SL_MULT, _barrier_labels

PROB_CSV = config.STORAGE_LIVE / "prob_history.csv"

KLINE = "https://api.bybit.com/v5/market/kline"
COLS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]
BLOCK = HORIZON      # one label horizon per block
N_BOOT = 4000
RNG = np.random.default_rng(42)


def say(*a):
    print(*a, flush=True)


def fetch_range(symbol, interval, start, end):
    frames, cursor = [], end
    while True:
        r = requests.get(KLINE, params={"symbol": symbol, "interval": interval,
                                        "limit": 1000, "end": int(cursor.timestamp() * 1000)},
                         timeout=20)
        rows = r.json()["result"]["list"]
        if not rows:
            break
        f = pd.DataFrame(rows, columns=COLS)
        f["timestamp"] = pd.to_datetime(f["timestamp"].astype("int64"), unit="ms", utc=True)
        for c in COLS[1:]:
            f[c] = f[c].astype(float)
        frames.append(f)
        if f["timestamp"].min() <= start:
            break
        cursor = f["timestamp"].min() - pd.Timedelta(seconds=1)
        time.sleep(0.25)
    out = pd.concat(frames, ignore_index=True).drop_duplicates(subset=["timestamp"])
    return out.sort_values("timestamp").reset_index(drop=True)


def auc(y, p):
    """Rank-based ROC-AUC; returns nan if only one class is present."""
    y = np.asarray(y)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    r = pd.Series(p).rank().values
    return (r[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def main():
    probs = pd.read_csv(PROB_CSV, parse_dates=["timestamp"])
    probs = probs.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    probs = probs[probs["symbol"] == "ETHUSDT"].sort_values("timestamp").reset_index(drop=True)

    lo, hi = probs["timestamp"].min(), probs["timestamp"].max()
    raw = fetch_range("ETHUSDT", "60", lo - pd.Timedelta(hours=60),
                      hi + pd.Timedelta(hours=HORIZON + 5))
    ohlc = cleaner.clean(raw, "60", "ETH-1h")
    ohlc["atr_14"] = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
    ohlc = ohlc.dropna(subset=["atr_14"]).reset_index(drop=True)
    c, a = ohlc["close"].values, ohlc["atr_14"].values
    ohlc["actual"] = _barrier_labels(ohlc["high"].values, ohlc["low"].values,
                                     tp=c + ATR_TP_MULT * a, sl=c - ATR_SL_MULT * a)

    m = probs.merge(ohlc[["timestamp", "actual"]], on="timestamp", how="inner")
    m = m.dropna(subset=["actual"]).sort_values("timestamp").reset_index(drop=True)

    y = m["actual"].values.astype(int)
    p = m["probability"].values
    n = len(m)

    say("=" * 74)
    say("  Is live AUC 0.444 distinguishable from 0.50?")
    say("=" * 74)
    say(f"  window {m['timestamp'].min()} ~ {m['timestamp'].max()}")
    say(f"  rows {n:,} | base rate {y.mean():.1%} | label horizon {HORIZON}h")
    say(f"  prob: mean {p.mean():.4f} std {p.std():.4f} min {p.min():.4f} max {p.max():.4f}")
    say(f"  independent blocks approx {n // BLOCK}")
    say("")

    point = auc(y, p)
    say(f"  point estimate AUC = {point:.4f}")

    # ---- moving-block bootstrap -------------------------------------------
    starts_pool = np.arange(0, n - BLOCK + 1)
    n_blocks = int(np.ceil(n / BLOCK))
    boots = np.empty(N_BOOT)
    for b in range(N_BOOT):
        s = RNG.choice(starts_pool, size=n_blocks, replace=True)
        idx = np.concatenate([np.arange(x, x + BLOCK) for x in s])[:n]
        boots[b] = auc(y[idx], p[idx])
    boots = boots[~np.isnan(boots)]
    q = np.quantile(boots, [0.025, 0.5, 0.975])
    say(f"  moving-block bootstrap (block={BLOCK}h, {len(boots):,} resamples)")
    say(f"    95% CI = [{q[0]:.4f}, {q[2]:.4f}]   median {q[1]:.4f}")
    say(f"    P(AUC < 0.50) = {(boots < 0.5).mean():.1%}")
    say(f"    0.50 inside CI: {'YES -> indistinguishable from random' if q[0] <= 0.5 <= q[2] else 'NO -> genuinely below random'}")

    # ---- per-month --------------------------------------------------------
    say("")
    say("  per month")
    say(f"  {'month':<10} {'n':>6} {'base':>7} {'AUC':>8} {'prob mean':>10} {'prob max':>9}")
    say("  " + "-" * 54)
    for k, g in m.groupby(m["timestamp"].dt.to_period("M")):
        say(f"  {str(k):<10} {len(g):>6,} {g['actual'].mean():>6.1%} "
            f"{auc(g['actual'].astype(int).values, g['probability'].values):>8.4f} "
            f"{g['probability'].mean():>10.4f} {g['probability'].max():>9.4f}")
    say("=" * 74)


if __name__ == "__main__":
    main()
