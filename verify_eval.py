"""Sanity-check the live evaluation: does the recorded prob row line up with the
same OHLC bar we labelled?"""
import sys
import io
import numpy as np
import pandas as pd
import pandas_ta as ta

from features.labels import HORIZON, ATR_TP_MULT, ATR_SL_MULT, _barrier_labels
from eval_live_accuracy import load_probs, fetch_actual_ohlc

# eval_live_accuracy already rebound sys.stdout; wrap only if still raw.
if not isinstance(sys.stdout, io.TextIOWrapper) or sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

probs = load_probs()
ohlc = fetch_actual_ohlc(probs["timestamp"].min(), probs["timestamp"].max())

j = probs.merge(ohlc[["timestamp", "close"]], on="timestamp", how="inner", suffixes=("_logged", "_bybit"))
diff = (j["close_logged"] - j["close_bybit"]).abs()
rel = diff / j["close_bybit"]

print("=" * 60)
print("  對齊驗證：prob_history 記的 close vs Bybit 同一根 K 棒的 close")
print("=" * 60)
print(f"  比對筆數      : {len(j):,}")
print(f"  完全相同      : {(diff < 1e-6).sum():,} ({(diff < 1e-6).mean():.1%})")
print(f"  相對誤差 max  : {rel.max():.2e}")
print(f"  相對誤差 mean : {rel.mean():.2e}")
if (diff < 1e-6).mean() > 0.99:
    print("  -> 時間戳對齊正確，評估可信")
else:
    print("  -> 對齊有問題，評估不可信")
print()

# Label sanity: live base rate vs training-set positive rate
ohlc["atr_14"] = ta.atr(ohlc["high"], ohlc["low"], ohlc["close"], length=14)
o = ohlc.dropna(subset=["atr_14"]).reset_index(drop=True)
c, a = o["close"].values, o["atr_14"].values
o["actual"] = _barrier_labels(o["high"].values, o["low"].values,
                              tp=c + ATR_TP_MULT * a, sl=c - ATR_SL_MULT * a)
live_base = o["actual"].dropna().mean()

import json
rep = json.load(open("storage/features/ETHUSDT_validation_report.json"))
train_base = rep["class_balance"]["target_atr"]["positive_rate"]

print(f"  訓練集 target_atr 正例率 : {train_base:.1%}")
print(f"  本次驗證期正例率         : {live_base:.1%}")
print("  -> 標籤定義一致" if abs(live_base - train_base) < 0.10 else "  -> 標籤可能不一致")
print("=" * 60)
