# Phase 4.1：向量化訊號回測 — 設計文件

**日期：** 2026-05-30
**專案：** BYBIT_ML（虛擬貨幣價格趨勢預測系統）
**階段：** Phase 4.1 / 5

---

## 目標

基於 Phase 3 訓練的 fold 模型，對 BTCUSDT/ETHUSDT × target_fixed/target_atr 四組組合執行 Out-of-Fold 向量化訊號回測，掃描進場機率閾值（0.50–0.80），找出最佳閾值並輸出資金曲線與 JSON 報告，供 Phase 5 實盤使用。

**核心假設：**
- 每筆訊號視為獨立交易（純訊號評估，不考慮部位上限）
- 手續費 0.1% 買入 + 0.1% 賣出 = 0.2% 雙邊
- 持倉邏輯：三重屏障出場（TP / SL / Timeout 24h）

---

## 目錄結構

```
BYBIT_ML/
├── backtest/
│   ├── __init__.py
│   ├── engine.py       # 向量化 P&L 計算 + 閾值掃描
│   └── reporter.py     # JSON 報告 + PNG 圖表
├── run_backtest.py     # Phase 4 執行入口
└── storage/
    └── backtest/
        ├── BTCUSDT_target_fixed_threshold_scan.json
        ├── BTCUSDT_target_fixed_threshold_tradeoff.png
        ├── BTCUSDT_target_fixed_optimal_equity.png
        ├── BTCUSDT_target_atr_*（同上）
        └── ETHUSDT_*（同上）
```

**config.py 新增：**
```python
STORAGE_BACKTEST = BASE_DIR / "storage" / "backtest"
```

---

## 資料流

```
features.parquet（含 OHLCV + atr_14 + 特徵 + targets）
    ↓ generate_oof_probabilities()
        fold1_model → val_idx_1 的 proba
        fold2_model → val_idx_2 的 proba
        ...（訓練期保持 NaN，確保無未來函數）
    ↓ run_threshold_scan(thresholds=np.arange(0.50, 0.81, 0.01))
        對每個 threshold：
            signal_indices = where(proba >= threshold)
            compute_trade_pnl() → pnl_array
            compute_metrics() → {win_rate, sharpe, drawdown, ...}
    ↓ reporter
        threshold_scan.json
        threshold_tradeoff.png
        optimal_equity.png
```

**關鍵：** `atr_14` 雖從 feature_columns 排除，但實體存在 parquet（46 欄）。target_atr 的動態 TP/SL 直接從此欄讀取，無需重算指標。

---

## 模組設計

### backtest/engine.py

#### `generate_oof_probabilities(df, feature_cols, fold_models) -> pd.Series`

對每個 fold 的驗證期 index 呼叫對應 fold 模型的 `predict_proba()[:,1]`，其餘 index 為 NaN。使用 `models.splitter.purged_walk_forward_split(len(df))` 獲取相同的 fold 切割，確保與訓練完全一致。

```
回傳：len(df) 的 Series，訓練期 = NaN，驗證期 = [0.0, 1.0] 機率值
```

#### `compute_trade_pnl(df, signal_indices, target, fee=0.002) -> pd.DataFrame`

對每個訊號位置 t，半向量化計算（list comprehension 建構 2D 矩陣，再 NumPy 向量化判斷）：

```python
future_highs = np.array([high_vals[i+1 : i+25] for i in signal_indices])
future_lows  = np.array([low_vals[i+1 : i+25] for i in signal_indices])
```

**TP/SL 規格（對應 labels.py 常數）：**

| target | TP | SL |
|--------|----|----|
| target_fixed | close × 1.02 | close × 0.99 |
| target_atr | close + 3.0 × atr_14 | close − 1.5 × atr_14 |

**出場判定（SL 平局勝，與 labels.py 一致）：**
- TP bar < SL bar：出場價 = TP 價，pnl = +tp_pct − fee
- SL bar ≤ TP bar：出場價 = SL 價，pnl = −sl_pct − fee
- 兩者皆未觸碰：Timeout，出場價 = close[t+24]，pnl = (exit−entry)/entry − fee

**邊界條件：** t + 25 超出 df 長度的訊號跳過（不計入回測）。

**回傳 DataFrame 欄位：** `entry_idx, timestamp, entry_price, exit_price, holding_bars, pnl, outcome`

#### `run_threshold_scan(df, feature_cols, fold_models, target, thresholds, fee=0.002, min_trades=20) -> dict`

1. 呼叫 `generate_oof_probabilities()` 一次
2. 對每個 threshold：
   - `signal_indices = np.where(proba >= threshold)[0]`
   - 若 `len(signal_indices) < min_trades`：跳過
   - 呼叫 `compute_trade_pnl()`
   - 計算指標（見下方）
3. 選出 sharpe_ratio 最大的閾值為 `optimal_threshold`
4. 回傳完整結果 dict（含 `optimal_equity` Series 供畫圖）

**每個閾值的指標：**

| 指標 | 計算方式 |
|------|---------|
| `n_trades` | len(signal_indices) |
| `win_rate` | sum(pnl > 0) / n_trades |
| `total_return_pct` | sum(pnl) |
| `avg_return_pct` | mean(pnl) |
| `sharpe_ratio` | mean(pnl)/std(pnl) × √(n_trades / total_years)，total_years = len(df) / 8760 |
| `max_drawdown_pct` | 從累積 pnl 曲線計算峰谷差最大值 |
| `avg_holding_bars` | mean(holding_bars) |

---

### backtest/reporter.py

#### `save_threshold_scan(results, symbol, target, output_dir)`

輸出 `{symbol}_{target}_threshold_scan.json`：

```json
{
  "symbol": "BTCUSDT",
  "target": "target_fixed",
  "fee_pct": 0.002,
  "horizon": 24,
  "optimal_threshold": 0.65,
  "optimal_metrics": {
    "n_trades": 234,
    "win_rate": 0.62,
    "total_return_pct": 0.45,
    "avg_return_pct": 0.0019,
    "sharpe_ratio": 1.23,
    "max_drawdown_pct": -0.08,
    "avg_holding_bars": 14.2
  },
  "threshold_scan": [
    {"threshold": 0.50, "n_trades": 1823, "win_rate": 0.52, ...},
    ...
  ]
}
```

#### `save_threshold_tradeoff_chart(results, symbol, target, output_dir)`

輸出 `{symbol}_{target}_threshold_tradeoff.png`：
- 雙 Y 軸折線圖：左軸 `win_rate`（藍線）、右軸 `sharpe_ratio`（橘線）
- X 軸：threshold 0.50–0.80
- 垂直虛線標出 `optimal_threshold`
- `n_trades < min_trades` 區間以灰色背景遮罩

#### `save_equity_curve(results, symbol, target, output_dir)`

輸出 `{symbol}_{target}_optimal_equity.png`：
- X 軸：進場 timestamp（datetime，從交易記錄的 timestamp 欄讀取）
- Y 軸：累積報酬率 % (`cumsum(pnl)`)
- 標題含 `threshold / n_trades / Sharpe`

---

### run_backtest.py

```python
for symbol in config.SYMBOLS:
    for target in ["target_fixed", "target_atr"]:
        results = engine.run_threshold_scan(df, feature_cols, fold_models, target)
        reporter.save_threshold_scan(results, symbol, target, backtest_dir)
        reporter.save_threshold_tradeoff_chart(results, symbol, target, backtest_dir)
        reporter.save_equity_curve(results, symbol, target, backtest_dir)
```

---

## 測試策略

| 測試 | 驗證內容 |
|------|---------|
| `test_oof_no_future_leak` | 訓練期 index 的 proba 全為 NaN |
| `test_oof_val_coverage` | 非 NaN 的 proba 總數 = 5 個驗證期 index 之和 |
| `test_pnl_tp_hit` | TP 先觸碰，pnl ≈ tp_pct − fee |
| `test_pnl_sl_hit` | SL 先觸碰，pnl ≈ −sl_pct − fee |
| `test_pnl_timeout` | 皆未觸碰，pnl = (close[t+24]−close[t])/close[t] − fee |
| `test_min_trades_filter` | n_trades < min_trades 的閾值不出現在 optimal_threshold 候選 |

---

## 依賴套件

Phase 3 已安裝，無需新增。使用：`numpy, pandas, matplotlib, joblib, xgboost`

---

## Phase 5 接口

- `threshold_scan.json` 的 `optimal_threshold` 供 Phase 5 直接讀取，決定實盤進場門檻
- `optimal_equity.png` 提供策略「性格」視覺驗證，確認獲利不集中於單一市場週期
