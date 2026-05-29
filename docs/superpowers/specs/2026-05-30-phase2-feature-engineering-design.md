# Phase 2：特徵工程 — 設計文件

**日期：** 2026-05-30
**專案：** BYBIT_ML（虛擬貨幣價格趨勢預測系統）
**階段：** Phase 2 / 5

---

## 目標

基於 Phase 1 產出的雙週期 Parquet（BTCUSDT/ETHUSDT 1H+1D），計算多窗口技術指標（特徵 X）與三重屏障法標籤（目標 Y），輸出可供 Phase 3 模型訓練使用的特徵矩陣，並附帶基礎統計驗證報告。

---

## 目錄結構

```
BYBIT_ML/
├── features/
│   ├── __init__.py
│   ├── labels.py       # 三重屏障標籤邏輯
│   ├── indicators.py   # pandas-ta 多窗口特徵計算
│   ├── validator.py    # 標籤分布、特徵相關性統計
│   └── builder.py      # 總指揮：讀取→計算→驗證→儲存
│
├── build_features.py   # Phase 2 執行入口
│
└── storage/
    ├── raw/            # Phase 1 產出（已存在）
    └── features/       # Phase 2 產出
        ├── BTCUSDT_features.parquet
        ├── ETHUSDT_features.parquet
        └── validation_report.json
```

---

## 資料流

```
storage/raw/{symbol}_1h.parquet + _1d.parquet
    └─ cleaner.align_daily_to_hourly()       ← 重用 Phase 1，防 look-ahead bias
         └─ indicators.compute(df)            ← 加特徵欄位（產生頭部 NaN）
              └─ labels.compute(df)           ← 加標籤欄位（產生尾部 NaN）
                   └─ builder: replace(inf→nan) + dropna()   ← 頭尾斬首
                        └─ validator.report(df)               ← 統計驗證
                             └─ storage/features/*.parquet
```

---

## 模組設計

### labels.py

**職責：** 為每根 1H K 線產生三重屏障法標籤，完全向量化實作。

**對外介面：**
```python
def compute(df: pd.DataFrame) -> pd.DataFrame:
    # 在 df 上新增 target_fixed、target_atr 兩欄
    # 最後 HORIZON 根設為 NaN（尾部保護）
    # 回傳加上標籤欄位的 df
```

**固定屏障（`target_fixed`）：**
- 停利（TP）：`entry_close × 1.02`（+2%）
- 停損（SL）：`entry_close × 0.99`（-1%）
- 時間上限（Horizon）：24 根 1H

**ATR 動態屏障（`target_atr`）：**
- 停利（TP）：`entry_close + 3 × ATR(14)`（維持 1:2 盈虧比）
- 停損（SL）：`entry_close - 1.5 × ATR(14)`
- 時間上限：24 根 1H（同固定）

**標籤判定規則（兩組相同邏輯）：**
```
先撞 TP → 1
先撞 SL → 0
Horizon 到期未觸任一側 → 0
同根 K 線同時觸 TP 與 SL → 0（極端保守原則，SL 贏平局）
```

**向量化實作：**
```python
from numpy.lib.stride_tricks import sliding_window_view

# 一次建出未來 24 根的 2D 矩陣（避免 Python for loop）
future_highs = sliding_window_view(highs, HORIZON)  # shape: (n - HORIZON, HORIZON)
future_lows  = sliding_window_view(lows,  HORIZON)

# 找每筆交易的第一次觸碰位置
# argmin/argmax 全程 numpy，39,000 筆在 1 秒內完成
```

**尾部保護：**
```python
df.loc[df.index[-HORIZON:], ["target_fixed", "target_atr"]] = np.nan
```

---

### indicators.py

**職責：** 以 pandas-ta 為計算引擎，產生所有多窗口技術指標特徵。

**對外介面：**
```python
def compute(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    # 在 hourly_df 上計算 1H 指標，同時計算日線指標後 merge_asof 貼回
    # 回傳含所有特徵欄位的 DataFrame
```

**1H 特徵清單：**

| 類別 | 欄位名稱 | 窗口/參數 | 說明 |
|------|----------|-----------|------|
| 動量 | `rsi_7`, `rsi_14`, `rsi_24` | [7, 14, 24] | 短/標準/幣圈一日 |
| 動量 | `ppo`, `ppo_signal`, `ppo_hist` | (12, 26, 9) | PPO 取代 MACD，結果為百分比避免絕對值陷阱 |
| 波動度 | `atr_14`, `atr_24` | [14, 24] | 同時供 labels.py 的動態屏障使用 |
| 波動度 | `bband_width_20`, `bband_width_50` | (20,2)/(50,2.5) | `(upper-lower)/middle`，相對寬度 |
| 均線乖離 | `ma_bias_20`, `ma_bias_50`, `ma_bias_200` | [20, 50, 200] | `(close-SMA_N)/SMA_N`，標準化後可跨幣比較 |
| 成交量 | `vol_ratio_12`, `vol_ratio_24` | [12, 24] | `volume / rolling_mean(volume, N)` |
| 成交額 | `turnover_ratio_12`, `turnover_ratio_24` | [12, 24] | `turnover / rolling_mean(turnover, N)` |

**日線特徵（`daily_` 前綴）：**

先對 daily_df 獨立計算指標，再沿用 Phase 1 的 `align_daily_to_hourly()` 防護網（date_available = timestamp + 1D）貼回 1H。

| 欄位名稱 | 說明 |
|----------|------|
| `daily_rsi_14` | 日線 RSI(14) |
| `daily_atr_14` | 日線 ATR(14)，反映日級波動度 |
| `daily_ma_bias_20`, `daily_ma_bias_50`, `daily_ma_bias_200` | 日線均線乖離率 |

> **為何選 PPO 而非 MACD：** MACD 輸出絕對差值，BTC 在 20k 時的 histogram 約 +200，在 70k 時可達 +700，模型會誤判高價位動能較強。PPO 除以慢速 EMA 後輸出百分比（約 -5% 到 +5%），跨時間可比較。

---

### validator.py

**職責：** 對完整特徵矩陣做統計體檢，輸出報告，不自動刪除欄位。

**對外介面：**
```python
def report(df: pd.DataFrame, output_path: Path) -> None:
    # 印出統計摘要，存 validation_report.json
```

**驗證項目：**

1. **標籤分布**：計算 `target_fixed` 和 `target_atr` 各自的正例比例（label=1 的 %），若 < 20% 或 > 80% 則發出 WARNING（嚴重不平衡，模型將只學多數類）

2. **特徵與標籤相關性**：對每個特徵欄位計算 Point-Biserial Correlation（二元標籤的皮爾森相關係數），排序後印出，|r| > 0.05 視為有信號

3. **特徵間高共線性**：找出相關係數 |r| > 0.95 的特徵對，印出 WARNING（不自動刪除，由 Phase 3 Feature Importance 做最終裁決）

4. **NaN / inf 確認**：驗證 dropna + replace 後無任何 NaN 或 inf 殘留

**`validation_report.json` 格式：**
```json
{
  "metadata": {
    "symbol": "BTCUSDT",
    "total_rows": 38420,
    "feature_columns": ["rsi_7", "rsi_14", "rsi_24", "ppo", "ppo_signal", "ppo_hist", "..."],
    "target_columns": ["target_fixed", "target_atr"]
  },
  "class_balance": {
    "target_fixed": {"positive_rate": 0.41, "warning": null},
    "target_atr":   {"positive_rate": 0.38, "warning": null}
  },
  "correlations_with_target_fixed": {
    "rsi_14": 0.12,
    "ma_bias_200": -0.08,
    "..."
  },
  "correlations_with_target_atr": {
    "rsi_14": 0.09,
    "ma_bias_200": -0.06,
    "..."
  },
  "high_collinearity_warnings": [
    ["vol_ratio_12", "turnover_ratio_12", 0.97]
  ]
}
```

> **為何存 feature_columns/target_columns：** Phase 3 讀取此 JSON 即可動態取得正確欄位清單，不需硬編碼，實現全自動化。

---

### builder.py

**職責：** 總指揮，依序呼叫所有模組，完成一個幣對的完整特徵工程。

**對外介面：**
```python
def build(symbol: str) -> None:
```

**執行步驟：**
```python
def build(symbol: str) -> None:
    # 1. 讀取 Phase 1 Parquet
    hourly_df = pd.read_parquet(STORAGE_RAW / f"{symbol}_1h.parquet")
    daily_df  = pd.read_parquet(STORAGE_RAW / f"{symbol}_1d.parquet")

    # 2. 雙週期對齊（重用 Phase 1，防 look-ahead bias）
    df = cleaner.align_daily_to_hourly(hourly_df, daily_df)

    # 3. 計算技術指標特徵（產生頭部 NaN）
    df = indicators.compute(df, daily_df)

    # 4. 計算三重屏障標籤（產生尾部 NaN）
    df = labels.compute(df)

    # 5. 清理無限大與空值（順序不可調換）
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.dropna(inplace=True)

    # 6. 統計驗證（驗證 100% 完整資料）
    validator.report(df, output_path=STORAGE_FEATURES / "validation_report.json")

    # 7. 儲存特徵矩陣
    df.to_parquet(STORAGE_FEATURES / f"{symbol}_features.parquet", index=False)
```

---

### build_features.py

Phase 2 執行入口，對所有幣對逐一呼叫 builder：

```python
for symbol in config.SYMBOLS:
    builder.build(symbol)
```

---

## NaN 來源與處理策略

| NaN 來源 | 位置 | 原因 | 處理 |
|----------|------|------|------|
| 頭部 NaN | 前 199 根 | MA200 需要 199 根暖機 | dropna() 一刀斬 |
| 尾部 NaN | 後 24 根 | 三重屏障往未來看 24H | labels.py 強制設 NaN，dropna() 斬掉 |
| inf | 任意位置 | Volume Ratio 分母為 0 | replace(inf → nan) 後 dropna() |

> 不補值。頭部是「資料不存在」，尾部是「未來未發生」，任何填補都是造假。

---

## 測試策略

| 測試檔案 | 測試重點 |
|----------|----------|
| `tests/test_labels.py` | 已知價格序列驗證 TP/SL/時間到期/同根雙穿四種情境 |
| `tests/test_indicators.py` | PPO/RSI/ATR/Bollinger 寬度/MA乖離/Volume比率數值正確性 |
| `tests/test_builder.py` | 端對端：輸出無 NaN/inf、欄位完整、row 數合理 |

---

## 依賴套件

在 Phase 1 基礎上新增：

```
pandas-ta>=0.3.14b    # 技術指標計算引擎
scipy>=1.10.0         # Point-Biserial Correlation
```

---

## Phase 3 接口

`storage/features/{symbol}_features.parquet` 欄位結構：

- `timestamp`：UTC 時間
- 所有特徵欄位（從 `validation_report.json["metadata"]["feature_columns"]` 動態讀取）
- `target_fixed`、`target_atr`（0/1 二元標籤）

Phase 3 讀取 JSON 取得欄位清單後，直接分割 X / y 進行模型訓練。
