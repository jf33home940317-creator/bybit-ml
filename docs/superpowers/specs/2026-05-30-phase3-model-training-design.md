# Phase 3：模型訓練 — 設計文件

**日期：** 2026-05-30
**專案：** BYBIT_ML（虛擬貨幣價格趨勢預測系統）
**階段：** Phase 3 / 5

---

## 目標

基於 Phase 2 產出的特徵矩陣（BTCUSDT/ETHUSDT `features.parquet`），以 Purged Walk-Forward CV 訓練 XGBoost 二元分類模型，對 `target_fixed` 與 `target_atr` 各產出一組模型與評估報告，供 Phase 4 回測與實盤使用。

---

## 目錄結構

```
BYBIT_ML/
├── models/
│   ├── __init__.py
│   ├── splitter.py     # PurgedWalkForwardCV 生成器
│   ├── trainer.py      # 單 fold XGBoost 訓練
│   ├── evaluator.py    # F1/Precision/Recall/AUC 計算
│   ├── reporter.py     # JSON 報告 + Feature Importance PNG
│   └── builder.py      # 總指揮：讀取→分割→訓練→評估→儲存
│
├── train_models.py     # Phase 3 執行入口
│
└── storage/
    ├── features/       # Phase 2 產出（已存在）
    └── models/         # Phase 3 產出
        ├── BTCUSDT_target_fixed_fold1.pkl
        ├── BTCUSDT_target_fixed_fold2.pkl
        ├── ...
        ├── BTCUSDT_target_fixed_fold5.pkl
        ├── BTCUSDT_target_fixed_final.pkl
        ├── BTCUSDT_target_fixed_training_report.json
        ├── BTCUSDT_target_fixed_feature_importance.png
        └── （BTCUSDT_target_atr_* / ETHUSDT_* 同上）
```

---

## 資料流

```
storage/features/{symbol}_features.parquet
    └─ validation_report.json → feature_columns（動態讀取，不硬編碼）
         └─ splitter.PurgedWalkForwardCV(n_folds=5, gap=24)
              └─ [fold 1..5] trainer.train_fold()
                   └─ evaluator.evaluate_fold()
                        └─ reporter.save_report() + save_chart()
                             └─ trainer.train_final()（全量，mean best_iteration）
                                  └─ storage/models/*.pkl
```

---

## 模組設計

### splitter.py

**職責：** 產生 Purged Walk-Forward CV 的 `(train_idx, val_idx)` 索引對。

**對外介面：**
```python
def purged_walk_forward_split(
    n: int,
    n_folds: int = 5,
    gap: int = 24,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
```

**擴張視窗邏輯（n=33,809, n_folds=5, gap=24）：**

```
fold_size = n // (n_folds + 1)  ≈ 5,634

Fold 1: train[0 : 5634]        gap 24根    val[5658 : 11292]
Fold 2: train[0 : 11292]       gap 24根    val[11316 : 16950]
Fold 3: train[0 : 16950]       gap 24根    val[16974 : 22608]
Fold 4: train[0 : 22608]       gap 24根    val[22632 : 28242]
Fold 5: train[0 : 28242]       gap 24根    val[28266 : 33809]
```

gap = HORIZON（三重屏障往未來看的最大根數），確保訓練集標籤不洩漏至驗證集。

---

### trainer.py

**職責：** 以固定超參數訓練單一 fold 的 XGBoost 分類器；以 mean(best_iteration) 訓練全量最終模型。

**對外介面：**
```python
def train_fold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_eval: pd.DataFrame,
    y_eval: pd.Series,
) -> tuple[XGBClassifier, int]:
    # 回傳 (trained_model, best_iteration)

def train_final(
    X: pd.DataFrame,
    y: pd.Series,
    n_estimators: int,          # = round(mean(best_iterations))
) -> XGBClassifier:
    # 全量訓練，無 early stopping，固定樹數
```

**超參數（Baseline，固定不調）：**
```python
{
    "objective":             "binary:logistic",
    "n_estimators":          500,
    "learning_rate":         0.05,
    "max_depth":             4,
    "subsample":             0.8,
    "colsample_bytree":      0.8,
    "scale_pos_weight":      (負例數) / (正例數),  # 動態計算，每 fold 獨立
    "eval_metric":           "logloss",
    "early_stopping_rounds": 50,
    "random_state":          42,
}
```

Early stopping 用訓練集的**後 20%** 作為 eval set（在 train 範圍內，不跨 gap）。

> **為何動態計算 scale_pos_weight：** 固定寫死 3（75/25）在換幣種或時間區段後會失準。動態計算確保每個 fold 的類別權重永遠正確。

> **為何不調參：** Phase 3 是 Baseline 階段，目的是驗證特徵有效性。若 Baseline 指標達標，Phase 3.5 再引入 per-fold Optuna（Nested CV, n_trials=20-30）。

---

### evaluator.py

**職責：** 對單一 fold 的預測結果計算評估指標。

**對外介面：**
```python
def evaluate_fold(
    model: XGBClassifier,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> dict:
    # 回傳 {"precision", "recall", "f1", "roc_auc", "positive_rate_val", "best_iteration"}
```

**指標說明：**

| 指標 | 說明 | 為何不用 Accuracy |
|------|------|------------------|
| `precision` | 預測買進中，真正獲利的比例 | Accuracy 在 25% 正例下無意義（全猜 0 也有 75%） |
| `recall` | 真正獲利機會中，被模型抓到的比例 | — |
| `f1` | Precision 與 Recall 的調和平均 | 主要評估指標 |
| `roc_auc` | ROC 曲線面積，閾值無關的整體辨別力 | — |
| `positive_rate_val` | 驗證集正例比例 | 監控分布漂移 |

預測閾值固定 **0.5**（Baseline 階段）。原始 `predict_proba()` 機率存入報告，Phase 4 可用於閾值掃描（例如設 0.7 閾值提高 Precision 換取更低 Recall）而無需重新訓練。

---

### reporter.py

**職責：** 將訓練結果序列化為 JSON 報告與 Feature Importance PNG。

**對外介面：**
```python
def save_report(
    results: list[dict],        # 每個 fold 的評估結果
    feature_importance: dict,   # 平均 feature importance
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:

def save_feature_importance_chart(
    feature_importance: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
```

**`{symbol}_{target}_training_report.json` 格式：**
```json
{
  "symbol": "BTCUSDT",
  "target": "target_fixed",
  "n_folds": 5,
  "gap": 24,
  "mean_best_iteration": 287,
  "folds": [
    {
      "fold": 1,
      "precision": 0.52,
      "recall": 0.48,
      "f1": 0.50,
      "roc_auc": 0.61,
      "positive_rate_val": 0.245,
      "best_iteration": 312
    }
  ],
  "aggregate": {
    "precision": {"mean": 0.51, "std": 0.03},
    "recall":    {"mean": 0.47, "std": 0.04},
    "f1":        {"mean": 0.49, "std": 0.02},
    "roc_auc":   {"mean": 0.60, "std": 0.03}
  },
  "feature_importance": {
    "rsi_14": 0.18,
    "ma_bias_200": 0.15,
    "..."
  }
}
```

`mean_best_iteration` 存入 JSON，作為 `train_final()` 的樹數量依據，確保可覆現。

---

### builder.py

**職責：** 總指揮，對單一 `(symbol, target)` 組合執行完整訓練流程。

**對外介面：**
```python
def build(
    symbol: str,
    target: str,
    features_dir: Path = None,
    models_dir: Path = None,
) -> None:
```

**執行步驟：**
```python
def build(symbol, target, features_dir=None, models_dir=None):
    # 1. 讀取特徵矩陣
    df = pd.read_parquet(features_dir / f"{symbol}_features.parquet")
    feature_cols = json.load(open(features_dir / f"{symbol}_validation_report.json"))
                       ["metadata"]["feature_columns"]

    # 2. 分離 X / y
    X = df[feature_cols]
    y = df[target]

    # 3. Purged Walk-Forward CV（5 fold）
    fold_results = []
    fold_models = []
    for fold_idx, (train_idx, val_idx) in enumerate(
        purged_walk_forward_split(len(df)), start=1
    ):
        model, best_iter = train_fold(X.iloc[train_idx], y.iloc[train_idx],
                                      X.iloc[val_idx],   y.iloc[val_idx])
        metrics = evaluate_fold(model, X.iloc[val_idx], y.iloc[val_idx])
        metrics["best_iteration"] = best_iter
        fold_results.append(metrics)
        fold_models.append(model)
        joblib.dump(model, models_dir / f"{symbol}_{target}_fold{fold_idx}.pkl")

    # 4. 全量重訓 final model
    mean_iters = round(np.mean([r["best_iteration"] for r in fold_results]))
    final_model = train_final(X, y, n_estimators=mean_iters)
    joblib.dump(final_model, models_dir / f"{symbol}_{target}_final.pkl")

    # 5. 平均 feature importance（跨 5 fold）
    avg_importance = average_feature_importance(fold_models, feature_cols)

    # 6. 儲存報告與圖表
    save_report(fold_results, avg_importance, symbol, target, models_dir)
    save_feature_importance_chart(avg_importance, symbol, target, models_dir)
```

---

### train_models.py

Phase 3 執行入口：

```python
for symbol in config.SYMBOLS:
    for target in ["target_fixed", "target_atr"]:
        builder.build(symbol, target)
```

---

## 儲存結構

```
storage/models/
├── BTCUSDT_target_fixed_fold1.pkl  … fold5.pkl
├── BTCUSDT_target_fixed_final.pkl      ← 全量重訓，實盤用
├── BTCUSDT_target_fixed_training_report.json
├── BTCUSDT_target_fixed_feature_importance.png
├── BTCUSDT_target_atr_fold1.pkl  … fold5.pkl
├── BTCUSDT_target_atr_final.pkl
├── BTCUSDT_target_atr_training_report.json
├── BTCUSDT_target_atr_feature_importance.png
└── ETHUSDT_*（同上）
```

**fold1–5.pkl：** Phase 4 回測用，依時間區間切換對應模型，拼出無未來函數的完整資金曲線。
**final.pkl：** 以 `mean(best_iteration)` 在全量資料重訓，Phase 4 實盤預測用。

---

## 測試策略

| 測試檔案 | 測試重點 |
|----------|----------|
| `tests/test_splitter.py` | fold 數量正確、gap 確實空出、train/val 無重疊、expanding window 驗證 |
| `tests/test_trainer.py` | 回傳型別正確（XGBClassifier + int）、best_iteration 在合理範圍、動態 scale_pos_weight 計算正確 |
| `tests/test_evaluator.py` | 完美預測得 precision=1.0、全錯得 precision=0.0、roc_auc 在 [0,1] 內 |
| `tests/test_builder.py` | 端對端：pkl 檔案存在、JSON 含必要欄位、PNG 存在、fold 數正確 |

---

## 依賴套件

在 Phase 2 基礎上新增：

```
xgboost>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
joblib>=1.3.0
```

---

## Phase 4 接口

- **回測：** 載入 `fold{n}.pkl`，依時間區間切換模型，呼叫 `predict_proba()` 取機率，閾值由 Phase 4 決定
- **實盤：** 載入 `final.pkl`，呼叫 `predict_proba()`，以機率 > 閾值決定進場
- **訓練報告：** `training_report.json` 記錄 `mean_best_iteration`，確保全量重訓可完整覆現
