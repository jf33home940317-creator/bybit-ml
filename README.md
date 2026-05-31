# BYBIT_ML — ETH/BTC 機器學習交易訊號系統

用 XGBoost 預測 ETHUSDT 未來 24 小時走勢，自動推播交易訊號到 Discord，24/7 雲端運行。

---

## 成果

| 指標 | 數值 |
|---|---|
| 回測期間 | 2022–2024（約 2 年） |
| 訓練幣種 | ETHUSDT |
| 訊號門檻 | 機率 ≥ 0.75 |
| 回測總報酬 | **+98.3%** |
| Sharpe Ratio | **1.50** |
| 最大回撤 (MDD) | -13.3% |
| 回測勝率 | 55.1% |
| CAGR | 19.4% |

> BTC 因 Sharpe 為負排除上線。所有數字為 **Paper Trading**，非真實資金。

---

## 系統架構

```
Phase 1: 資料抓取
  └─ Bybit V5 API → 1h / 日線 K 棒 → Parquet 儲存

Phase 2: 特徵工程（30+ 特徵）
  └─ RSI / PPO / ATR / Bollinger Band / MA Bias / ROC
  └─ 時間週期（hour sin/cos、day of week）
  └─ 日線指標（防 look-ahead bias，shift +1 天）
  └─ 跨資產：BTC ROC_24（對 ETH 模型）

Phase 3: 模型訓練
  └─ XGBoost 5-fold Purged Walk-Forward CV
  └─ 標籤：Triple-Barrier（TP = +3×ATR, SL = -1.5×ATR, 24h timeout）
  └─ scale_pos_weight 處理類別不平衡

Phase 4: 回測
  └─ OOF 機率掃描最佳門檻（Sharpe 最大化）
  └─ DRC 風控組合回測（2% 固定風險, 最多 3 個並行倉）

Phase 5: 實盤訊號（24/7 Oracle Cloud VM）
  └─ 每小時 :01 執行 heartbeat
  └─ 已收盤 K 棒推論 → prob ≥ 0.75 觸發
  └─ Discord 即時通知：買入 / 止盈 / 止損 / 到期 / 每日健康心跳
  └─ Paper trading ledger 記錄每筆損益
```

---

## Discord 通知範例

```
🚀 ETHUSDT 買入訊號
機率：0.7821 > 0.75
進場價：2,041.3000
SL：1,998.5000  |  TP：2,163.2000
帳戶淨值：$1,000,000 USD
虛擬部位：$123,456 USD
預計出場：2026-06-02T10:00:00+00:00
```

```
📊 24h 健康心跳
過去 24h 訊號: 1 筆
已結算: 1 贏 / 0 輸
24h 損益: +$2,345 USD
當前持倉: 0 筆
```

---

## 技術棧

| 類別 | 套件 |
|---|---|
| 資料來源 | [pybit](https://github.com/bybit-exchange/pybit) (Bybit V5 API) |
| 特徵工程 | pandas-ta 0.4, pandas, numpy |
| 模型 | xgboost, scikit-learn |
| 回測 | 自製 Triple-Barrier + DRC 模擬器 |
| 排程 | schedule |
| 通知 | Discord Webhook |
| 部署 | Oracle Cloud Free Tier (Ubuntu 22.04, systemd) |
| 測試 | pytest（118 個測試） |

---

## 本機快速開始

```bash
# 1. Clone
git clone https://github.com/jf33home940317-creator/bybit-ml.git
cd bybit-ml

# 2. 建立環境（需要 Python 3.12）
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# 3. 設定 .env
cp .env.example .env
# 填入 BYBIT_API_KEY、BYBIT_API_SECRET、DISCORD_WEBHOOK_URL

# 4. 抓歷史資料
python main.py

# 5. 建構特徵 + 訓練模型 + 回測
python build_features.py
python train_models.py
python run_backtest.py
python run_portfolio_backtest.py

# 6. 啟動實盤訊號 daemon
python run_live.py

# 7. 查看 paper trading 結果
python show_results.py
```

---

## 專案結構

```
bybit-ml/
├── config.py                   # 全局設定（幣種、路徑、風控參數）
├── run_live.py                 # 實盤 daemon（Phase 5 入口）
├── show_results.py             # Paper trading 結果查詢
├── main.py                     # 歷史資料抓取
├── build_features.py           # 特徵工程
├── train_models.py             # 模型訓練
├── run_backtest.py             # 單幣回測
├── run_portfolio_backtest.py   # DRC 組合回測
│
├── data/                       # 資料抓取 / 清洗 / 匯出
├── features/                   # 指標計算 / 標籤生成 / 驗證
├── models/                     # XGBoost 訓練 / 評估 / 報告
├── backtest/                   # 回測引擎 / DRC 模擬器 / 報告
├── live/                       # 實盤模組（fetcher / pipeline / state / ledger / notifier）
│
├── tests/                      # pytest 測試（118 個）
├── deploy_oracle.ps1           # 一鍵部署到 Oracle VM
└── check_results.ps1           # 從本機查詢雲端 paper trading 結果
```

---

## 注意事項

- `.env` 含 API Key，已被 `.gitignore` 排除，**不會出現在 repo**
- `storage/models/`（訓練好的 .pkl）因體積過大不上傳，需自行重訓
- 所有交易為 **Paper Trading**，非真實下單
