# BYBIT_ML — ETH 機器學習交易訊號系統

[![Tests](https://github.com/jf33home940317-creator/bybit-ml/actions/workflows/test.yml/badge.svg)](https://github.com/jf33home940317-creator/bybit-ml/actions/workflows/test.yml)

用 XGBoost 預測 ETHUSDT 未來 24 小時走勢，自動推播交易訊號到 Discord，24/7 雲端運行。

---

## 成果

| 指標 | 數值 |
|---|---|
| 回測期間 | 2022-07 ~ 2026-05（約 3.9 年） |
| 訓練幣種 | ETHUSDT |
| 訊號門檻 | 機率 >= 0.75（影子門檻 0.70） |
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
  Bybit V5 API -> 1h / 日線 K 棒 + Funding Rate + Open Interest -> Parquet

Phase 2: 特徵工程（30+ 特徵）
  RSI / PPO / ATR / Bollinger Band / MA Bias / ROC
  時間週期（hour sin/cos、day of week）
  日線指標（防 look-ahead bias，shift +1 天）
  跨資產：BTC ROC_24（對 ETH 模型）

Phase 3: 模型訓練
  XGBoost 5-fold Purged Walk-Forward CV
  標籤：Triple-Barrier（TP = +3xATR, SL = -1.5xATR, 24h timeout）
  scale_pos_weight 處理類別不平衡

Phase 4: 回測
  OOF 機率掃描最佳門檻（Sharpe 最大化）
  DRC 風控組合回測（2% 固定風險, 最多 3 個並行倉）

Phase 5: 實盤訊號（24/7 Oracle Cloud VM）
  每小時 :01 執行 heartbeat
  已收盤 K 棒推論 -> prob >= 0.75 觸發
  風控三道防線（MDD / 連續虧損 / 單日上限）
  影子訊號（0.70-0.74）記錄但不開倉，用於門檻比較
  Discord 即時通知 7 種類型
  Paper trading ledger 記錄每筆損益
  Google Drive 每日自動備份

Phase 6: 特徵實驗（已完成）
  Funding Rate + Open Interest 6 個新特徵
  重訓後 Sharpe 下降 -> 自動回滾至原模型
  歷史資料已保留，供未來實驗使用
```

---

## 風控機制

| 防線 | 觸發條件 | 行為 |
|---|---|---|
| 回撤熔斷 | 帳戶淨值跌超 peak -15% | 暫停新訊號 + Discord 警告 |
| 連續虧損 | 最近 5 筆全輸 | 暫停新訊號 + Discord 警告 |
| 單日上限 | 當天虧損 > 帳戶 5% | 暫停新訊號 + Discord 警告 |
| Kill Switch | `touch .disabled` | 手動暫停訊號 |

> 所有風控只停新訊號，舊倉的 SL/TP/timeout 監控照常運作。

---

## 雙門檻策略

| prob 範圍 | 行為 |
|---|---|
| < 0.70 | 不動作 |
| 0.70 - 0.74 | 寫 shadow 到 ledger，不開倉，自動追蹤 SL/TP/timeout 結算 |
| >= 0.75 | 正式開倉 + Discord 通知 |

`show_results.py` 會顯示兩個門檻的勝率對比，累積足夠數據後可決定是否降門檻。

---

## Discord 通知（7 種）

```
1. 🚀 買入訊號（進場價、SL、TP、部位大小、帳戶淨值）
2. 🎯 止盈出場（P&L 金額）
3. 🛡️ 止損保護（P&L 金額）
4. 📋 倉位到期（24h timeout 結算）
5. 📊 每日健康心跳（24h 訊號數 / 勝率 / 損益 / 持倉）
6. 🚨 風控熔斷（三道防線觸發時）
7. ⚠️ 錯誤警告（pipeline 失敗，6h 僅推一次）
```

---

## 技術棧

| 類別 | 套件 |
|---|---|
| 資料來源 | [pybit](https://github.com/bybit-exchange/pybit) (Bybit V5 API) |
| 特徵工程 | pandas-ta 0.4, pandas, numpy 2.x |
| 模型 | xgboost, scikit-learn |
| 回測 | 自製 Triple-Barrier + DRC 模擬器 |
| 排程 | schedule |
| 通知 | Discord Webhook |
| 部署 | Oracle Cloud Free Tier (Ubuntu 22.04, systemd) |
| 備份 | rclone + Google Drive（cron 每日自動） |
| CI | GitHub Actions（push 自動跑 pytest） |
| 測試 | pytest（144 個測試） |

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

# 4. 抓歷史資料（K 棒 + Funding Rate + Open Interest）
python main.py

# 5. 建構特徵 + 訓練模型 + 回測
python build_features.py
python train_models.py
python run_backtest.py
python run_portfolio_backtest.py

# 6. 啟動實盤訊號 daemon
python run_live.py

# 7. 查看 paper trading 結果（含影子訊號對比）
python show_results.py
```

---

## 專案結構

```
bybit-ml/
├── config.py                   # 全局設定（幣種、路徑、風控參數、門檻）
├── run_live.py                 # 實盤 daemon（Phase 5 入口）
├── show_results.py             # Paper trading 結果 + 影子訊號對比
├── compare_models.py           # 新舊模型 Sharpe 比較 + 自動回滾
├── main.py                     # 歷史資料抓取（K 棒 + FR + OI）
├── build_features.py           # 特徵工程
├── train_models.py             # 模型訓練
├── run_backtest.py             # 單幣回測
├── run_portfolio_backtest.py   # DRC 組合回測
│
├── data/                       # 資料抓取 / 清洗 / 匯出
│   └── fetcher.py              #   K 棒 + Funding Rate + Open Interest
├── features/                   # 指標計算 / 標籤生成 / 驗證
│   └── indicators.py           #   30+ 技術指標 + FR/OI 衍生特徵
├── models/                     # XGBoost 訓練 / 評估 / 報告
├── backtest/                   # 回測引擎 / DRC 模擬器 / 報告
├── live/                       # 實盤模組
│   ├── fetcher.py              #   Bybit API 抓取（含 retry）
│   ├── pipeline.py             #   特徵計算 + 推論（partial-bar 過濾 + 新鮮度檢查）
│   ├── state.py                #   持倉狀態（crash-safe 原子寫入）
│   ├── ledger.py               #   交易帳本（crash-safe 原子寫入）
│   └── notifier.py             #   Discord 推播
│
├── tests/                      # pytest（144 個測試）
├── docs/superpowers/           # 設計文件 + 實作計畫
├── .github/workflows/test.yml  # GitHub Actions CI
├── deploy_oracle.ps1           # 一鍵部署到 Oracle VM
├── backup_ledger.ps1           # 本機手動備份 ledger
├── check_results.ps1           # 從本機查詢雲端結果
└── upload_final.ps1            # 補傳 validation reports
```

---

## 部署架構

```
本機 (Windows)                     Oracle Cloud VM (Ubuntu 22.04)
┌──────────────────┐              ┌─────────────────────────────┐
│ deploy_oracle.ps1│── scp/ssh ──>│ systemd: bybit-ml.service   │
│ backup_ledger.ps1│              │   └─ run_live.py (24/7)     │
│ check_results.ps1│              │       ├─ heartbeat :01/hr   │
└──────────────────┘              │       ├─ SL/TP/timeout 監控  │
                                  │       ├─ 影子訊號追蹤        │
GitHub                            │       ├─ 風控三道防線        │
┌──────────────────┐              │       └─ Discord 推播       │
│ Actions CI       │              │                             │
│ (push -> pytest) │              │ cron 00:30 UTC              │
└──────────────────┘              │   └─ rclone -> Google Drive │
                                  └─────────────────────────────┘
```

---

## Dashboard

即時監控面板，手機/電腦打開瀏覽器就能看：

**http://140.238.37.45:8501**

| 頁面 | 內容 |
|---|---|
| 概覽 | Prob 趨勢 + K 線圖（Prob 疊在右軸）+ 帳戶淨值 |
| 交易紀錄 | 正式/影子訊號列表 + 勝率 |
| 資金曲線 | 淨值變化 + MDD 標記 |
| Prob 分析 | 分佈直方圖 + 統計 + 每日最高 Prob |
| 系統狀態 | Daemon 心跳 + 回測基準 |

技術：Streamlit + Plotly，跑在同一台 Oracle VM，每 60 秒自動刷新。

---

## 注意事項

- `.env` 含 API Key，已被 `.gitignore` 排除，**不會出現在 repo**
- `storage/models/`（訓練好的 .pkl）因體積過大不上傳，需自行重訓
- 所有交易為 **Paper Trading**，非真實下單
- 風控閾值可在 `config.py` 調整（`MAX_DRAWDOWN_PCT`、`MAX_CONSECUTIVE_LOSSES`、`MAX_DAILY_LOSS_PCT`）
- 影子門檻可在 `config.py` 調整（`SHADOW_THRESHOLD`）
