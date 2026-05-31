# Phase 5 Live Signal (Paper Trading) Design Spec

## Goal
建立每小時心跳迴圈，從 Bybit 公開 API 抓取最新 K 線，重建特徵，對 ETHUSDT 模型進行集成推論，並透過 Discord Webhook 推播交易訊號，同時以 JSON 檔案追蹤虛擬持倉。

## Architecture

### 元件
- **`live/fetcher.py`** — Bybit V5 公開 API 抓取（無需 API Key）
- **`live/pipeline.py`** — 特徵重建 + 集成推論
- **`live/state.py`** — `active_positions.json` 讀寫（持倉追蹤）
- **`live/ledger.py`** — `paper_trading_ledger.json` 讀寫（交易紀錄）
- **`live/notifier.py`** — Discord Webhook 推播
- **`run_live.py`** — 心跳迴圈主程式（`schedule` 每小時 :01 執行）

### 僅 ETHUSDT
BTCUSDT 在 Phase 4.2 回測結果為負 Sharpe，不上線。BTCUSDT hourly 資料僅作為 ETHUSDT 的 cross-asset 特徵輸入（`cross_roc_24`）。

### 狀態持久化
- `storage/live/active_positions.json`：記錄目前最多 3 個開倉的 metadata
- `storage/live/paper_trading_ledger.json`：所有歷史開/平倉紀錄

### Discord 推播
使用 Webhook URL（不修改 music bot）。訊號觸發 → POST `/api/webhooks/...`。

## Signal Logic
```
1. 到期清倉：exit_time <= now 的倉位移出 active_positions，加入 ledger（outcome=timeout）
2. 併發檢查：若 count(active) >= 3 → 跳過
3. 特徵計算：fetch ETH hourly(300) + ETH daily(220) + BTC hourly(300)
4. 推論：ensemble average of 5 fold models
5. 閾值判斷：prob >= 0.75 → 開倉
6. 寫入 active_positions.json + ledger + Discord 推播
```

## Discord 訊息格式

**訊號觸發：**
```
[BYBIT_ML] 🚀 ETHUSDT 買入訊號
機率：0.7823 > 0.75
進場價：2,531.40
SL：2,468.32  |  TP：2,672.58
虛擬部位：$20,506 USD
預計出場：2026-05-31T11:01:00+00:00
```

**倉位到期：**
```
[BYBIT_ML] 📋 ETHUSDT 倉位到期
進場：2,531.40 @ 2026-05-30T11:01:00
結果：Timeout（24 小時）
```
