# ETF DCA ML — Design Spec

- **Date**: 2026-06-03
- **Status**: Approved (pending implementation plan)
- **Target repo**: `etf-dca-ml` (new, sibling to `BYBIT_ML`)
- **Predecessor lessons**: BYBIT_ML Phase 1-6 (data pipeline, walk-forward, live daemon, paper ledger)

## 1. Purpose

Build a "smart DCA" bot for stock/ETF: instead of mechanically buying on a fixed schedule, an ML model scores every trading day and the bot only deploys monthly budget on days predicted to be in the top 20% of the month by **forward 30-day Calmar ratio** (return / max drawdown).

**Decision boundary**: signal generation + paper trading ledger + Discord notification. Human places real orders manually.

## 2. Strategy Specification

### 2.1 Label definition

Top 20% of each natural month, ranked by forward-30-day Calmar (return / |max drawdown|).

```python
def compute_dca_labels(df, window=30, top_q=0.20):
    forward_return = df['close'].shift(-window) / df['close'] - 1
    rolling_future_low = df['low'].shift(-window).rolling(window, min_periods=1).min()
    forward_mdd = (rolling_future_low - df['close']) / df['close']
    forward_mdd = forward_mdd.clip(upper=0)
    perf_metric = forward_return / (np.abs(forward_mdd) + 1e-5)

    df['year_month'] = df['timestamp'].dt.to_period('M')
    df['month_rank_pct'] = df.groupby('year_month')['perf_metric'].rank(pct=True)
    df['label'] = (df['month_rank_pct'] >= (1.0 - top_q)).astype(int)
    return df
```

Rationale: relative ranking inside each month forces the model to learn timing alpha (not market beta). Calmar over plain return discriminates "smooth dip-buy" entries from "caught a falling knife" entries.

### 2.2 Execution rules (per symbol)

- Monthly budget `B` (asset-specific in `config.py`).
- Single-trade cap: `B / 4` (max 4 shots per month).
- Signal threshold: `prob >= τ` (default 0.55, grid-searched per symbol in [0.50, 0.55, 0.60, 0.65]).
- If `today == last_trading_day_of_month_for_symbol AND budget_left > 0` → `FORCED_BUY` of remaining budget.
- Budget resets on the 1st of each calendar month after retrain.

### 2.3 Why this design

- Pure dynamic accumulation (no monthly-1st baseline buy): in bear months, forced month-end buy will often land at lower prices than month-start buys. Backtest will validate.
- Asset-specific last-trading-day check: TW (春節) and US (Thanksgiving) calendars diverge — per-symbol determination from each symbol's own OHLCV index.

## 3. Data

### 3.1 Source

`yfinance` with `auto_adjust=True` (mandatory: stock dividends and splits must be back-adjusted, else technical features get artificial gaps).

### 3.2 Tickers

| Ticker | Role |
|---|---|
| `SPY`, `QQQ`, `0050.TW` | Targets (Phase 1 universe) |
| `^VIX` | US fear gauge |
| `^TNX` | US 10Y yield |
| `DX-Y.NYB` | US Dollar Index (drives EM outflows / TW foreign selling) |
| `TSM` or `SMH` | Overnight semi proxy for 0050.TW |

History: 2005-01-01 onward (~5000 daily bars).

### 3.3 Cleaner rules

- Each symbol keeps its **own** trading calendar (no outer-join across markets).
- Macro features (VIX/TNX/DXY/TSM) broadcast via `pd.merge_asof(direction='backward')`, after `to_datetime` + `sort_values`.
- Drop rows with zero volume; flag (do not auto-drop) bars with |daily return| > 50%.

## 4. Feature Engineering

Per symbol, ~22 features:

| Category | Features |
|---|---|
| Trend | `close/MA20`, `close/MA60`, `MA20/MA60` |
| Momentum | RSI(14), MACD histogram, ROC(5), ROC(20) |
| Volatility | ATR(14)/close, BB width(20), realized vol(20) |
| Position | 60-day channel position |
| Drawdown | DD from 60d high, DD from 252d high |
| Volume | volume / MA_vol(20), OBV slope |
| Seasonality | day_of_month, day_of_week |
| Macro (broadcast) | VIX level, VIX / VIX_MA200, VIX 5d change, TNX level, TNX slope, DXY level, DXY 5d change, TSM (or SMH) ROC_1 |

**Excluded**: `month_of_year` — only ~20 samples per month over 20 years; XGBoost overfits regime-specific events (e.g. 2008-09, 2020-03) as universal patterns.

**`features/builder.py` flow**:
```python
for symbol in DCA_SYMBOLS:
    tech = indicators.compute(raw[symbol])
    fmat = pd.merge_asof(tech, macro_df, on='date')
    feat[symbol] = fmat
labels[symbol] = compute_dca_labels(feat[symbol])
```

## 5. Model & Training

### 5.1 Model

XGBoost binary classifier (`objective='binary:logistic'`), one independent fit per symbol → `storage/models/{SYMBOL}.pkl`.

`scale_pos_weight = 4` (corrects 20/80 class imbalance; recalibrates output so 0.5 is the neutral cutoff and τ=0.55 carries genuine "above-average confidence" semantics).

### 5.2 Walk-forward retrain cadence

- **Monthly**, on the 1st calendar day at 06:00 TPE.
- Training set: 2005-01-01 to last completed trading day of previous month.
- Last 30 trading days of labels dropped (forward window incomplete).
- Hyperparameter selection: 5-fold time-series CV with **20-trading-day Purge Gap** between train and validation folds (prevents leakage from long-window features like MA60, 252d high).

### 5.3 Output artifacts

- `storage/models/{SYMBOL}.pkl` (overwrites prior month)
- `storage/models/{SYMBOL}_meta.json`: training date, sample count, CV scores, feature importance top-10

## 6. Backtest

### 6.1 Engine (`backtest/engine.py`)

Walk-forward: each month, the model used for prediction is trained on data up to the prior month-end. Zero look-ahead.

**Timing convention** (mirrors live daemon exactly):
- On the morning of trading day `d`, features are computed from data **through day `d-1` close**.
- The order, if any, fills at **day `d` open**.
- "Last trading day of month" means day `d` itself is the last session — the forced buy still fills at day `d`'s open, inside the same month.

```
for each trading day d in backtest window:
    for symbol in DCA_SYMBOLS:
        if d is first trading day of month for symbol:
            budget_left[symbol] = B[symbol]
        prob = model[symbol].predict(features[symbol, through d-1])
        if prob >= τ and budget_left[symbol] > 0:
            amount = min(B[symbol] / 4, budget_left[symbol])
            execute_buy(symbol, d_open, amount)
            budget_left[symbol] -= amount
        if d == last_trading_day_of_month_for(symbol) and budget_left[symbol] > 0:
            execute_buy(symbol, d_open, budget_left[symbol], reason="FORCED_BUY")
            budget_left[symbol] = 0
```

### 6.2 Baselines (mandatory comparison)

- **Baseline A**: classic DCA — full monthly budget on 1st trading day.
- **Baseline B**: weekly DCA — `B/4` every Monday (no timing).
- **Strategy**: model-driven engine above.

### 6.3 Metrics

- Total return, CAGR
- Sharpe, Sortino, Calmar
- Max drawdown
- **Cost basis vs monthly mean price** — the DCA-native metric: average effective entry price relative to that month's mean price, expressed as % savings vs Baseline B.
- Per-month bullet usage distribution (0/1/2/3/4 shots fired; how often FORCED_BUY triggered)

## 7. Live Deployment

### 7.1 Host

Reuse existing Oracle Cloud VM (same instance as BYBIT_ML, isolated project directory, isolated cron entries).

### 7.2 Cron schedule (Asia/Taipei)

| Time | Job | Script |
|---|---|---|
| 1st of month 06:00 | Walk-forward retrain | `live/monthly_retrain.py` |
| Daily 06:30 | Fetch → predict → notify → ledger append | `live/predict_daemon.py` |
| Daily 07:00 | Settle prior-day signals at today's open | `live/ledger_settle.py` |
| Sunday 23:00 | Weekly report to Discord | `live/weekly_report.py` |

06:30 (not 06:00) handles US Winter Time (EST) yfinance settlement lag: EST close is 05:00 TPE, and yfinance can take 60-90min to finalize the daily bar.

### 7.3 `predict_daemon.py` flow

```
1. Fetch latest yfinance increment (all tickers including macro)
2. Data integrity check: assert raw_df.index[-1] >= expected_last_session
   On failure: retry 3× with 10min spacing; on full failure: Discord alert + abort
3. For each symbol:
     features = build_latest_features(symbol)
     prob = model.predict_proba(features)[-1, 1]
     budget_left = read_budget_state(symbol)
     if prob >= τ[symbol] and budget_left > 0:
         amount = min(B[symbol] / 4, budget_left)
         append_signal(symbol, prob, amount, "BUY")
         update_budget(symbol, budget_left - amount)
     if cleaner.is_last_trading_day_of_month(symbol, today) and budget_left > 0:
         append_signal(symbol, prob, budget_left, "FORCED_BUY")
         update_budget(symbol, 0)
4. Compose single Discord message covering all symbols
5. notifier.send(message)
6. Write heartbeat
```

### 7.4 Discord message format

```
📅 2026-06-04 ETF DCA Signals (06:30 TPE)

🟢 SPY    prob=0.68  BUY $250 (remaining $250/mo)
⚪ QQQ    prob=0.42  no action (remaining $1000/mo)
🔴 0050   prob=0.31  no action (last trading day → FORCED_BUY NT$800)

Next run: tomorrow 06:30
```

### 7.5 `monthly_retrain.py`

```
1. Pull latest yfinance increment
2. For each symbol: refit XGBoost, overwrite pkl + meta.json
3. Reset budget_state.json: B[symbol] for every symbol
4. Discord: "✅ Monthly retrain complete. CV scores: ..."
```

### 7.6 Ledger

- `storage/live/predictions.jsonl`: one signal per line, atomic append (reused from BYBIT_ML).
- `storage/live/ledger.json`: positions, cost basis, marked-to-market.
- `storage/live/budget_state.json`: `{symbol: budget_left}` per month.
- Settlement basis: each signal is settled at the **open price of its target market's next trading session after the signal was emitted**. `ledger_settle.py` runs daily at 07:00 TPE and processes any pending signal whose fill bar is now available in yfinance — e.g. a Friday-morning US signal fills at Friday 21:30 TPE (US open) and is settled Monday 07:00; a Monday-morning TW signal fills at Monday 09:00 TPE and is settled Tuesday 07:00. The script does not need to know which day a signal was issued, only whether the fill bar exists.

### 7.7 Monitoring

- yfinance retry chain (3×, 10min spacing).
- Heartbeat miss > 25h → local `check_results.ps1` reports stale.
- Anomaly: all prob < 0.3 for N consecutive days → Discord warning (model possibly broken).

## 8. Repo Structure

```
etf-dca-ml/
├── config.py                    # DCA_SYMBOLS, budgets, thresholds, paths
├── data/
│   ├── fetcher.py               # yfinance with auto_adjust=True
│   └── cleaner.py               # per-symbol calendar, merge_asof, last-trading-day helper
├── features/
│   ├── indicators.py            # technical features
│   ├── macro.py                 # VIX/TNX/DXY/TSM derived features
│   ├── builder.py               # per-symbol assembly + macro broadcast
│   └── labels.py                # compute_dca_labels
├── models/
│   ├── trainer.py               # XGBoost fit per symbol
│   └── splitter.py              # walk-forward TS-CV with Purge Gap
├── backtest/
│   ├── engine.py                # monthly budget + signal-driven accumulation
│   └── metrics.py               # Sharpe, MDD, vs-baseline cost-basis comparison
├── live/
│   ├── predict_daemon.py
│   ├── ledger_settle.py
│   ├── monthly_retrain.py
│   ├── weekly_report.py
│   ├── notifier.py              # adapted from BYBIT_ML
│   └── ledger.py                # adapted from BYBIT_ML
├── storage/                     # .gitignore'd
│   ├── raw/                     # yfinance cache
│   ├── live/                    # predictions.jsonl, ledger.json, budget_state.json, heartbeat.json
│   └── models/                  # {SYMBOL}.pkl, {SYMBOL}_meta.json
├── tests/
├── README.md
└── .gitignore
```

## 9. Config (starter values)

```python
DCA_SYMBOLS = ["SPY", "QQQ", "0050.TW"]
MONTHLY_BUDGET = {"SPY": 1000, "QQQ": 1000, "0050.TW": 30000}    # USD, USD, TWD
SIGNAL_THRESHOLD = {"SPY": 0.55, "QQQ": 0.55, "0050.TW": 0.55}
MAX_PER_TRADE_RATIO = 0.25
HISTORY_START = "2005-01-01"
LABEL_WINDOW = 30
LABEL_TOP_Q = 0.20
PURGE_GAP_DAYS = 20
DISCORD_WEBHOOK_URL = os.environ["DCA_DISCORD_WEBHOOK"]
TZ = "Asia/Taipei"
```

## 10. Out of Scope (explicit)

- Real brokerage auto-execution (Shioaji / IBKR). Signal only; manual follow-up.
- TW chip data (foreign net buy/sell). Phase 2 enhancement.
- Multi-timeframe (intraday) signals. Daily granularity only.
- Crypto assets (kept in BYBIT_ML).
- Currency hedging on USD-denominated TWD budget.

## 11. Acceptance Criteria

- Backtest 2010–2026 walk-forward shows strategy Sharpe > both baselines on at least 2 of 3 symbols.
- Cost-basis savings vs Baseline B is positive on aggregate.
- Live daemon runs 30 consecutive days on VM without manual intervention, heartbeat never stale > 25h.
- Discord notifications fire daily and include data integrity status.
