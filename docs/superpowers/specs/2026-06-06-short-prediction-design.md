# Short Prediction: Mirror Triple-Barrier Models for ETH/BTC

## Goal

Train independent XGBoost short-side models for ETHUSDT and BTCUSDT using a mirrored
triple-barrier label, evaluate Sharpe / win-rate / MDD against the existing long
models, and produce a combined long+short equity report. Research-only — the live
daemon is **not** modified.

## Decision Log

- **Scope**: research artifacts only. `run_live.py`, `config.LIVE_TARGET`,
  `config.LIVE_SYMBOLS`, ledger, state, notifier, dashboard — all untouched.
- **Label**: mirror triple-barrier with same multipliers as long
  (TP=3×ATR, SL=1.5×ATR, 24h horizon).
- **Architecture choice**: add `direction: "long" | "short"` parameter to
  `labels.py`, `backtest/engine.py`, `backtest/simulator.py`. Existing long
  callers default to `"long"`. Chosen over duplicate-file mirroring (C) and
  string-suffix branching (B) for clean reuse + low regression risk.
- **target_fixed_short**: not implemented (YAGNI — live uses target_atr only).
- **Coins**: ETHUSDT + BTCUSDT both trained. BTC long had negative Sharpe;
  BTC short may have alpha on the opposite side.
- **Capital model for combined report**: two independent 1M sub-accounts
  in parallel (long + short = 2M total). Avoids cash-flow dependency that
  would confound the short model's standalone Sharpe.

## New Label

`features/labels.py` adds `target_atr_short`:

```python
df["target_atr_short"] = _barrier_labels(
    highs, lows,
    tp=closes - ATR_TP_MULT * atrs,    # 3×ATR drop = profit
    sl=closes + ATR_SL_MULT * atrs,    # 1.5×ATR rise = stop
    direction="short",
)
```

`_barrier_labels` gains `direction` parameter:

| direction | tp_hit | sl_hit |
|---|---|---|
| long | `future_highs >= tp` | `future_lows <= sl` |
| short | `future_lows <= tp` | `future_highs >= sl` |

SL still wins on tie (conservative pessimism) regardless of direction.
`target_fixed` short variant is **not** generated.

## Backtest Engine Changes

`backtest/engine.py compute_trade_pnl(...)` gains `direction` parameter:

- **YAGNI guard**: `target == "target_fixed" and direction == "short"` raises
  `NotImplementedError("Fixed target is not supported for short models")`.
- TP/SL price formulas branch on direction (signs flip for short).
- Hit-detection matrices branch on direction (highs/lows roles swap).
- P&L sign normalization:
  ```python
  sign = 1 if direction == "long" else -1
  tp_pct = sign * (tp_prices - entry_prices) / entry_prices   # ≥ 0
  sl_pct = sign * (sl_prices - entry_prices) / entry_prices   # ≤ 0
  timeout_pct = sign * (timeout_exit - entry_prices) / entry_prices
  ```
- **Invariant**: fee is always subtracted as a positive scalar
  (`pnl - fee`), never sign-multiplied. Round-trip fee is symmetric for
  long and short executions.

`backtest/engine.py run_threshold_scan(...)` accepts and forwards `direction`.

`backtest/simulator.py run_portfolio_simulation(...)` accepts and forwards
`direction`. `sl_distance = 1.5 * atr` formula is unchanged (the distance
is direction-agnostic — the position simply takes the opposite side).
DRC loop, concurrency, position sizing logic are unmodified.

**Direction derivation helper** (avoids every caller passing both args):

```python
def _direction_from_target(target: str) -> str:
    return "short" if target.endswith("_short") else "long"
```

Used as default inside `engine.py` / `simulator.py` when `direction` is None.

## Entry-Point Changes

| File | Change |
|---|---|
| `train_models.py` | TARGETS list adds `"target_atr_short"` |
| `run_backtest.py` | TARGETS list adds `"target_atr_short"` |
| `run_portfolio_backtest.py` | COMBINATIONS gains short tuples for ETH + BTC |
| `build_features.py` | **no change** — `labels.compute()` auto-adds column |
| `models/trainer.py`, `models/builder.py` | **no change** — target-agnostic |
| `features/validator.py` | **no change** — validates feature_columns only |
| `live/*` | **no change** |

## New Files

### `combine_long_short.py`

For each symbol ∈ {ETHUSDT, BTCUSDT}:

1. Load long `portfolio_report.json` (closed_trades + equity_log).
2. Load short `portfolio_report.json` (closed_trades + equity_log).
3. Treat as two independent 1M sub-accounts (initial combined equity = 2M).
4. Merge closed_trades by `entry_idx`, accumulate P&L on the combined
   equity timeline, produce `combined_equity_log`.
5. Compute combined metrics: Sharpe, MDD%, MDD$, CAGR, total return.
6. Compute **daily return correlation** between long-leg and short-leg
   equity curves: resample each `equity_log` to daily (forward-fill flat
   between trades), compute `pct_change()` on each, drop NaN, then
   `long_daily_ret.corr(short_daily_ret)`.
7. Write `storage/backtest/{SYMBOL}_long_short_combined.json`:

```json
{
  "symbol": "ETHUSDT",
  "initial_combined_equity": 2000000.0,
  "final_combined_equity": ...,
  "long_final_equity": ...,
  "short_final_equity": ...,
  "combined_metrics": {
    "total_return_pct": ...,
    "cagr_pct": ...,
    "sharpe_ratio": ...,
    "max_drawdown_pct": ...,
    "max_drawdown_usd": ...
  },
  "daily_return_correlation": ...,
  "n_long_trades": ...,
  "n_short_trades": ...
}
```

### `compare_long_short.py`

Terminal-only report. Reads:

- `{SYMBOL}_target_atr_threshold_scan.json` × 2
- `{SYMBOL}_target_atr_short_threshold_scan.json` × 2
- `{SYMBOL}_long_short_combined.json` × 2

Prints side-by-side table:

```
                  ETHUSDT     BTCUSDT
Long  Sharpe      1.50        -0.42
Long  Win%        55.1%       45.8%
Long  Trades      N           N
Short Sharpe      ?           ?
Short Win%        ?           ?
Short Trades      ?           ?
Combined Sharpe   ?           ?
Combined MDD%     ?           ?
Long-Short ρ      ?           ?
```

No file output.

## Execution Order

```
build_features.py           # rebuild parquet with target_atr_short column
train_models.py             # +10 fold models (2 coins × 5 folds × short)
run_backtest.py             # +2 threshold scan jsons
run_portfolio_backtest.py   # +2 portfolio reports
combine_long_short.py       # +2 combined reports
compare_long_short.py       # terminal report
```

## Test Plan

All new tests; existing 144 tests untouched.

| Test file | New cases |
|---|---|
| `tests/test_labels.py` | `test_compute_short_mirror_barrier` — ATR=10, close=100 fixture: low drops 30 → label 1; high rises 15 only → label 0; same-bar tie → label 0 (SL wins) |
| `tests/test_backtest_engine.py` | `test_compute_trade_pnl_short_direction` — mirror existing long fixture, assert P&L sign matches expectation, fee deducted as absolute scalar |
| `tests/test_backtest_engine.py` | `test_fixed_short_raises_not_implemented` — YAGNI guard |
| `tests/test_backtest_engine.py` | `test_threshold_scan_short_direction` — short prob array flows through correctly |
| `tests/test_simulator.py` | `test_portfolio_simulation_short` — DRC runs with short, sl_distance / position_qty stay positive |
| `tests/test_combine_long_short.py` (new) | (a) merged final_equity = long final + short final; (b) correlation computed correctly; (c) high-positive-corr fixture → ρ near 1; high-negative-corr fixture → ρ near −1 |

## File Outputs

```
storage/features/{ETHUSDT,BTCUSDT}_features.parquet         # +target_atr_short column
storage/models/{ETHUSDT,BTCUSDT}_target_atr_short_fold{1..5}.pkl
storage/models/{ETHUSDT,BTCUSDT}_target_atr_short_final.pkl
storage/models/{ETHUSDT,BTCUSDT}_target_atr_short_training_report.json
storage/backtest/{ETHUSDT,BTCUSDT}_target_atr_short_threshold_scan.json
storage/backtest/{ETHUSDT,BTCUSDT}_target_atr_short_portfolio_report.json
storage/backtest/{ETHUSDT,BTCUSDT}_long_short_combined.json
```

## Out of Scope

- Live trading integration (no `LIVE_TARGET` change, no daemon edits)
- Discord notifications for short signals
- Dashboard pages for short prob trends
- Shared-account multi-direction DRC (deferred — decided after seeing
  standalone short Sharpe results)
- `target_fixed_short`
- Re-training existing long models (no feature change)

## Decision Tree (post-run)

After `compare_long_short.py` output:

- **Short Sharpe > 0 AND Combined MDD < Long MDD** → strong candidate for
  future live integration; spin off a new plan to wire short signals
  into `run_live.py`.
- **Short Sharpe weak BUT Combined Sharpe > Long Sharpe** → short model
  provides drawdown protection even if standalone unprofitable; consider
  smaller-size live integration.
- **Combined Sharpe ≤ Long Sharpe AND ρ > 0.5** → short adds noise
  without diversification; archive models, document negative result,
  no live changes.
