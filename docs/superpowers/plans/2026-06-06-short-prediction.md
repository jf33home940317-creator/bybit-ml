# Short Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add mirrored triple-barrier short-side prediction for ETHUSDT and BTCUSDT, evaluate via threshold scan + portfolio backtest, produce combined long+short report with daily-return correlation. Research-only — `run_live.py` and `config.LIVE_TARGET` untouched.

**Architecture:** Add `direction: "long" | "short"` parameter to `features/labels.py`, `backtest/engine.py`, `backtest/simulator.py`. Existing callers default to `"long"` (no regression). New `target_atr_short` label flows automatically through trainer, backtest, and portfolio simulator. Combined report sums two independent 1M sub-accounts (long + short) and computes daily-return correlation.

**Tech Stack:** Python 3.12, pandas, numpy, xgboost, pytest. No new dependencies.

---

## File Map

**Modify:**
- `features/labels.py` — add `direction` to `_barrier_labels`, add `target_atr_short` column
- `backtest/engine.py` — add `direction` to `compute_trade_pnl` + `run_threshold_scan`, YAGNI guard for `target_fixed` short
- `backtest/simulator.py` — add `direction` to `run_portfolio_simulation`
- `backtest/reporter.py` — extend `save_portfolio_report` to include `closed_trades` + `equity_log`
- `train_models.py` — add `target_atr_short` to TARGETS list
- `run_backtest.py` — add `target_atr_short` to TARGETS list
- `run_portfolio_backtest.py` — extend COMBINATIONS with short tuples

**Create:**
- `combine_long_short.py` — merge long+short equity, write combined report json
- `compare_long_short.py` — terminal-only side-by-side metrics table
- `tests/test_combine_long_short.py` — new tests
- (existing test files get new test methods, not new files)

**Storage outputs (auto-generated):**
- `storage/features/{ETHUSDT,BTCUSDT}_features.parquet` — gains `target_atr_short` column
- `storage/models/{ETHUSDT,BTCUSDT}_target_atr_short_fold{1..5}.pkl` + `_final.pkl` + `_training_report.json`
- `storage/backtest/{ETHUSDT,BTCUSDT}_target_atr_short_threshold_scan.json` + charts
- `storage/backtest/{ETHUSDT,BTCUSDT}_target_atr_short_portfolio_report.json` + charts
- `storage/backtest/{ETHUSDT,BTCUSDT}_long_short_combined.json`

---

## Task 1: Add `direction` parameter to `_barrier_labels` + new `target_atr_short` column

**Files:**
- Modify: `features/labels.py`
- Test: `tests/test_labels.py`

- [ ] **Step 1: Write failing test for short label TP hit**

Add to `tests/test_labels.py` inside `class TestLabels`:

```python
    def test_atr_short_tp_hit_label_1(self):
        """Short TP = close - 3*ATR. Bar 5 low drops below TP → label=1."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        lows[5] = 96.9          # 100 - 3*1 = 97 → low 96.9 hits short TP
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_atr_short"].iloc[0] == 1.0

    def test_atr_short_sl_hit_label_0(self):
        """Short SL = close + 1.5*ATR. Bar 3 high above SL → label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[3] = 101.6        # 100 + 1.5*1 = 101.5 → high 101.6 hits short SL
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_atr_short"].iloc[0] == 0.0

    def test_atr_short_intra_bar_collision_sl_wins(self):
        """Same bar hits both short TP and short SL → SL wins (label=0)."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        lows[2] = 96.9          # hits short TP
        highs[2] = 101.6        # hits short SL same bar
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_atr_short"].iloc[0] == 0.0

    def test_atr_short_column_present_with_nan_tail(self):
        """target_atr_short must exist with NaN tail like other labels."""
        from features.labels import compute
        n = HORIZON + 10
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert "target_atr_short" in result.columns
        assert result["target_atr_short"].iloc[-HORIZON:].isna().all()
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_labels.py -v -k short`
Expected: 4 FAIL with `KeyError: 'target_atr_short'` (column does not exist yet)

- [ ] **Step 3: Modify `_barrier_labels` to accept `direction`**

In `features/labels.py`, replace the existing `_barrier_labels` function:

```python
def _barrier_labels(
    highs: np.ndarray,
    lows: np.ndarray,
    tp: np.ndarray,
    sl: np.ndarray,
    direction: str = "long",
) -> np.ndarray:
    n = len(highs)
    n_valid = n - HORIZON
    labels = np.full(n, np.nan)
    if n_valid <= 0:
        return labels

    future_highs = sliding_window_view(highs[1:], HORIZON)
    future_lows  = sliding_window_view(lows[1:],  HORIZON)

    if direction == "long":
        tp_hit = future_highs >= tp[:n_valid, np.newaxis]
        sl_hit = future_lows  <= sl[:n_valid, np.newaxis]
    elif direction == "short":
        tp_hit = future_lows  <= tp[:n_valid, np.newaxis]
        sl_hit = future_highs >= sl[:n_valid, np.newaxis]
    else:
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")

    tp_first = np.where(tp_hit.any(axis=1), np.argmax(tp_hit, axis=1), HORIZON)
    sl_first = np.where(sl_hit.any(axis=1), np.argmax(sl_hit, axis=1), HORIZON)

    # SL wins ties regardless of direction (conservative pessimism)
    wins_tp = (sl_first > tp_first) & (tp_first < HORIZON)
    labels[:n_valid] = np.where(wins_tp, 1.0, 0.0)
    return labels
```

- [ ] **Step 4: Add `target_atr_short` to `compute()`**

In `features/labels.py`, modify `compute()`:

```python
def compute(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    closes = df["close"].values
    highs = df["high"].values
    lows = df["low"].values
    atrs = df["atr_14"].values

    df["target_fixed"] = _barrier_labels(
        highs, lows,
        tp=closes * (1.0 + TP_PCT_FIXED),
        sl=closes * (1.0 - SL_PCT_FIXED),
        direction="long",
    )
    df["target_atr"] = _barrier_labels(
        highs, lows,
        tp=closes + ATR_TP_MULT * atrs,
        sl=closes - ATR_SL_MULT * atrs,
        direction="long",
    )
    df["target_atr_short"] = _barrier_labels(
        highs, lows,
        tp=closes - ATR_TP_MULT * atrs,
        sl=closes + ATR_SL_MULT * atrs,
        direction="short",
    )
    return df
```

- [ ] **Step 5: Run all label tests to verify pass**

Run: `pytest tests/test_labels.py -v`
Expected: all green (existing 6 + new 4 = 10 passed)

- [ ] **Step 6: Commit**

```bash
git add features/labels.py tests/test_labels.py
git commit -m "feat(labels): add direction param + target_atr_short mirror label"
```

---

## Task 2: Add `direction` to `compute_trade_pnl` + YAGNI guard

**Files:**
- Modify: `backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing tests for short-direction P&L**

Add to `tests/test_backtest_engine.py` inside `class TestComputeTradePnl` (or as new class at file bottom — match existing structure):

```python
    def test_short_pnl_tp_hit(self):
        """Short TP at close - 3*ATR. Bar 1 low <= TP → pnl = +3*ATR_pct - fee."""
        df = _make_df(100, close=1000.0)
        # ATR = 10 (close * 0.01). Short TP = 1000 - 30 = 970.
        # Bar 1: low=965 (≤ 970), high=1005 (< short SL 1015) → TP first.
        df.loc[1, 'high'] = 1005.0
        df.loc[1, 'low']  = 965.0

        trades = compute_trade_pnl(df, signal_indices=[0],
                                    target='target_atr', direction='short')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'tp'
        # (1000 - 970) / 1000 = 0.03 → pnl = 0.03 - 0.002 = 0.028
        assert abs(row['pnl'] - 0.028) < 1e-9
        assert row['holding_bars'] == 1

    def test_short_pnl_sl_hit(self):
        """Short SL at close + 1.5*ATR. Bar 1 high >= SL → pnl = -1.5*ATR_pct - fee."""
        df = _make_df(100, close=1000.0)
        # ATR = 10. Short SL = 1015.
        # Bar 1: high=1016 (≥ SL), low=975 (> TP 970) → SL first.
        df.loc[1, 'high'] = 1016.0
        df.loc[1, 'low']  = 975.0

        trades = compute_trade_pnl(df, signal_indices=[0],
                                    target='target_atr', direction='short')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'sl'
        # (1000 - 1015) / 1000 = -0.015 → pnl = -0.015 - 0.002 = -0.017
        assert abs(row['pnl'] - (-0.017)) < 1e-9

    def test_short_fee_is_absolute_not_sign_multiplied(self):
        """Fee deducted as positive scalar regardless of direction."""
        df = _make_df(100, close=1000.0)
        df.loc[1, 'high'] = 1005.0
        df.loc[1, 'low']  = 965.0
        trades = compute_trade_pnl(df, signal_indices=[0],
                                    target='target_atr', direction='short',
                                    fee=0.005)
        # raw = +0.03, with fee=0.005 → 0.025. NOT 0.03 - (-0.005) = 0.035.
        assert abs(trades.iloc[0]['pnl'] - 0.025) < 1e-9

    def test_fixed_short_raises_not_implemented(self):
        """YAGNI guard: target_fixed + direction=short → NotImplementedError."""
        df = _make_df(100, close=1000.0)
        with pytest.raises(NotImplementedError, match="Fixed target"):
            compute_trade_pnl(df, signal_indices=[0],
                              target='target_fixed', direction='short')

    def test_long_direction_default_unchanged(self):
        """Direction defaults to 'long' — existing call sites stay working."""
        df = _make_df(100, close=1000.0)
        df.loc[1, 'high'] = 1025.0
        df.loc[1, 'low']  = 995.0
        # No direction arg → long behavior
        trades = compute_trade_pnl(df, signal_indices=[0], target='target_fixed')
        assert trades.iloc[0]['outcome'] == 'tp'
        assert abs(trades.iloc[0]['pnl'] - 0.018) < 1e-9
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_backtest_engine.py -v -k "short or fixed_short or absolute or default_unchanged"`
Expected: most FAIL (unexpected keyword `direction` or wrong outcome)

- [ ] **Step 3: Add `_direction_from_target` helper at top of `backtest/engine.py`**

Add after the imports:

```python
def _direction_from_target(target: str) -> str:
    """Derive direction from target name suffix. Used as default when caller
    does not pass `direction` explicitly."""
    return "short" if target.endswith("_short") else "long"
```

- [ ] **Step 4: Modify `compute_trade_pnl` to accept and use `direction`**

Replace `compute_trade_pnl` in `backtest/engine.py`:

```python
def compute_trade_pnl(
    df: pd.DataFrame,
    signal_indices: list,
    target: str,
    fee: float = 0.002,
    direction: str = None,
) -> pd.DataFrame:
    """Semi-vectorized P&L: list-comprehension builds 2D matrix, NumPy broadcasts.

    Signals within 25 bars of the end of df are skipped (no complete horizon).
    SL wins on ties (same bar as TP).

    direction: "long" (default for backwards-compat) or "short". If None,
    derived from target name (suffix "_short" → "short").
    """
    if direction is None:
        direction = _direction_from_target(target)
    if direction not in ("long", "short"):
        raise ValueError(f"direction must be 'long' or 'short', got {direction!r}")
    if target == "target_fixed" and direction == "short":
        raise NotImplementedError(
            "Fixed target is not supported for short models (YAGNI — use target_atr)"
        )

    HORIZON = 24
    max_valid = len(df) - HORIZON - 1
    signal_indices = [i for i in signal_indices if i <= max_valid]

    if not signal_indices:
        return pd.DataFrame(
            columns=['entry_idx', 'timestamp', 'entry_price',
                     'exit_price', 'holding_bars', 'pnl', 'outcome']
        )

    high_vals  = df['high'].values
    low_vals   = df['low'].values
    close_vals = df['close'].values
    atr_vals   = df['atr_14'].values
    sig_arr    = np.array(signal_indices, dtype=int)

    entry_prices = close_vals[sig_arr]

    future_highs = np.array([high_vals[i + 1 : i + 1 + HORIZON] for i in signal_indices])
    future_lows  = np.array([low_vals[ i + 1 : i + 1 + HORIZON] for i in signal_indices])

    if target == 'target_fixed':
        # direction == "long" guaranteed by guard above
        tp_prices = entry_prices * 1.02
        sl_prices = entry_prices * 0.99
    else:  # target_atr (long or short)
        if direction == "long":
            tp_prices = entry_prices + 3.0 * atr_vals[sig_arr]
            sl_prices = entry_prices - 1.5 * atr_vals[sig_arr]
        else:  # short
            tp_prices = entry_prices - 3.0 * atr_vals[sig_arr]
            sl_prices = entry_prices + 1.5 * atr_vals[sig_arr]

    # Direction sign normalizes pct so profit is always positive
    sign = 1 if direction == "long" else -1
    tp_pct = sign * (tp_prices - entry_prices) / entry_prices   # ≥ 0
    sl_pct = sign * (sl_prices - entry_prices) / entry_prices   # ≤ 0

    if direction == "long":
        tp_hit = future_highs >= tp_prices[:, None]
        sl_hit = future_lows  <= sl_prices[:, None]
    else:  # short
        tp_hit = future_lows  <= tp_prices[:, None]
        sl_hit = future_highs >= sl_prices[:, None]

    tp_first = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), HORIZON)
    sl_first = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), HORIZON)

    # SL wins on tie (same as before)
    tp_wins = tp_first < sl_first
    sl_wins = (~tp_wins) & sl_hit.any(axis=1)

    timeout_exit = close_vals[sig_arr + HORIZON]
    timeout_pct  = sign * (timeout_exit - entry_prices) / entry_prices

    # Fee is ALWAYS subtracted as a positive scalar regardless of direction
    pnl_arr = np.where(tp_wins, tp_pct - fee,
              np.where(sl_wins, sl_pct - fee,
                       timeout_pct - fee))

    holding_arr = np.where(tp_wins, tp_first + 1,
                  np.where(sl_wins, sl_first + 1, HORIZON))

    outcome_arr = np.where(tp_wins, 'tp',
                  np.where(sl_wins, 'sl', 'timeout'))

    exit_price_arr = np.where(tp_wins, tp_prices,
                     np.where(sl_wins, sl_prices, timeout_exit))

    return pd.DataFrame({
        'entry_idx':    sig_arr,
        'timestamp':    df['timestamp'].values[sig_arr],
        'entry_price':  entry_prices,
        'exit_price':   exit_price_arr,
        'holding_bars': holding_arr,
        'pnl':          pnl_arr,
        'outcome':      outcome_arr,
    })
```

- [ ] **Step 5: Run all backtest engine tests to verify pass**

Run: `pytest tests/test_backtest_engine.py -v`
Expected: all green (existing + 5 new)

- [ ] **Step 6: Commit**

```bash
git add backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat(backtest): add direction param to compute_trade_pnl + YAGNI guard for fixed_short"
```

---

## Task 3: Add `direction` to `run_threshold_scan`

**Files:**
- Modify: `backtest/engine.py`
- Test: `tests/test_backtest_engine.py`

- [ ] **Step 1: Write failing test for short direction scan**

Add to `tests/test_backtest_engine.py`:

```python
class TestRunThresholdScanShort:
    def test_threshold_scan_short_runs_without_error(self):
        """Threshold scan with direction='short' returns same shape as long."""
        from backtest.engine import run_threshold_scan
        df = _make_df(600, close=1000.0)
        # Manufacture a price drop at bar 1 of each potential signal so short TP hits
        df.loc[1::24, 'low'] = 965.0  # short TP price = 970
        fold_models = [_ConstantModel(0.85)] * 5
        result = run_threshold_scan(
            df, feature_cols=[], fold_models=fold_models,
            target='target_atr', direction='short',
            thresholds=np.array([0.50, 0.70, 0.80]),
            min_trades=1,
        )
        assert 'threshold_scan' in result
        assert 'optimal_threshold' in result
        assert result['optimal_threshold'] in (0.50, 0.70, 0.80, None)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_backtest_engine.py::TestRunThresholdScanShort -v`
Expected: FAIL with `unexpected keyword argument 'direction'`

- [ ] **Step 3: Modify `run_threshold_scan` to accept and forward `direction`**

In `backtest/engine.py`, replace `run_threshold_scan` signature and `compute_trade_pnl` calls:

```python
def run_threshold_scan(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
    target: str,
    thresholds: np.ndarray = None,
    fee: float = 0.002,
    min_trades: int = 20,
    direction: str = None,
) -> dict:
    """Scan signal thresholds and return metrics + optimal threshold by Sharpe.

    direction: "long" or "short". If None, derived from target name.
    """
    if direction is None:
        direction = _direction_from_target(target)

    if thresholds is None:
        thresholds = np.round(np.linspace(0.50, 0.80, 31), 2)

    proba = generate_oof_probabilities(df, feature_cols, fold_models)
    total_years = len(df) / 8760.0

    scan_results = []
    best_sharpe = -np.inf
    optimal_threshold = None
    optimal_trades_df = None

    for thr in thresholds:
        thr = round(float(thr), 2)
        signal_indices = np.where(proba >= thr)[0].tolist()
        if len(signal_indices) < min_trades:
            continue

        trades_df = compute_trade_pnl(df, signal_indices, target, fee,
                                        direction=direction)
        if len(trades_df) < min_trades:
            continue

        pnl_vals = trades_df['pnl'].values
        n_trades = len(pnl_vals)
        mean_pnl = float(pnl_vals.mean())
        std_pnl  = float(pnl_vals.std(ddof=1))
        sharpe   = float(mean_pnl / std_pnl * np.sqrt(n_trades / total_years)) if std_pnl > 0 else 0.0

        cumsum      = np.cumsum(pnl_vals)
        running_max = np.maximum.accumulate(cumsum)
        max_dd      = float((cumsum - running_max).min())

        metrics = {
            'threshold':        thr,
            'n_trades':         n_trades,
            'win_rate':         round(float((pnl_vals > 0).sum() / n_trades), 4),
            'total_return_pct': round(float(pnl_vals.sum()), 6),
            'avg_return_pct':   round(mean_pnl, 6),
            'sharpe_ratio':     round(sharpe, 4),
            'max_drawdown_pct': round(max_dd, 6),
            'avg_holding_bars': round(float(trades_df['holding_bars'].mean()), 2),
        }
        scan_results.append(metrics)

        if sharpe > best_sharpe:
            best_sharpe       = sharpe
            optimal_threshold = thr
            optimal_trades_df = trades_df

    return {
        'threshold_scan':    scan_results,
        'optimal_threshold': optimal_threshold,
        'optimal_metrics':   next((r for r in scan_results if r['threshold'] == optimal_threshold), None),
        'optimal_trades_df': optimal_trades_df,
        'total_years':       total_years,
    }
```

- [ ] **Step 4: Run engine tests to verify pass**

Run: `pytest tests/test_backtest_engine.py -v`
Expected: all green

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat(backtest): forward direction param to run_threshold_scan"
```

---

## Task 4: Add `direction` to `run_portfolio_simulation`

**Files:**
- Modify: `backtest/simulator.py`
- Test: `tests/test_simulator.py`

- [ ] **Step 1: Write failing test for portfolio short**

Add to `tests/test_simulator.py`:

```python
class TestPortfolioShort:
    def test_short_direction_runs_drc(self):
        """Portfolio sim with direction='short' executes DRC trades with positive position_qty."""
        df = _make_df(300)
        # Manufacture price drops so short TP fires
        df.loc[1::25, 'low'] = df['close'] * 0.96   # below short TP (close - 3% nominal)
        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_atr',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=2,
            direction='short',
        )

        assert results['executed_trades'] >= 1
        for t in results['closed_trades']:
            assert t['position_qty'] > 0, "position_qty must be positive regardless of direction"
            assert t['sl_distance'] > 0,  "sl_distance must be positive (1.5*ATR magnitude)"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_simulator.py::TestPortfolioShort -v`
Expected: FAIL with `unexpected keyword argument 'direction'`

- [ ] **Step 3: Modify `run_portfolio_simulation` to accept and use `direction`**

In `backtest/simulator.py`, update imports and signature:

```python
import numpy as np
import pandas as pd
from backtest.engine import generate_oof_probabilities, compute_trade_pnl, _direction_from_target


def run_portfolio_simulation(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
    target: str,
    optimal_threshold: float,
    initial_equity: float = 1_000_000,
    risk_pct: float = 0.02,
    max_concurrent: int = 3,
    direction: str = None,
) -> dict:
    """DRC portfolio simulation. direction defaults to derivation from target."""
    if direction is None:
        direction = _direction_from_target(target)

    # ── Phase 1：預計算 ──────────────────────────────────────────────
    proba = generate_oof_probabilities(df, feature_cols, fold_models)
    signal_indices = np.where(proba >= optimal_threshold)[0].tolist()
    total_signals = len(signal_indices)

    raw_trades_df = compute_trade_pnl(df, signal_indices, target,
                                       direction=direction)
    total_signals = len(raw_trades_df)
    if total_signals == 0:
        return _empty_results(initial_equity, len(signal_indices))

    raw_trades_df = raw_trades_df.copy()
    raw_trades_df['exit_bar'] = (
        raw_trades_df['entry_idx'] + raw_trades_df['holding_bars']
    ).astype(int)
    raw_trades_df['atr_at_entry'] = (
        df['atr_14'].values[raw_trades_df['entry_idx'].values.astype(int)]
    )
    raw_trades_df = raw_trades_df.sort_values('entry_idx').reset_index(drop=True)

    # ── Phase 2：事件驅動順序迴圈 ────────────────────────────────────
    equity: float = initial_equity
    open_slots: list = []
    closed_trades: list = []
    equity_log: list = []
    skipped: int = 0

    for row in raw_trades_df.itertuples(index=False):
        to_close = [t for t in open_slots if t['exit_bar'] <= row.entry_idx]
        to_close.sort(key=lambda t: t['exit_bar'])
        for t in to_close:
            equity += t['pnl_usd']
            equity_log.append((t['exit_bar'], equity))
            closed_trades.append({**t, 'trade_roe': t['pnl_usd'] / t['equity_at_entry']})
        open_slots = [t for t in open_slots if t['exit_bar'] > row.entry_idx]

        if len(open_slots) >= max_concurrent:
            skipped += 1
            continue

        # sl_distance is always a positive magnitude (1.5*ATR) regardless of direction
        if target == 'target_fixed':
            sl_distance = row.entry_price * 0.01
        else:
            sl_distance = 1.5 * row.atr_at_entry

        if sl_distance <= 0:
            skipped += 1
            continue

        risk_budget  = equity * risk_pct
        position_qty = risk_budget / sl_distance
        position_usd = position_qty * row.entry_price
        pnl_usd      = position_usd * row.pnl

        open_slots.append({
            'entry_idx':       row.entry_idx,
            'exit_bar':        row.exit_bar,
            'timestamp':       row.timestamp,
            'outcome':         row.outcome,
            'entry_price':     row.entry_price,
            'exit_price':      row.exit_price,
            'atr_at_entry':    row.atr_at_entry,
            'sl_distance':     sl_distance,
            'position_qty':    position_qty,
            'position_usd':    position_usd,
            'pnl_pct':         row.pnl,
            'pnl_usd':         pnl_usd,
            'equity_at_entry': equity,
        })

    for t in sorted(open_slots, key=lambda t: t['exit_bar']):
        equity += t['pnl_usd']
        equity_log.append((t['exit_bar'], equity))
        closed_trades.append({**t, 'trade_roe': t['pnl_usd'] / t['equity_at_entry']})

    executed = len(closed_trades)
    metrics = _compute_metrics(closed_trades, equity_log, initial_equity, equity, len(df))

    return {
        'closed_trades':     closed_trades,
        'equity_log':        equity_log,
        'final_equity':      equity,
        'total_signals':     total_signals,
        'executed_trades':   executed,
        'skipped_signals':   skipped,
        'metrics':           metrics,
        'initial_equity':    initial_equity,
        'risk_pct':          risk_pct,
        'max_concurrent':    max_concurrent,
        'optimal_threshold': optimal_threshold,
        'direction':         direction,
    }
```

- [ ] **Step 4: Run simulator tests to verify pass**

Run: `pytest tests/test_simulator.py -v`
Expected: all green

- [ ] **Step 5: Commit**

```bash
git add backtest/simulator.py tests/test_simulator.py
git commit -m "feat(backtest): add direction param to run_portfolio_simulation"
```

---

## Task 5: Extend `save_portfolio_report` to include `closed_trades` + `equity_log`

Combine_long_short needs the raw trade list and equity timeline; current report only saves metrics. Make it additive.

**Files:**
- Modify: `backtest/reporter.py`
- Test: new test cases added to `tests/` (see step 1) — no existing test file covers reporter; add minimal one.

- [ ] **Step 1: Create test file with failing test**

Create `tests/test_reporter.py`:

```python
import json
from pathlib import Path
from backtest.reporter import save_portfolio_report


def _sample_results():
    return {
        'closed_trades': [
            {'entry_idx': 0, 'exit_bar': 24, 'pnl_usd': 100.0, 'pnl_pct': 0.01,
             'equity_at_entry': 1_000_000, 'trade_roe': 0.0001, 'timestamp': '2024-01-01T00:00:00',
             'outcome': 'tp', 'entry_price': 1000.0, 'exit_price': 1010.0,
             'atr_at_entry': 10.0, 'sl_distance': 15.0, 'position_qty': 100.0,
             'position_usd': 100_000},
        ],
        'equity_log': [(24, 1_000_100.0)],
        'final_equity': 1_000_100.0,
        'total_signals': 1,
        'executed_trades': 1,
        'skipped_signals': 0,
        'metrics': {'sharpe_ratio': 1.0},
        'initial_equity': 1_000_000.0,
        'risk_pct': 0.02,
        'max_concurrent': 3,
        'optimal_threshold': 0.75,
    }


def test_portfolio_report_includes_closed_trades_and_equity_log(tmp_path):
    save_portfolio_report(_sample_results(), 'ETHUSDT', 'target_atr', tmp_path)
    path = tmp_path / 'ETHUSDT_target_atr_portfolio_report.json'
    data = json.loads(path.read_text())
    assert 'closed_trades' in data
    assert 'equity_log' in data
    assert len(data['closed_trades']) == 1
    assert data['equity_log'] == [[24, 1_000_100.0]]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_reporter.py -v`
Expected: FAIL — `closed_trades` not in saved JSON

- [ ] **Step 3: Modify `save_portfolio_report` to include the two fields**

In `backtest/reporter.py`, replace `save_portfolio_report`:

```python
def save_portfolio_report(
    results: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """Write portfolio_report.json for the given symbol/target combination.

    Includes closed_trades + equity_log so downstream tools (combine_long_short)
    can reconstruct the equity timeline without re-running the simulation.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        'symbol':            symbol,
        'target':            target,
        'initial_equity':    results['initial_equity'],
        'final_equity':      results['final_equity'],
        'risk_pct':          results['risk_pct'],
        'max_concurrent':    results['max_concurrent'],
        'optimal_threshold': results['optimal_threshold'],
        'total_signals':     results['total_signals'],
        'executed_trades':   results['executed_trades'],
        'skipped_signals':   results['skipped_signals'],
        'metrics':           results['metrics'],
        'closed_trades':     results['closed_trades'],
        'equity_log':        [list(pair) for pair in results['equity_log']],
    }

    path = output_dir / f"{symbol}_{target}_portfolio_report.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)
```

- [ ] **Step 4: Run reporter test to verify pass**

Run: `pytest tests/test_reporter.py -v`
Expected: PASS

- [ ] **Step 5: Run full test suite to verify nothing else regressed**

Run: `pytest tests/ -v`
Expected: all green

- [ ] **Step 6: Commit**

```bash
git add backtest/reporter.py tests/test_reporter.py
git commit -m "feat(reporter): persist closed_trades + equity_log in portfolio JSON"
```

---

## Task 6: Wire `target_atr_short` into entry-point scripts

**Files:**
- Modify: `train_models.py`
- Modify: `run_backtest.py`
- Modify: `run_portfolio_backtest.py`

- [ ] **Step 1: Update `train_models.py`**

Replace the `for target` line:

```python
import logging
import config
from models import builder

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
)

if __name__ == "__main__":
    for symbol in config.SYMBOLS:
        for target in ["target_fixed", "target_atr", "target_atr_short"]:
            builder.build(symbol, target)
```

- [ ] **Step 2: Update `run_backtest.py`**

Modify the inner `for target` loop (target_fixed_short still skipped — only ATR has short variant):

```python
        for target in ["target_fixed", "target_atr", "target_atr_short"]:
            logger.info(f"[{symbol}][{target}] Loading fold models...")
            fold_paths = [models_dir / f"{symbol}_{target}_fold{k}.pkl" for k in range(1, 6)]
            missing = [p for p in fold_paths if not p.exists()]
            if missing:
                logger.error(f"[{symbol}][{target}] Missing models: {missing}. Run Phase 3 first.")
                continue
            fold_models = [joblib.load(p) for p in fold_paths]

            logger.info(f"[{symbol}][{target}] Running threshold scan...")
            results = engine.run_threshold_scan(df, feature_cols, fold_models, target)
            # direction auto-derived from target name (suffix "_short")
            opt = results['optimal_threshold']
            m   = results['optimal_metrics']
            logger.info(
                f"[{symbol}][{target}] optimal_threshold={opt}, "
                f"n_trades={m['n_trades'] if m else 'N/A'}, "
                f"sharpe={m['sharpe_ratio'] if m else 'N/A'}"
            )

            reporter.save_threshold_scan(results, symbol, target, backtest_dir)
            reporter.save_threshold_tradeoff_chart(results, symbol, target, backtest_dir)
            reporter.save_equity_curve(results, symbol, target, backtest_dir)
            logger.info(f"[{symbol}][{target}] Saved to {backtest_dir}")
```

- [ ] **Step 3: Update `run_portfolio_backtest.py`**

Replace the `COMBINATIONS` constant near the top:

```python
COMBINATIONS = [
    ("ETHUSDT", "target_atr"),
    ("BTCUSDT", "target_atr"),
    ("ETHUSDT", "target_atr_short"),
    ("BTCUSDT", "target_atr_short"),
]
```

`load_assets` and `simulator.run_portfolio_simulation` will auto-derive direction from target name — no other change needed.

- [ ] **Step 4: Verify scripts parse**

Run: `python -c "import ast; ast.parse(open('train_models.py').read()); ast.parse(open('run_backtest.py').read()); ast.parse(open('run_portfolio_backtest.py').read()); print('OK')"`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add train_models.py run_backtest.py run_portfolio_backtest.py
git commit -m "feat: wire target_atr_short into train/backtest/portfolio entry points"
```

---

## Task 7: Run the data pipeline + commit storage outputs

**Note:** This task runs real training (~30–60 min). Existing long models get retrained but produce identical bits (deterministic `random_state=42`). If short on time you may delete this step and run manually — but storage outputs are required for Tasks 8–9.

**Files:**
- Generated under `storage/features/`, `storage/models/`, `storage/backtest/`

- [ ] **Step 1: Rebuild features (adds `target_atr_short` column)**

Run: `python build_features.py`
Expected log lines confirming both symbols' parquet rebuilt.

Verify column added:

```bash
python -c "import pandas as pd; df = pd.read_parquet('storage/features/ETHUSDT_features.parquet'); assert 'target_atr_short' in df.columns; print('target_atr_short col present, non-null:', df['target_atr_short'].notna().sum())"
```

Expected: prints column count > 0.

- [ ] **Step 2: Train all models (long + short for both coins)**

Run: `python train_models.py`
Expected: 6 targets × 2 symbols = 12 model groups trained, log lines for each fold.

Verify new files:

```bash
ls storage/models/ETHUSDT_target_atr_short_fold*.pkl
ls storage/models/BTCUSDT_target_atr_short_fold*.pkl
```

Expected: 5 files each.

- [ ] **Step 3: Run single backtest (threshold scan for all targets)**

Run: `python run_backtest.py`

Verify new outputs:

```bash
ls storage/backtest/ETHUSDT_target_atr_short_threshold_scan.json
ls storage/backtest/BTCUSDT_target_atr_short_threshold_scan.json
```

- [ ] **Step 4: Run portfolio backtest**

Run: `python run_portfolio_backtest.py`

Verify new outputs:

```bash
ls storage/backtest/ETHUSDT_target_atr_short_portfolio_report.json
ls storage/backtest/BTCUSDT_target_atr_short_portfolio_report.json
```

Spot-check that the new reports contain `closed_trades` + `equity_log`:

```bash
python -c "import json; d = json.load(open('storage/backtest/ETHUSDT_target_atr_short_portfolio_report.json')); assert 'closed_trades' in d and 'equity_log' in d; print('OK trades=', len(d['closed_trades']), 'log=', len(d['equity_log']))"
```

- [ ] **Step 5: Commit new storage artifacts**

```bash
git add storage/features/ storage/models/ storage/backtest/
git commit -m "chore: regenerate features/models/reports with target_atr_short"
```

---

## Task 8: Create `combine_long_short.py` + tests

**Files:**
- Create: `combine_long_short.py`
- Test: `tests/test_combine_long_short.py`

- [ ] **Step 1: Write failing tests**

Create `tests/test_combine_long_short.py`:

```python
import json
import numpy as np
import pandas as pd
import pytest
from pathlib import Path

from combine_long_short import combine_one_symbol


def _write_report(path: Path, closed_trades: list, equity_log: list,
                  initial: float, final: float):
    payload = {
        'symbol': 'ETHUSDT',
        'target': 'target_atr',
        'initial_equity': initial,
        'final_equity':   final,
        'risk_pct': 0.02,
        'max_concurrent': 3,
        'optimal_threshold': 0.75,
        'total_signals': len(closed_trades),
        'executed_trades': len(closed_trades),
        'skipped_signals': 0,
        'metrics': {'sharpe_ratio': 0.0},
        'closed_trades': closed_trades,
        'equity_log': [list(p) for p in equity_log],
    }
    path.write_text(json.dumps(payload))


def test_combined_final_equity_equals_sum_of_legs(tmp_path):
    long_trades = [{'entry_idx': 0, 'exit_bar': 24, 'pnl_usd': 50_000,
                    'timestamp': '2024-01-01T00:00:00', 'equity_at_entry': 1_000_000}]
    short_trades = [{'entry_idx': 30, 'exit_bar': 54, 'pnl_usd': 30_000,
                     'timestamp': '2024-01-02T06:00:00', 'equity_at_entry': 1_000_000}]
    _write_report(tmp_path / 'ETHUSDT_target_atr_portfolio_report.json',
                  long_trades,  [(24, 1_050_000)], 1_000_000, 1_050_000)
    _write_report(tmp_path / 'ETHUSDT_target_atr_short_portfolio_report.json',
                  short_trades, [(54, 1_030_000)], 1_000_000, 1_030_000)

    result = combine_one_symbol('ETHUSDT', tmp_path)
    assert result['initial_combined_equity'] == 2_000_000
    assert result['final_combined_equity']   == 2_080_000
    assert result['long_final_equity']  == 1_050_000
    assert result['short_final_equity'] == 1_030_000
    assert result['n_long_trades']  == 1
    assert result['n_short_trades'] == 1


def test_correlation_high_positive(tmp_path):
    """Two identical equity trajectories → ρ ≈ 1."""
    ts = pd.date_range('2024-01-01', periods=10, freq='1D')
    long_trades  = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd': 1000.0 * (i+1),
                     'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                    for i in range(10)]
    short_trades = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd': 1000.0 * (i+1),
                     'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                    for i in range(10)]
    long_log  = [(i+1, 1_000_000 + sum(1000.0 * (j+1) for j in range(i+1))) for i in range(10)]
    short_log = list(long_log)  # identical
    _write_report(tmp_path / 'ETHUSDT_target_atr_portfolio_report.json',
                  long_trades, long_log, 1_000_000, long_log[-1][1])
    _write_report(tmp_path / 'ETHUSDT_target_atr_short_portfolio_report.json',
                  short_trades, short_log, 1_000_000, short_log[-1][1])

    result = combine_one_symbol('ETHUSDT', tmp_path)
    assert result['daily_return_correlation'] > 0.99


def test_correlation_high_negative(tmp_path):
    """Mirror-image equity trajectories → ρ ≈ -1."""
    ts = pd.date_range('2024-01-01', periods=10, freq='1D')
    long_trades  = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd':  1000.0,
                     'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                    for i in range(10)]
    short_trades = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd': -1000.0,
                     'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                    for i in range(10)]
    long_log  = [(i+1, 1_000_000 + 1000.0  * (i+1)) for i in range(10)]
    short_log = [(i+1, 1_000_000 - 1000.0  * (i+1)) for i in range(10)]
    _write_report(tmp_path / 'ETHUSDT_target_atr_portfolio_report.json',
                  long_trades, long_log, 1_000_000, long_log[-1][1])
    _write_report(tmp_path / 'ETHUSDT_target_atr_short_portfolio_report.json',
                  short_trades, short_log, 1_000_000, short_log[-1][1])

    result = combine_one_symbol('ETHUSDT', tmp_path)
    assert result['daily_return_correlation'] < -0.99
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_combine_long_short.py -v`
Expected: 3 FAIL with `ModuleNotFoundError: combine_long_short`

- [ ] **Step 3: Create `combine_long_short.py`**

Create new file at project root:

```python
"""Combine long + short portfolio reports into a single equity timeline.

Treats long and short as independent 1M sub-accounts. Final combined equity
is the sum of leg final_equity values. Daily return correlation is computed
between resampled-daily equity curves of the two legs.
"""
import json
import logging
import numpy as np
import pandas as pd
from pathlib import Path

import config

logger = logging.getLogger(__name__)


def _equity_log_to_daily_returns(equity_log: list,
                                  trade_timestamps: list) -> pd.Series:
    """Convert (exit_bar, equity) list into a daily return series.

    Uses each trade's entry timestamp as the index for its equity update —
    daily resampling smooths the entry-vs-exit granularity difference.
    Returns an empty series if input is empty.
    """
    if not equity_log or not trade_timestamps:
        return pd.Series([], dtype=float)
    series_index = pd.to_datetime(trade_timestamps, utc=True)
    series = pd.Series([eq for _, eq in equity_log], index=series_index)
    daily = series.resample('1D').last().ffill()
    return daily.pct_change().dropna()


def combine_one_symbol(symbol: str, backtest_dir: Path) -> dict:
    backtest_dir = Path(backtest_dir)
    long_path  = backtest_dir / f"{symbol}_target_atr_portfolio_report.json"
    short_path = backtest_dir / f"{symbol}_target_atr_short_portfolio_report.json"

    if not long_path.exists():
        raise FileNotFoundError(f"Missing long report: {long_path}")
    if not short_path.exists():
        raise FileNotFoundError(f"Missing short report: {short_path}")

    long_data  = json.loads(long_path.read_text())
    short_data = json.loads(short_path.read_text())

    long_initial  = long_data['initial_equity']
    short_initial = short_data['initial_equity']
    long_final    = long_data['final_equity']
    short_final   = short_data['final_equity']

    combined_initial = long_initial + short_initial
    combined_final   = long_final   + short_final
    total_return_pct = (combined_final / combined_initial - 1.0) * 100.0

    long_ts  = [t['timestamp'] for t in long_data['closed_trades']]
    short_ts = [t['timestamp'] for t in short_data['closed_trades']]

    long_daily  = _equity_log_to_daily_returns(long_data['equity_log'],  long_ts)
    short_daily = _equity_log_to_daily_returns(short_data['equity_log'], short_ts)

    # Align on common index for correlation
    aligned = pd.concat([long_daily.rename('long'), short_daily.rename('short')], axis=1).dropna()
    if len(aligned) >= 2:
        correlation = float(aligned['long'].corr(aligned['short']))
    else:
        correlation = None

    # Build combined equity timeline by merging both legs' (exit_timestamp, pnl)
    long_events  = [(pd.Timestamp(t['timestamp']), t['pnl_usd']) for t in long_data['closed_trades']]
    short_events = [(pd.Timestamp(t['timestamp']), t['pnl_usd']) for t in short_data['closed_trades']]
    all_events   = sorted(long_events + short_events, key=lambda x: x[0])

    equity = combined_initial
    peak   = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    daily_pnls = []
    for ts, pnl in all_events:
        equity += pnl
        peak = max(peak, equity)
        dd_pct = (equity - peak) / peak * 100.0 if peak > 0 else 0.0
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = equity - peak

    # Sharpe via combined daily returns
    combined_daily = (long_daily.fillna(0) + short_daily.fillna(0))
    if len(combined_daily) >= 2 and combined_daily.std() > 0:
        sharpe = float(combined_daily.mean() / combined_daily.std() * np.sqrt(365))
    else:
        sharpe = 0.0

    # CAGR
    total_days = (all_events[-1][0] - all_events[0][0]).days if len(all_events) >= 2 else 0
    total_years = total_days / 365.25 if total_days > 0 else 0.0
    if total_years > 0 and combined_final > 0:
        cagr = ((combined_final / combined_initial) ** (1.0 / total_years) - 1.0) * 100.0
    else:
        cagr = 0.0

    return {
        'symbol': symbol,
        'initial_combined_equity': combined_initial,
        'final_combined_equity':   combined_final,
        'long_final_equity':       long_final,
        'short_final_equity':      short_final,
        'combined_metrics': {
            'total_return_pct': round(total_return_pct, 4),
            'cagr_pct':         round(cagr, 4),
            'sharpe_ratio':     round(sharpe, 4),
            'max_drawdown_pct': round(max_dd_pct, 4),
            'max_drawdown_usd': round(max_dd_usd, 2),
        },
        'daily_return_correlation': round(correlation, 4) if correlation is not None else None,
        'n_long_trades':  len(long_data['closed_trades']),
        'n_short_trades': len(short_data['closed_trades']),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    backtest_dir = config.STORAGE_BACKTEST
    for symbol in ("ETHUSDT", "BTCUSDT"):
        try:
            result = combine_one_symbol(symbol, backtest_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue
        out_path = backtest_dir / f"{symbol}_long_short_combined.json"
        out_path.write_text(json.dumps(result, indent=2))
        logger.info(
            f"[{symbol}] combined Sharpe={result['combined_metrics']['sharpe_ratio']} "
            f"MDD%={result['combined_metrics']['max_drawdown_pct']} "
            f"ρ={result['daily_return_correlation']} → {out_path.name}"
        )


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run combine tests to verify pass**

Run: `pytest tests/test_combine_long_short.py -v`
Expected: 3 PASS

- [ ] **Step 5: Run combine on real data**

Run: `python combine_long_short.py`
Expected: log lines for ETH + BTC with Sharpe / MDD / ρ values. Files written to `storage/backtest/`.

Verify:
```bash
ls storage/backtest/ETHUSDT_long_short_combined.json storage/backtest/BTCUSDT_long_short_combined.json
```

- [ ] **Step 6: Commit**

```bash
git add combine_long_short.py tests/test_combine_long_short.py storage/backtest/
git commit -m "feat: combine_long_short — independent sub-account equity merge + daily-return correlation"
```

---

## Task 9: Create `compare_long_short.py`

**Files:**
- Create: `compare_long_short.py`

- [ ] **Step 1: Write the script**

Create `compare_long_short.py`:

```python
"""Side-by-side terminal table comparing long vs short threshold-scan + combined metrics.

Reads:
  storage/backtest/{SYMBOL}_target_atr_threshold_scan.json
  storage/backtest/{SYMBOL}_target_atr_short_threshold_scan.json
  storage/backtest/{SYMBOL}_long_short_combined.json

Prints a single table. No file output.
"""
import json
import sys
from pathlib import Path

import config

SYMBOLS = ("ETHUSDT", "BTCUSDT")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(val, suffix: str = "", precision: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{precision}f}{suffix}"
    return f"{val}{suffix}"


def main() -> int:
    backtest_dir = config.STORAGE_BACKTEST

    rows = []
    for symbol in SYMBOLS:
        long_scan  = _load(backtest_dir / f"{symbol}_target_atr_threshold_scan.json")
        short_scan = _load(backtest_dir / f"{symbol}_target_atr_short_threshold_scan.json")
        combined   = _load(backtest_dir / f"{symbol}_long_short_combined.json")

        long_opt  = (long_scan  or {}).get('optimal_metrics') or {}
        short_opt = (short_scan or {}).get('optimal_metrics') or {}
        comb_m    = (combined   or {}).get('combined_metrics') or {}

        rows.append({
            'symbol': symbol,
            'long_sharpe':   long_opt.get('sharpe_ratio'),
            'long_winrate':  long_opt.get('win_rate'),
            'long_n':        long_opt.get('n_trades'),
            'short_sharpe':  short_opt.get('sharpe_ratio'),
            'short_winrate': short_opt.get('win_rate'),
            'short_n':       short_opt.get('n_trades'),
            'combined_sharpe': comb_m.get('sharpe_ratio'),
            'combined_mdd':    comb_m.get('max_drawdown_pct'),
            'correlation':     (combined or {}).get('daily_return_correlation'),
        })

    if not rows:
        print("No reports found.")
        return 1

    label_w = 20
    col_w   = 14
    headers = ['Metric'] + [r['symbol'] for r in rows]
    print()
    print(headers[0].ljust(label_w) + ''.join(h.ljust(col_w) for h in headers[1:]))
    print('-' * (label_w + col_w * len(rows)))
    metric_rows = [
        ('Long  Sharpe',     'long_sharpe',     '',  2),
        ('Long  Win%',       'long_winrate',    '',  4),
        ('Long  Trades',     'long_n',          '',  0),
        ('Short Sharpe',     'short_sharpe',    '',  2),
        ('Short Win%',       'short_winrate',   '',  4),
        ('Short Trades',     'short_n',         '',  0),
        ('Combined Sharpe',  'combined_sharpe', '',  2),
        ('Combined MDD%',    'combined_mdd',    '%', 2),
        ('Long-Short ρ',     'correlation',     '',  3),
    ]
    for label, key, suffix, prec in metric_rows:
        line = label.ljust(label_w)
        for r in rows:
            line += _fmt(r[key], suffix, prec).ljust(col_w)
        print(line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 2: Run the script against real data**

Run: `python compare_long_short.py`
Expected: table prints with values from all loaded reports.

- [ ] **Step 3: Commit**

```bash
git add compare_long_short.py
git commit -m "feat: compare_long_short terminal table"
```

---

## Task 10: End-to-end verification + final commit

- [ ] **Step 1: Run full test suite**

Run: `pytest tests/ -v`
Expected: all green. Existing 144 tests plus new ones from Tasks 1–8.

- [ ] **Step 2: Verify all expected storage outputs exist**

Run:
```bash
ls storage/features/ETHUSDT_features.parquet storage/features/BTCUSDT_features.parquet \
   storage/models/ETHUSDT_target_atr_short_final.pkl storage/models/BTCUSDT_target_atr_short_final.pkl \
   storage/backtest/ETHUSDT_target_atr_short_threshold_scan.json \
   storage/backtest/BTCUSDT_target_atr_short_threshold_scan.json \
   storage/backtest/ETHUSDT_target_atr_short_portfolio_report.json \
   storage/backtest/BTCUSDT_target_atr_short_portfolio_report.json \
   storage/backtest/ETHUSDT_long_short_combined.json \
   storage/backtest/BTCUSDT_long_short_combined.json
```

Expected: all paths exist.

- [ ] **Step 3: Verify `live/` files unchanged**

Run: `git log --oneline live/`
Expected: no new commits from this branch on live/ files (only the unrelated fetcher rate-limit fix from earlier).

- [ ] **Step 4: View the side-by-side comparison table**

Run: `python compare_long_short.py`
Capture the output — that's the research deliverable.

- [ ] **Step 5: Document research result in README (optional)**

If the user wants, append a "Short prediction research" section to `README.md` summarizing the Sharpe/MDD/ρ table.

- [ ] **Step 6: Final push (manual)**

The user decides whether to `git push origin master`. This plan does not auto-push.

---

## Notes / Out of Scope (reminder)

- `live/` files unchanged. `config.LIVE_TARGET` stays `"target_atr"` (long).
- Dashboard not modified.
- No Discord integration for short signals.
- `target_fixed_short` not implemented.
- Shared-account multi-direction DRC not implemented (deferred until standalone short results justify it).
