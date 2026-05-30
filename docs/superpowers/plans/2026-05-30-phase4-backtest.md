# Phase 4.1 Vectorized Signal Backtest Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an OOF vectorized backtest engine that scans signal thresholds (0.50–0.80) across all four BTCUSDT/ETHUSDT × target_fixed/target_atr combinations and outputs JSON reports + PNG charts.

**Architecture:** `backtest/engine.py` generates OOF probabilities from saved fold models, then evaluates P&L for each threshold via semi-vectorized NumPy operations. `backtest/reporter.py` serialises results to JSON and PNG. `run_backtest.py` is the entry point that iterates symbols × targets.

**Tech Stack:** numpy, pandas, matplotlib, joblib, xgboost (all already installed)

---

## File Map

| Action | Path | Responsibility |
|--------|------|---------------|
| Modify | `config.py` | Add `STORAGE_BACKTEST` constant |
| Create | `backtest/__init__.py` | Package marker |
| Create | `backtest/engine.py` | OOF proba generation, P&L calc, threshold scan |
| Create | `backtest/reporter.py` | JSON report + 2 PNG charts per combination |
| Create | `run_backtest.py` | Entry point: loops symbols × targets |
| Create | `tests/test_backtest_engine.py` | 6 unit tests (OOF leak, val coverage, TP/SL/timeout P&L, min_trades filter) |

---

## Task 1: Config + Module Scaffold

**Files:**
- Modify: `config.py`
- Create: `backtest/__init__.py`

- [ ] **Step 1: Add STORAGE_BACKTEST to config.py**

In `config.py`, add after the `STORAGE_MODELS` line:

```python
STORAGE_BACKTEST = BASE_DIR / "storage" / "backtest"
```

- [ ] **Step 2: Create backtest/__init__.py**

```python
```
(empty file — package marker only)

- [ ] **Step 3: Verify import works**

Run:
```
python -c "import config; print(config.STORAGE_BACKTEST)"
```
Expected output: `E:\93050207\python\BYBIT_ML\storage\backtest` (or equivalent)

- [ ] **Step 4: Commit**

```bash
git add config.py backtest/__init__.py
git commit -m "feat: add STORAGE_BACKTEST config and backtest package scaffold"
```

---

## Task 2: generate_oof_probabilities (TDD)

**Files:**
- Create: `tests/test_backtest_engine.py`
- Create: `backtest/engine.py` (partial — this function only)

- [ ] **Step 1: Write failing tests**

Create `tests/test_backtest_engine.py`:

```python
import numpy as np
import pandas as pd
import pytest
from backtest.engine import generate_oof_probabilities
from models.splitter import purged_walk_forward_split


class _ConstantModel:
    """Mock model that always returns a fixed probability for all inputs."""
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _make_df(n: int = 200, close: float = 1000.0) -> pd.DataFrame:
    """Minimal DataFrame with all columns required by the backtest engine."""
    return pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01', periods=n, freq='1h'),
        'open':   close,
        'high':   close * 1.005,
        'low':    close * 0.995,
        'close':  close,
        'volume': 1.0,
        'atr_14': close * 0.01,
    })


class TestGenerateOofProbabilities:

    def test_oof_no_future_leak(self):
        """Training-period rows (never in any val set) must be NaN."""
        n = 600
        df = _make_df(n)
        feature_cols = []
        fold_models = [_ConstantModel(0.7)] * 5

        proba = generate_oof_probabilities(df, feature_cols, fold_models)

        # With n=600: fold_size=100, first val starts at index 124.
        # Indices [0, 124) were NEVER in a validation set.
        assert proba.iloc[:124].isna().all(), "Training-period rows must be NaN"

    def test_oof_val_coverage(self):
        """Number of non-NaN values must equal the sum of all val-set sizes."""
        n = 600
        df = _make_df(n)
        feature_cols = []
        fold_models = [_ConstantModel(0.7)] * 5

        proba = generate_oof_probabilities(df, feature_cols, fold_models)

        expected = sum(len(v) for _, v in purged_walk_forward_split(n))
        assert proba.notna().sum() == expected  # 5 × 76 = 380 for n=600
```

- [ ] **Step 2: Run tests to confirm they fail**

```
pytest tests/test_backtest_engine.py -v
```
Expected: `ImportError` — `backtest.engine` does not exist yet.

- [ ] **Step 3: Implement generate_oof_probabilities in backtest/engine.py**

Create `backtest/engine.py`:

```python
import numpy as np
import pandas as pd
from models.splitter import purged_walk_forward_split


def generate_oof_probabilities(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
) -> pd.Series:
    """Generate out-of-fold probabilities; training rows remain NaN."""
    proba = pd.Series(np.nan, index=df.index, dtype=float)
    for (_, val_idx), model in zip(purged_walk_forward_split(len(df)), fold_models):
        X_val = df.iloc[val_idx][feature_cols]
        proba.iloc[val_idx] = model.predict_proba(X_val)[:, 1]
    return proba
```

- [ ] **Step 4: Run tests to confirm they pass**

```
pytest tests/test_backtest_engine.py::TestGenerateOofProbabilities -v
```
Expected: 2 PASSED

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: implement generate_oof_probabilities with no-leak guarantee"
```

---

## Task 3: compute_trade_pnl (TDD)

**Files:**
- Modify: `tests/test_backtest_engine.py` (add P&L tests)
- Modify: `backtest/engine.py` (add compute_trade_pnl)

- [ ] **Step 1: Add P&L tests to tests/test_backtest_engine.py**

Append the following class after `TestGenerateOofProbabilities`:

```python
from backtest.engine import compute_trade_pnl


class TestComputeTradePnl:

    def test_pnl_tp_hit(self):
        """TP hit before SL: pnl = +2% − fee."""
        df = _make_df(100, close=1000.0)
        # Bar 1: high=1025 (≥ TP 1020), low=995 (> SL 990) → TP first
        df.loc[1, 'high'] = 1025.0
        df.loc[1, 'low']  = 995.0

        trades = compute_trade_pnl(df, signal_indices=[0], target='target_fixed')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'tp'
        assert abs(row['pnl'] - (0.02 - 0.002)) < 1e-9
        assert row['holding_bars'] == 1

    def test_pnl_sl_hit(self):
        """SL hit before TP: pnl = −1% − fee."""
        df = _make_df(100, close=1000.0)
        # Bar 1: high=1010 (< TP 1020), low=985 (≤ SL 990) → SL first
        df.loc[1, 'high'] = 1010.0
        df.loc[1, 'low']  = 985.0

        trades = compute_trade_pnl(df, signal_indices=[0], target='target_fixed')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'sl'
        assert abs(row['pnl'] - (-0.01 - 0.002)) < 1e-9
        assert row['holding_bars'] == 1

    def test_pnl_timeout(self):
        """Neither TP nor SL hit in 24 bars: exit at close[t+24]."""
        df = _make_df(100, close=1000.0)
        # All highs < TP (1020), all lows > SL (990)
        df['high'] = 1005.0
        df['low']  = 995.0
        # close[24] = 1010 → timeout pnl = +1% − fee
        df.loc[24, 'close'] = 1010.0

        trades = compute_trade_pnl(df, signal_indices=[0], target='target_fixed')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'timeout'
        assert abs(row['pnl'] - (0.01 - 0.002)) < 1e-9
        assert row['holding_bars'] == 24
```

- [ ] **Step 2: Run new tests to confirm they fail**

```
pytest tests/test_backtest_engine.py::TestComputeTradePnl -v
```
Expected: `ImportError` — `compute_trade_pnl` not yet defined.

- [ ] **Step 3: Implement compute_trade_pnl in backtest/engine.py**

Append to `backtest/engine.py` (after `generate_oof_probabilities`):

```python
def compute_trade_pnl(
    df: pd.DataFrame,
    signal_indices: list,
    target: str,
    fee: float = 0.002,
) -> pd.DataFrame:
    """Semi-vectorized P&L: list-comprehension builds 2D matrix, NumPy broadcasts.

    Signals within 25 bars of the end of df are skipped (no complete horizon).
    SL wins on ties (same bar as TP).
    """
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

    # Build 2D future price matrices: shape (n_signals, HORIZON)
    future_highs = np.array([high_vals[i + 1 : i + 1 + HORIZON] for i in signal_indices])
    future_lows  = np.array([low_vals[ i + 1 : i + 1 + HORIZON] for i in signal_indices])

    if target == 'target_fixed':
        tp_prices = entry_prices * 1.02
        sl_prices = entry_prices * 0.99
    else:  # target_atr
        tp_prices = entry_prices + 3.0 * atr_vals[sig_arr]
        sl_prices = entry_prices - 1.5 * atr_vals[sig_arr]

    tp_pct = (tp_prices - entry_prices) / entry_prices   # positive
    sl_pct = (sl_prices - entry_prices) / entry_prices   # negative

    # Boolean hit matrices (broadcast TP/SL price per row)
    tp_hit = future_highs >= tp_prices[:, None]   # (n, HORIZON)
    sl_hit = future_lows  <= sl_prices[:, None]   # (n, HORIZON)

    # First bar hit; HORIZON sentinel = never hit
    tp_first = np.where(tp_hit.any(axis=1), tp_hit.argmax(axis=1), HORIZON)
    sl_first = np.where(sl_hit.any(axis=1), sl_hit.argmax(axis=1), HORIZON)

    # SL wins on tie
    tp_wins = tp_first < sl_first
    sl_wins = (~tp_wins) & sl_hit.any(axis=1)

    timeout_exit = close_vals[sig_arr + HORIZON]
    timeout_pct  = (timeout_exit - entry_prices) / entry_prices

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

- [ ] **Step 4: Run all engine tests so far**

```
pytest tests/test_backtest_engine.py -v
```
Expected: 5 PASSED (2 OOF + 3 P&L)

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: implement compute_trade_pnl with semi-vectorized P&L"
```

---

## Task 4: run_threshold_scan (TDD)

**Files:**
- Modify: `tests/test_backtest_engine.py` (add min_trades test)
- Modify: `backtest/engine.py` (add run_threshold_scan)

- [ ] **Step 1: Add min_trades test to tests/test_backtest_engine.py**

Add the following import at the top of the file:
```python
from backtest.engine import run_threshold_scan
```

Append the following class after `TestComputeTradePnl`:

```python
class TestRunThresholdScan:

    def test_min_trades_filter(self):
        """Thresholds yielding fewer than min_trades trades must not appear in results."""
        n = 600
        df = _make_df(n, close=1000.0)
        feature_cols = []
        # Model returns exactly 0.60 for all OOF rows.
        # threshold=0.50 → all 380 OOF rows pass (many trades).
        # threshold=0.65 → 0 rows pass (proba never reaches 0.65).
        fold_models = [_ConstantModel(0.60)] * 5

        results = run_threshold_scan(
            df, feature_cols, fold_models,
            target='target_fixed',
            thresholds=np.array([0.50, 0.65]),
            min_trades=50,
        )

        scan_thresholds = [entry['threshold'] for entry in results['threshold_scan']]
        assert 0.65 not in scan_thresholds, "threshold with 0 trades must be filtered"
        assert results['optimal_threshold'] == 0.50
```

- [ ] **Step 2: Run new test to confirm it fails**

```
pytest tests/test_backtest_engine.py::TestRunThresholdScan -v
```
Expected: `ImportError` — `run_threshold_scan` not yet defined.

- [ ] **Step 3: Implement run_threshold_scan in backtest/engine.py**

Append to `backtest/engine.py`:

```python
def run_threshold_scan(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
    target: str,
    thresholds: np.ndarray = None,
    fee: float = 0.002,
    min_trades: int = 20,
) -> dict:
    """Scan signal thresholds and return metrics + optimal threshold by Sharpe.

    Returns dict with keys:
        threshold_scan      - list of metric dicts, one per valid threshold
        optimal_threshold   - threshold with highest Sharpe (>= min_trades)
        optimal_metrics     - metrics dict for optimal_threshold
        optimal_trades_df   - trades DataFrame for optimal_threshold (for equity curve)
        total_years         - float, used in Sharpe normalisation
    """
    if thresholds is None:
        thresholds = np.round(np.arange(0.50, 0.81, 0.01), 2)

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

        trades_df = compute_trade_pnl(df, signal_indices, target, fee)
        if len(trades_df) < min_trades:
            continue

        pnl_vals = trades_df['pnl'].values
        n_trades = len(pnl_vals)
        mean_pnl = float(pnl_vals.mean())
        std_pnl  = float(pnl_vals.std())
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
            best_sharpe      = sharpe
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

- [ ] **Step 4: Run all engine tests**

```
pytest tests/test_backtest_engine.py -v
```
Expected: 6 PASSED

- [ ] **Step 5: Commit**

```bash
git add backtest/engine.py tests/test_backtest_engine.py
git commit -m "feat: implement run_threshold_scan with min_trades filter and Sharpe optimisation"
```

---

## Task 5: backtest/reporter.py

**Files:**
- Create: `backtest/reporter.py`

No unit tests — outputs are visual/file artefacts verified by running run_backtest.py in Task 6.

- [ ] **Step 1: Create backtest/reporter.py**

```python
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def save_threshold_scan(
    results: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """Write threshold_scan.json for the given symbol/target combination."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out = {
        'symbol':             symbol,
        'target':             target,
        'fee_pct':            0.002,
        'horizon':            24,
        'optimal_threshold':  results['optimal_threshold'],
        'optimal_metrics':    results['optimal_metrics'],
        'threshold_scan':     results['threshold_scan'],
    }

    path = output_dir / f"{symbol}_{target}_threshold_scan.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2)


def save_threshold_tradeoff_chart(
    results: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """Dual-Y-axis chart: win_rate (left, blue) and sharpe_ratio (right, orange)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    scan = results['threshold_scan']
    if not scan:
        return

    thresholds = [e['threshold']    for e in scan]
    win_rates  = [e['win_rate']     for e in scan]
    sharpes    = [e['sharpe_ratio'] for e in scan]

    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax2 = ax1.twinx()

    ax1.plot(thresholds, win_rates, 'b-o', label='Win Rate',     linewidth=2, markersize=4)
    ax2.plot(thresholds, sharpes,   color='orange', marker='s',
             label='Sharpe Ratio', linewidth=2, markersize=4)

    opt = results['optimal_threshold']
    if opt is not None:
        ax1.axvline(opt, color='red', linestyle='--', linewidth=1.5,
                    label=f'Optimal: {opt}')

    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Win Rate', color='b')
    ax2.set_ylabel('Sharpe Ratio', color='orange')
    ax1.set_title(f"{symbol} {target} — Threshold Tradeoff")
    ax1.tick_params(axis='y', labelcolor='b')
    ax2.tick_params(axis='y', labelcolor='orange')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

    plt.tight_layout()
    path = output_dir / f"{symbol}_{target}_threshold_tradeoff.png"
    plt.savefig(path, dpi=150)
    plt.close()


def save_equity_curve(
    results: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """Cumulative return curve with entry timestamps on X-axis."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    trades_df = results.get('optimal_trades_df')
    if trades_df is None or len(trades_df) == 0:
        return

    metrics = results['optimal_metrics']
    thr     = results['optimal_threshold']

    timestamps     = pd.to_datetime(trades_df['timestamp'])
    cumulative_pct = trades_df['pnl'].cumsum() * 100  # convert to %

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(timestamps, cumulative_pct, 'b-', linewidth=1.5)
    ax.axhline(0, color='gray', linestyle='--', linewidth=1)
    ax.fill_between(timestamps, cumulative_pct, 0,
                    where=(cumulative_pct >= 0), alpha=0.25, color='green')
    ax.fill_between(timestamps, cumulative_pct, 0,
                    where=(cumulative_pct < 0),  alpha=0.25, color='red')

    n      = metrics['n_trades']
    sharpe = metrics['sharpe_ratio']
    ax.set_title(f"{symbol} {target} — Equity Curve  (thr={thr}, n={n}, Sharpe={sharpe:.2f})")
    ax.set_xlabel('Entry Timestamp')
    ax.set_ylabel('Cumulative Return (%)')
    plt.xticks(rotation=30)
    plt.tight_layout()

    path = output_dir / f"{symbol}_{target}_optimal_equity.png"
    plt.savefig(path, dpi=150)
    plt.close()
```

- [ ] **Step 2: Verify import**

```
python -c "from backtest import reporter; print('OK')"
```
Expected: `OK`

- [ ] **Step 3: Commit**

```bash
git add backtest/reporter.py
git commit -m "feat: implement backtest reporter (JSON + threshold tradeoff + equity curve)"
```

---

## Task 6: run_backtest.py Entry Point + End-to-End Smoke Test

**Files:**
- Create: `run_backtest.py`

- [ ] **Step 1: Create run_backtest.py**

```python
import json
import logging
import joblib
import pandas as pd
from pathlib import Path

import config
from backtest import engine, reporter

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def main() -> None:
    backtest_dir = config.STORAGE_BACKTEST
    backtest_dir.mkdir(parents=True, exist_ok=True)

    for symbol in config.SYMBOLS:
        feature_path = config.STORAGE_FEATURES / f"{symbol}_features.parquet"
        report_path  = config.STORAGE_FEATURES / f"{symbol}_validation_report.json"

        df = pd.read_parquet(feature_path)
        feature_cols = json.load(open(report_path))["metadata"]["feature_columns"]

        for target in ["target_fixed", "target_atr"]:
            logger.info(f"[{symbol}][{target}] Loading fold models…")
            fold_models = [
                joblib.load(config.STORAGE_MODELS / f"{symbol}_{target}_fold{k}.pkl")
                for k in range(1, 6)
            ]

            logger.info(f"[{symbol}][{target}] Running threshold scan…")
            results = engine.run_threshold_scan(df, feature_cols, fold_models, target)
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


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run full backtest**

```
python run_backtest.py
```
Expected log output (4 combinations, ≈2–5 min total):
```
... [BTCUSDT][target_fixed] optimal_threshold=0.XX, n_trades=XXX, sharpe=X.XX
... [BTCUSDT][target_atr]   optimal_threshold=0.XX ...
... [ETHUSDT][target_fixed] optimal_threshold=0.XX ...
... [ETHUSDT][target_atr]   optimal_threshold=0.XX ...
```

- [ ] **Step 3: Verify output files exist**

```
python -c "
import os
from pathlib import Path
d = Path('storage/backtest')
files = sorted(os.listdir(d))
for f in files: print(f)
"
```
Expected: 12 files — 3 per combination (JSON + 2 PNG) × 4 combinations:
```
BTCUSDT_target_atr_optimal_equity.png
BTCUSDT_target_atr_threshold_scan.json
BTCUSDT_target_atr_threshold_tradeoff.png
BTCUSDT_target_fixed_optimal_equity.png
BTCUSDT_target_fixed_threshold_scan.json
BTCUSDT_target_fixed_threshold_tradeoff.png
ETHUSDT_target_atr_optimal_equity.png
ETHUSDT_target_atr_threshold_scan.json
ETHUSDT_target_atr_threshold_tradeoff.png
ETHUSDT_target_fixed_optimal_equity.png
ETHUSDT_target_fixed_threshold_scan.json
ETHUSDT_target_fixed_threshold_tradeoff.png
```

- [ ] **Step 4: Run full test suite to verify no regressions**

```
pytest tests/ -v --tb=short
```
Expected: all existing tests + 6 new backtest tests pass.

- [ ] **Step 5: Commit**

```bash
git add run_backtest.py
git commit -m "feat: add run_backtest.py entry point and complete Phase 4.1 implementation"
```
