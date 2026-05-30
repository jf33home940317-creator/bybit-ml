import numpy as np
import pandas as pd
import pytest
from backtest.engine import generate_oof_probabilities
from backtest.engine import compute_trade_pnl
from backtest.engine import run_threshold_scan
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

    def test_pnl_sl_wins_on_tie(self):
        """When TP and SL both trigger on the same bar, SL wins."""
        df = _make_df(100, close=1000.0)
        # Bar 1: high=1025 (≥ TP 1020) AND low=985 (≤ SL 990) — simultaneous hit
        df.loc[1, 'high'] = 1025.0
        df.loc[1, 'low']  = 985.0

        trades = compute_trade_pnl(df, signal_indices=[0], target='target_fixed')

        assert len(trades) == 1
        row = trades.iloc[0]
        assert row['outcome'] == 'sl', "SL must win when both TP and SL trigger on the same bar"
        assert abs(row['pnl'] - (-0.01 - 0.002)) < 1e-9
        assert row['holding_bars'] == 1


class TestRunThresholdScan:

    def test_min_trades_filter(self):
        """Thresholds yielding fewer than min_trades trades must not appear in results."""
        n = 600
        df = _make_df(n, close=1000.0)
        feature_cols = []
        # Model returns exactly 0.60 for all OOF rows.
        # threshold=0.50 → all OOF rows pass (many trades).
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
