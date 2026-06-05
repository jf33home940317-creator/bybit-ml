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
    """Two identical equity trajectories => rho ~ 1."""
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
    """Mirror-image daily returns => rho ~ -1.

    Equity oscillates: long goes +1k,-0.5k,+1k,-0.5k... while short does the
    mirror (-1k,+0.5k,-1k,+0.5k...). This gives the two legs anti-correlated
    daily pct_change values.
    """
    ts = pd.date_range('2024-01-01', periods=10, freq='1D')
    pnl_pattern_long  = [1000.0 if i % 2 == 0 else -500.0 for i in range(10)]
    pnl_pattern_short = [-x for x in pnl_pattern_long]

    long_trades = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd': pnl_pattern_long[i],
                    'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                   for i in range(10)]
    short_trades = [{'entry_idx': i, 'exit_bar': i+1, 'pnl_usd': pnl_pattern_short[i],
                     'timestamp': ts[i].isoformat(), 'equity_at_entry': 1_000_000}
                    for i in range(10)]

    long_equity = 1_000_000
    long_log = []
    for i, p in enumerate(pnl_pattern_long):
        long_equity += p
        long_log.append((i + 1, long_equity))
    short_equity = 1_000_000
    short_log = []
    for i, p in enumerate(pnl_pattern_short):
        short_equity += p
        short_log.append((i + 1, short_equity))

    _write_report(tmp_path / 'ETHUSDT_target_atr_portfolio_report.json',
                  long_trades, long_log, 1_000_000, long_log[-1][1])
    _write_report(tmp_path / 'ETHUSDT_target_atr_short_portfolio_report.json',
                  short_trades, short_log, 1_000_000, short_log[-1][1])

    result = combine_one_symbol('ETHUSDT', tmp_path)
    assert result['daily_return_correlation'] < -0.99
