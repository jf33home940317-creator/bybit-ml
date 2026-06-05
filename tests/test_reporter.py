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
