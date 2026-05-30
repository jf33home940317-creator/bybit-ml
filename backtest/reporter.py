import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return super().default(obj)


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
        'symbol':            symbol,
        'target':            target,
        'fee_pct':           0.002,
        'horizon':           24,
        'optimal_threshold': results['optimal_threshold'],
        'optimal_metrics':   results['optimal_metrics'],
        'threshold_scan':    results['threshold_scan'],
    }

    path = output_dir / f"{symbol}_{target}_threshold_scan.json"
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(out, f, indent=2, cls=_NumpyEncoder)


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
