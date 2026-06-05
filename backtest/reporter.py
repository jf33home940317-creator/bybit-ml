import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, (pd.Timestamp, np.datetime64)):
            return pd.Timestamp(obj).isoformat()
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


def save_portfolio_equity_curve(
    results: dict,
    symbol: str,
    target: str,
    output_dir: Path,
) -> None:
    """雙子圖 PNG：上圖 USD 資金曲線（含 MDD 陰影），下圖 Drawdown % 瀑布圖。"""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    closed     = results.get('closed_trades', [])
    equity_log = results.get('equity_log', [])
    if not equity_log or not closed:
        return

    metrics        = results.get('metrics', {})
    initial_equity = results['initial_equity']

    # equity_log 與 closed_trades 一一對應（同順序，長度相等）
    exit_timestamps = [
        pd.Timestamp(t['timestamp']) + pd.Timedelta(hours=int(t['exit_bar'] - t['entry_idx']))
        for t in closed
    ]

    # 插入起始點（第一筆出場時間 - 1h）
    t0 = exit_timestamps[0] - pd.Timedelta(hours=1)
    timestamps    = [t0] + exit_timestamps
    equities_arr  = np.array([initial_equity] + [e for _, e in equity_log])

    # Drawdown 計算
    running_peak = np.maximum.accumulate(equities_arr)
    dd_pct = np.where(running_peak > 0,
                      (equities_arr - running_peak) / running_peak * 100,
                      0.0)

    mdd_pct  = metrics.get('max_drawdown_pct', 0.0)
    sharpe   = metrics.get('sharpe_ratio', 0.0)
    final_eq = results['final_equity']

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(14, 8),
        gridspec_kw={'height_ratios': [7, 3]},
        sharex=True,
    )

    # ── 上圖：USD 資金曲線 ──────────────────────────────────────────
    ax_top.step(timestamps, equities_arr, where='post', color='steelblue', linewidth=1.5)
    mdd_mask = equities_arr < running_peak
    ax_top.fill_between(timestamps, equities_arr, running_peak,
                        where=mdd_mask, alpha=0.25, color='red', step='post')
    ax_top.axhline(initial_equity, color='gray', linestyle='--', linewidth=1)
    ax_top.set_ylabel('Portfolio Equity (USD)')
    ax_top.set_title(
        f"{symbol} {target} Portfolio  |  "
        f"Start: ${initial_equity:,.0f}  →  End: ${final_eq:,.0f}  |  "
        f"MDD: {mdd_pct:.1f}%  |  Sharpe: {sharpe:.2f}"
    )
    ax_top.yaxis.set_major_formatter(lambda x, _: f'${x:,.0f}')

    # ── 下圖：Drawdown % 瀑布圖 ─────────────────────────────────────
    ax_bot.fill_between(timestamps, dd_pct, 0,
                        where=(dd_pct <= 0), alpha=0.6, color='red', step='post')
    ax_bot.step(timestamps, dd_pct, where='post', color='darkred', linewidth=1)
    ax_bot.axhline(0, color='gray', linestyle='--', linewidth=0.8)
    ax_bot.set_ylabel('Drawdown (%)')
    ax_bot.set_xlabel('Exit Timestamp')

    plt.xticks(rotation=30)
    plt.tight_layout()

    path = output_dir / f"{symbol}_{target}_portfolio_equity.png"
    plt.savefig(path, dpi=150)
    plt.close()
