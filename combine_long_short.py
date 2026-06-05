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
    aligned = pd.concat(
        [long_daily.rename('long'), short_daily.rename('short')],
        axis=1, sort=True,
    ).dropna()
    if len(aligned) >= 2:
        correlation = float(aligned['long'].corr(aligned['short']))
    else:
        correlation = None

    # Build combined equity timeline by merging both legs' (timestamp, pnl)
    long_events  = [(pd.Timestamp(t['timestamp']), t['pnl_usd']) for t in long_data['closed_trades']]
    short_events = [(pd.Timestamp(t['timestamp']), t['pnl_usd']) for t in short_data['closed_trades']]
    all_events   = sorted(long_events + short_events, key=lambda x: x[0])

    equity = combined_initial
    peak   = equity
    max_dd_pct = 0.0
    max_dd_usd = 0.0
    equity_timeline = []  # (timestamp, equity_after_event) — used for Sharpe too
    for ts, pnl in all_events:
        equity += pnl
        equity_timeline.append((ts, equity))
        peak = max(peak, equity)
        dd_pct = (equity - peak) / peak * 100.0 if peak > 0 else 0.0
        if dd_pct < max_dd_pct:
            max_dd_pct = dd_pct
            max_dd_usd = equity - peak

    # Combined Sharpe — compute on the COMBINED USD equity timeline, not by
    # summing two pct_change series (each leg's pct_change is normalized to
    # its own 1M base, so summing double-weights single-leg days).
    if equity_timeline:
        eq_idx = pd.to_datetime([t for t, _ in equity_timeline], utc=True)
        eq_series = pd.Series([e for _, e in equity_timeline], index=eq_idx)
        combined_daily_eq = eq_series.resample('1D').last().ffill()
        combined_daily_returns = combined_daily_eq.pct_change().dropna()
        if len(combined_daily_returns) >= 2 and combined_daily_returns.std() > 0:
            sharpe = float(combined_daily_returns.mean() / combined_daily_returns.std() * np.sqrt(365))
        else:
            sharpe = 0.0
    else:
        sharpe = 0.0

    # CAGR
    if all_events:
        total_days = (all_events[-1][0] - all_events[0][0]).days
    else:
        total_days = 0
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
            f"rho={result['daily_return_correlation']} -> {out_path.name}"
        )


if __name__ == "__main__":
    main()
