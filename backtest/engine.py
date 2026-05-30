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
