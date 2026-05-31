import numpy as np
import pandas as pd
from backtest.engine import generate_oof_probabilities, compute_trade_pnl


def run_portfolio_simulation(
    df: pd.DataFrame,
    feature_cols: list,
    fold_models: list,
    target: str,
    optimal_threshold: float,
    initial_equity: float = 1_000_000,
    risk_pct: float = 0.02,
    max_concurrent: int = 3,
) -> dict:
    """
    DRC 風控投資組合模擬。

    Phase 1：重用 Phase 4.1 函數預計算原始交易清單（忽略併發）。
    Phase 2：事件驅動順序迴圈疊加 Fixed Risk Sizing、Concurrency Control、複利。

    Returns dict with keys:
        closed_trades    list[dict]          所有已平倉交易（含 trade_roe）
        equity_log       list[(bar, float)]  步階資金曲線資料點
        final_equity     float               最終帳戶淨值
        total_signals    int                 OOF 訊號總數
        executed_trades  int                 實際成交筆數
        skipped_signals  int                 因容量上限跳過的訊號數
        metrics          dict                績效指標
    """
    # ── Phase 1：預計算 ──────────────────────────────────────────────
    proba = generate_oof_probabilities(df, feature_cols, fold_models)
    signal_indices = np.where(proba >= optimal_threshold)[0].tolist()
    total_signals = len(signal_indices)

    raw_trades_df = compute_trade_pnl(df, signal_indices, target)
    # Update to count only signals that yielded valid trades (some near end-of-data get dropped)
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
    open_slots: list = []       # 至多 max_concurrent 個 dict
    closed_trades: list = []
    equity_log: list = []       # (exit_bar, equity) 步階點
    skipped: int = 0

    for row in raw_trades_df.itertuples(index=False):
        # A. 釋放容量：exit_bar ≤ 當前 entry_idx 的舊倉先平倉
        to_close = [t for t in open_slots if t['exit_bar'] <= row.entry_idx]
        to_close.sort(key=lambda t: t['exit_bar'])
        for t in to_close:
            equity += t['pnl_usd']
            equity_log.append((t['exit_bar'], equity))
            closed_trades.append({**t, 'trade_roe': t['pnl_usd'] / t['equity_at_entry']})
        open_slots = [t for t in open_slots if t['exit_bar'] > row.entry_idx]

        # B. 併發上限檢查
        if len(open_slots) >= max_concurrent:
            skipped += 1
            continue

        # C. 動態部位計算
        if target == 'target_fixed':
            sl_distance = row.entry_price * 0.01        # 固定 1% SL
        else:
            sl_distance = 1.5 * row.atr_at_entry        # 動態 ATR SL

        if sl_distance <= 0:
            skipped += 1
            continue

        risk_budget  = equity * risk_pct
        position_qty = risk_budget / sl_distance
        position_usd = position_qty * row.entry_price
        pnl_usd      = position_usd * row.pnl           # row.pnl 已含手續費

        # D. 登記開倉
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

    # 迴圈結束：flush 剩餘開倉
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
    }


def _compute_metrics(
    closed_trades: list,
    equity_log: list,
    initial_equity: float,
    final_equity: float,
    df_len: int,
) -> dict:
    if not closed_trades:
        return {}

    total_years = df_len / 8760.0
    n_trades = len(closed_trades)

    # Portfolio Sharpe：使用 trade_roe = pnl_usd / equity_at_entry（帳戶級回報）
    roe_arr = np.array([t['trade_roe'] for t in closed_trades])
    mean_roe = float(roe_arr.mean())
    std_roe  = float(roe_arr.std(ddof=1))
    sharpe   = float(mean_roe / std_roe * np.sqrt(n_trades / total_years)) if std_roe > 0 else 0.0

    # 資金曲線（插入初始點 bar=0, equity=initial_equity，確保 MDD 從起點計算）
    equity_arr = np.array([initial_equity] + [e for _, e in equity_log])
    running_peak = np.maximum.accumulate(equity_arr)
    dd_arr       = equity_arr - running_peak
    mdd_usd_val  = float(dd_arr.min())

    # MDD% = 谷值相對峰值的跌幅
    peak_at_trough = running_peak[np.argmin(dd_arr)]
    mdd_pct = float(mdd_usd_val / peak_at_trough * 100) if peak_at_trough > 0 else 0.0

    # CAGR
    total_return = final_equity / initial_equity
    cagr = (total_return ** (1.0 / total_years) - 1.0) * 100 if total_years > 0 else 0.0

    win_rate = float(sum(1 for t in closed_trades if t['pnl_usd'] > 0) / n_trades)
    avg_pos  = float(np.mean([t['position_usd'] for t in closed_trades]))
    avg_hold = float(np.mean([t['exit_bar'] - t['entry_idx'] for t in closed_trades]))

    return {
        'total_return_pct': round((total_return - 1) * 100, 4),
        'cagr_pct':         round(cagr, 4),
        'sharpe_ratio':     round(sharpe, 4),
        'max_drawdown_usd': round(mdd_usd_val, 2),
        'max_drawdown_pct': round(mdd_pct, 4),
        'win_rate':         round(win_rate, 4),
        'avg_position_usd': round(avg_pos, 2),
        'avg_holding_bars': round(avg_hold, 2),
    }


def _empty_results(initial_equity: float, total_signals: int) -> dict:
    return {
        'closed_trades':     [],
        'equity_log':        [],
        'final_equity':      initial_equity,
        'total_signals':     total_signals,
        'executed_trades':   0,
        'skipped_signals':   0,
        'metrics':           {},
        'initial_equity':    initial_equity,
        'risk_pct':          0.02,
        'max_concurrent':    3,
        'optimal_threshold': None,
    }
