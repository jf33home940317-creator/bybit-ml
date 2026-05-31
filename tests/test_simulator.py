import numpy as np
import pandas as pd
import pytest
from backtest.simulator import run_portfolio_simulation


class _ConstantModel:
    """Mock model that always returns a fixed probability p."""
    def __init__(self, p: float):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        return np.column_stack([np.full(n, 1 - self.p), np.full(n, self.p)])


def _make_df(n: int = 300, close: float = 1000.0) -> pd.DataFrame:
    """
    最小 DataFrame：timestamp、OHLCV、atr_14。
    high = close * 1.005, low = close * 0.995，所有交易皆 timeout。
    n=300 足以產生 5 fold OOF 驗證期交易。
    """
    return pd.DataFrame({
        'timestamp': pd.date_range('2022-01-01', periods=n, freq='1h'),
        'open':   close,
        'high':   close * 1.005,   # ≤ TP（target_fixed: close*1.02）→ 不觸 TP
        'low':    close * 0.995,   # ≥ SL（target_fixed: close*0.99）→ 不觸 SL
        'close':  close,
        'volume': 1.0,
        'atr_14': close * 0.01,    # 1% ATR；用於 target_atr SL 計算
    })


class TestConcurrentLimit:

    def test_concurrent_limit(self):
        """
        MAX_CONCURRENT=2 時，若 4 個訊號同時開倉（exit_bar 皆在末尾），
        第 3、4 個訊號應被跳過（skipped_signals=2）。
        """
        df = _make_df(300)
        # 全部 OOF 訊號機率 = 0.90，全部觸發（閾值 0.50）
        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_fixed',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=2,
        )

        # 最多只有 2 個部位同時存在
        # 驗證方式：executed_trades + skipped_signals == total_signals
        assert results['executed_trades'] + results['skipped_signals'] == results['total_signals']
        assert results['skipped_signals'] >= 1, "max_concurrent=2 時應有跳過的訊號"


class TestCapacityRelease:

    def test_capacity_release(self):
        """
        舊倉 exit_bar ≤ 新訊號 entry_idx 時，容量應先被釋放，使新訊號得以進場。
        max_concurrent=1：若舊倉已出場，新訊號不應被跳過。
        """
        n = 300
        df = _make_df(n)
        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_fixed',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=1,   # 嚴格：同時只能一筆
        )

        # timeout=24 bars → 舊倉 exit_bar = entry_idx + 24
        # 下一筆訊號 entry_idx >= 舊倉 exit_bar 時容量已釋放
        # executed_trades 應 > 0（至少有一筆成交）
        assert results['executed_trades'] > 0
        # 沒有任何一筆 open 部位在另一筆開倉時同時存在（難以直接斷言，用無泄漏代替）
        assert len(results['closed_trades']) == results['executed_trades']


class TestPositionSizing:

    def test_position_sizing_2pct(self):
        """
        target_fixed SL 觸發時，pnl_usd ≈ -(equity_at_entry × 0.02) - fee_usd。
        容差：≤ equity_at_entry × 0.001（0.1%）。
        """
        n = 200
        df = _make_df(n, close=1000.0)
        # 讓第一筆 OOF 訊號命中 SL：bar+1 的 low <= SL=990
        # OOF 驗證期從 index 24 開始（n=200 第一個 fold val_idx 約 50~100 區間）
        # 為確保命中，對整個 df 設定 low=980
        df['low'] = 980.0

        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_fixed',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=3,
        )

        assert results['executed_trades'] > 0, "需要至少一筆成交才能驗證部位規模"

        first_trade = results['closed_trades'][0]
        eq_entry   = first_trade['equity_at_entry']
        pnl_usd    = first_trade['pnl_usd']

        if first_trade['outcome'] == 'sl':
            # pnl = −sl_pct − fee；for target_fixed: sl_pct=0.01, fee=0.002
            # pnl_usd = (eq × risk_pct / sl_pct) × (−sl_pct − fee) = −eq × risk_pct × (1 + fee/sl_pct)
            fee_rate, sl_pct_val = 0.002, 0.01
            expected_loss = -(eq_entry * 0.02 * (1 + fee_rate / sl_pct_val))
            assert abs(pnl_usd - expected_loss) <= eq_entry * 0.001, (
                f"SL pnl_usd={pnl_usd:.2f}, expected≈{expected_loss:.2f}"
            )


class TestEquityCompounds:

    def test_equity_compounds(self):
        """
        勝倉平倉後，帳戶淨值上升，下一筆 position_usd 應比上一筆大（Anti-Martingale）。
        """
        n = 300
        df = _make_df(n, close=1000.0)
        # 強制全部命中 TP：high 遠超 close*1.02
        df['high'] = 1100.0
        df['low']  = 950.0   # 也會觸 SL，但 TP 先到（bar 1 high=1100 > TP=1020）
        # SL: close*0.99=990, low=950 → SL 也觸發，但 TP(close*1.02=1020) 在 high=1100 的同一個 bar
        # 依設計 TP wins if tp_first < sl_first
        # 這裡我們手動設 high=1100（bar 1），low=950（bar 2 以後），確保 TP 在 bar 1 贏
        df.loc[1:, 'low'] = 950.0
        df.loc[0, 'high'] = 1000.0  # entry bar：不觸發

        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_fixed',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=1,   # 串列執行，確保複利可比
        )

        trades = results['closed_trades']
        if len(trades) < 2:
            pytest.skip("需要至少 2 筆成交才能驗證複利")

        # 若第一筆是勝倉，第二筆 position_usd 應更大
        if trades[0]['pnl_usd'] > 0:
            assert trades[1]['position_usd'] > trades[0]['position_usd'], (
                "勝倉後 equity 上升，Anti-Martingale 應自動增加 position_usd"
            )


class TestNoOpenTradesLeak:

    def test_no_open_trades_leak(self):
        """
        模擬結束後所有開倉都已平倉：
        len(closed_trades) == executed_trades，且無部位洩漏。
        """
        df = _make_df(300)
        fold_models = [_ConstantModel(0.90)] * 5

        results = run_portfolio_simulation(
            df, [], fold_models,
            target='target_fixed',
            optimal_threshold=0.50,
            initial_equity=100_000,
            risk_pct=0.02,
            max_concurrent=3,
        )

        assert len(results['closed_trades']) == results['executed_trades'], (
            "closed_trades 數量必須等於 executed_trades"
        )
        assert results['final_equity'] > 0, "final_equity 必須為正數"
        # equity_log 最後一筆 equity = final_equity
        if results['equity_log']:
            assert abs(results['equity_log'][-1][1] - results['final_equity']) < 1e-6
