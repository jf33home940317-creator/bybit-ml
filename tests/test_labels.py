import numpy as np
import pandas as pd

from features.labels import HORIZON


def _make_df(n, closes, highs, lows):
    ts = pd.date_range("2022-01-01", periods=n, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open": closes.copy(),
        "high": highs,
        "low": lows,
        "close": closes.copy(),
        "volume": np.ones(n),
        "turnover": np.ones(n) * 100.0,
        "atr_14": np.ones(n) * 1.0,
        "atr_24": np.ones(n) * 1.0,
    })


class TestLabels:
    def test_tp_hit_within_horizon_label_1(self):
        """Bar 5 high >= TP (+2%) → entry bar 0 gets label=1."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[5] = 102.0          # bar 5: high hits TP (100 * 1.02 = 102.0)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 1.0

    def test_sl_hit_within_horizon_label_0(self):
        """Bar 3 low <= SL (-1%) → entry bar 0 gets label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        lows[3] = 98.9            # bar 3: low hits SL (100 * 0.99 = 99.0)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_time_limit_no_hit_label_0(self):
        """No barrier hit within 24 bars → time limit → label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)   # never reaches 102
        lows = np.full(n, 99.5)     # never reaches 99
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_intra_bar_collision_sl_wins(self):
        """Same bar hits both TP and SL → conservative pessimism → label=0."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[2] = 102.1           # bar 2 hits TP
        lows[2] = 98.9             # bar 2 also hits SL (same bar collision)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 0.0

    def test_tail_nans_last_horizon_rows(self):
        """Last HORIZON rows must have NaN for both target columns."""
        from features.labels import compute
        n = HORIZON + 10
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[-HORIZON:].isna().all()
        assert result["target_atr"].iloc[-HORIZON:].isna().all()

    def test_atr_barrier_uses_multipliers(self):
        """target_atr TP = close + 3*atr → high=102.1 misses TP when atr=1 (TP=103)."""
        from features.labels import compute
        n = HORIZON + 5
        closes = np.full(n, 100.0)
        highs = np.full(n, 100.5)
        lows = np.full(n, 99.5)
        highs[5] = 102.1          # hits target_fixed TP (102.0) but NOT target_atr TP (103.0)
        df = _make_df(n, closes, highs, lows)
        result = compute(df)
        assert result["target_fixed"].iloc[0] == 1.0   # fixed TP hit
        assert result["target_atr"].iloc[0] == 0.0     # ATR TP not hit (need 103.0, got 102.1)
