import pandas as pd
import pytest


# ─── _realized_pnl_pct ─────────────────────────────────────────────────────────

class TestRealizedPnlPct:
    """Live P&L must deduct round-trip fee, matching backtest engine.compute_trade_pnl."""

    def test_winning_trade_pays_fee(self):
        from run_live import _realized_pnl_pct
        # +2% gross → +1.8% after 0.2% fee
        assert _realized_pnl_pct(entry=2000.0, exit_price=2040.0) == pytest.approx(1.8, abs=1e-4)

    def test_losing_trade_pays_fee(self):
        from run_live import _realized_pnl_pct
        # -1% gross → -1.2% after fee
        assert _realized_pnl_pct(entry=2000.0, exit_price=1980.0) == pytest.approx(-1.2, abs=1e-4)

    def test_flat_trade_pays_only_fee(self):
        from run_live import _realized_pnl_pct
        assert _realized_pnl_pct(entry=2000.0, exit_price=2000.0) == pytest.approx(-0.2, abs=1e-4)


# ─── _build_daily_summary ──────────────────────────────────────────────────────

class TestBuildDailySummary:
    """24h health-heartbeat summary must reflect ledger activity in the past day."""

    def test_counts_signals_and_results_within_24h(self):
        from run_live import _build_daily_summary
        now = pd.Timestamp("2026-06-01T00:01:00+00:00")
        records = [
            # outside 24h window — should be ignored
            {"status": "open",  "entry_time": "2026-05-30T10:00:00+00:00"},
            # within 24h
            {"status": "open",  "entry_time": "2026-05-31T05:00:00+00:00"},
            {"status": "open",  "entry_time": "2026-05-31T15:00:00+00:00"},
            # close within 24h, win
            {"outcome": "win", "exit_time_actual": "2026-05-31T20:00:00+00:00",
             "position_usd": 10_000, "pnl_pct": 2.5},
            # close within 24h, loss
            {"outcome": "loss", "exit_time_actual": "2026-05-31T22:00:00+00:00",
             "position_usd": 8_000, "pnl_pct": -1.5},
        ]
        state = {"positions": [{"symbol": "ETHUSDT"}]}

        msg = _build_daily_summary(records, state, now)

        assert "2" in msg, "two signals within 24h"
        assert "1 贏" in msg
        assert "1 輸" in msg
        assert "持倉: 1" in msg

    def test_zero_activity_window(self):
        from run_live import _build_daily_summary
        now = pd.Timestamp("2026-06-01T00:01:00+00:00")
        msg = _build_daily_summary([], {"positions": []}, now)
        assert "0" in msg


# ─── _ErrorThrottle ────────────────────────────────────────────────────────────

class TestErrorThrottle:

    def test_allows_first_alert_per_key(self):
        from run_live import _ErrorThrottle
        t = _ErrorThrottle(cooldown_hours=6)
        now = pd.Timestamp("2026-06-01T00:00:00+00:00")
        assert t.should_alert("ETHUSDT_signal", now) is True

    def test_blocks_duplicate_within_cooldown(self):
        from run_live import _ErrorThrottle
        t = _ErrorThrottle(cooldown_hours=6)
        now = pd.Timestamp("2026-06-01T00:00:00+00:00")
        t.should_alert("ETHUSDT_signal", now)
        # 1 hour later — still in cooldown
        later = now + pd.Timedelta(hours=1)
        assert t.should_alert("ETHUSDT_signal", later) is False

    def test_allows_after_cooldown(self):
        from run_live import _ErrorThrottle
        t = _ErrorThrottle(cooldown_hours=6)
        now = pd.Timestamp("2026-06-01T00:00:00+00:00")
        t.should_alert("ETHUSDT_signal", now)
        later = now + pd.Timedelta(hours=7)
        assert t.should_alert("ETHUSDT_signal", later) is True

    def test_independent_keys(self):
        from run_live import _ErrorThrottle
        t = _ErrorThrottle(cooldown_hours=6)
        now = pd.Timestamp("2026-06-01T00:00:00+00:00")
        assert t.should_alert("ETHUSDT_signal", now) is True
        assert t.should_alert("BTCUSDT_signal", now) is True
