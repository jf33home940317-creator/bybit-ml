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


# ─── _is_disabled (kill switch) ────────────────────────────────────────────────

class TestIsDisabled:
    """Touch .disabled to pause signal generation; rm it to resume."""

    def test_returns_false_when_no_flag_file(self, tmp_path):
        from run_live import _is_disabled
        assert _is_disabled(tmp_path / ".disabled") is False

    def test_returns_true_when_flag_file_exists(self, tmp_path):
        from run_live import _is_disabled
        flag = tmp_path / ".disabled"
        flag.touch()
        assert _is_disabled(flag) is True


# ─── _record_prob (prob history CSV) ───────────────────────────────────────────

class TestRecordProb:
    """Every heartbeat writes one row to prob_history.csv."""

    def test_creates_csv_with_header_on_first_write(self, tmp_path):
        from run_live import _record_prob
        csv_path = tmp_path / "prob_history.csv"
        _record_prob(csv_path, "2026-06-01T12:01:00+00:00", "ETHUSDT", 0.5415, False, 2001.5)
        lines = csv_path.read_text().strip().split("\n")
        assert lines[0] == "timestamp,symbol,probability,signal,close"
        assert "ETHUSDT" in lines[1]
        assert "0.5415" in lines[1]

    def test_appends_without_repeating_header(self, tmp_path):
        from run_live import _record_prob
        csv_path = tmp_path / "prob_history.csv"
        _record_prob(csv_path, "2026-06-01T12:00:00+00:00", "ETHUSDT", 0.54, False, 2000.0)
        _record_prob(csv_path, "2026-06-01T13:00:00+00:00", "ETHUSDT", 0.78, True, 2010.0)
        lines = csv_path.read_text().strip().split("\n")
        assert len(lines) == 3  # header + 2 rows
        header_count = sum(1 for l in lines if l.startswith("timestamp,"))
        assert header_count == 1


# ─── _get_assets (model cache) ─────────────────────────────────────────────────

class TestGetAssets:
    """load_assets should only be called once per symbol+target pair."""

    def test_caches_result_after_first_call(self, monkeypatch):
        from run_live import _get_assets, _assets_cache
        _assets_cache.clear()

        call_count = {"n": 0}
        fake_cols = ["rsi_14", "ppo"]
        fake_models = ["model1", "model2"]

        def mock_load(symbol, target):
            call_count["n"] += 1
            return fake_cols, fake_models

        import run_live
        monkeypatch.setattr(run_live.pipeline, "load_assets", mock_load)

        r1 = _get_assets("ETHUSDT", "target_atr")
        r2 = _get_assets("ETHUSDT", "target_atr")
        assert r1 == r2 == (fake_cols, fake_models)
        assert call_count["n"] == 1, "load_assets should only be called once"
        _assets_cache.clear()

    def test_different_keys_call_separately(self, monkeypatch):
        from run_live import _get_assets, _assets_cache
        _assets_cache.clear()

        call_count = {"n": 0}
        def mock_load(symbol, target):
            call_count["n"] += 1
            return [f"col_{symbol}"], [f"model_{symbol}"]

        import run_live
        monkeypatch.setattr(run_live.pipeline, "load_assets", mock_load)

        _get_assets("ETHUSDT", "target_atr")
        _get_assets("BTCUSDT", "target_atr")
        assert call_count["n"] == 2
        _assets_cache.clear()
