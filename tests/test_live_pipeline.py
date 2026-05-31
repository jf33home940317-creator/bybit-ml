import pandas as pd
import pytest


# ─── _drop_partial_bars helper ────────────────────────────────────────────────

class TestDropPartialBars:
    """Bybit returns the still-forming candle as the last row of kline responses.
    _drop_partial_bars removes any bar whose period has not yet fully closed."""

    def test_drops_partial_hourly_bar(self, monkeypatch):
        from live import pipeline

        fake_now = pd.Timestamp("2026-06-01T12:30:00+00:00")
        monkeypatch.setattr(
            pipeline.pd.Timestamp, "now",
            classmethod(lambda cls, tz="UTC": fake_now),
        )

        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-06-01T10:00:00+00:00",
                "2026-06-01T11:00:00+00:00",
                "2026-06-01T12:00:00+00:00",
            ], utc=True),
            "close": [100.0, 101.0, 102.5],
        })

        result = pipeline._drop_partial_bars(df, period_hours=1)

        assert len(result) == 2
        assert result["timestamp"].iloc[-1] == pd.Timestamp("2026-06-01T11:00:00+00:00")

    def test_drops_partial_daily_bar(self, monkeypatch):
        from live import pipeline

        fake_now = pd.Timestamp("2026-06-01T12:30:00+00:00")
        monkeypatch.setattr(
            pipeline.pd.Timestamp, "now",
            classmethod(lambda cls, tz="UTC": fake_now),
        )

        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-05-30T00:00:00+00:00",
                "2026-05-31T00:00:00+00:00",
                "2026-06-01T00:00:00+00:00",
            ], utc=True),
            "close": [100.0, 101.0, 102.5],
        })

        result = pipeline._drop_partial_bars(df, period_hours=24)

        assert len(result) == 2
        assert result["timestamp"].iloc[-1] == pd.Timestamp("2026-05-31T00:00:00+00:00")

    def test_keeps_all_when_no_partial_present(self, monkeypatch):
        from live import pipeline

        fake_now = pd.Timestamp("2026-06-01T13:05:00+00:00")
        monkeypatch.setattr(
            pipeline.pd.Timestamp, "now",
            classmethod(lambda cls, tz="UTC": fake_now),
        )

        df = pd.DataFrame({
            "timestamp": pd.to_datetime([
                "2026-06-01T11:00:00+00:00",
                "2026-06-01T12:00:00+00:00",
            ], utc=True),
            "close": [100.0, 101.0],
        })

        result = pipeline._drop_partial_bars(df, period_hours=1)
        assert len(result) == 2


# ─── _check_freshness helper ──────────────────────────────────────────────────

class TestCheckFreshness:
    """compute_signal should refuse to act on stale data."""

    def test_raises_when_last_closed_bar_too_old(self, monkeypatch):
        from live import pipeline

        fake_now = pd.Timestamp("2026-06-01T18:00:00+00:00")
        monkeypatch.setattr(
            pipeline.pd.Timestamp, "now",
            classmethod(lambda cls, tz="UTC": fake_now),
        )

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-06-01T14:00:00+00:00"], utc=True),
            "close": [100.0],
        })

        with pytest.raises(ValueError, match=r"(?i)stale"):
            pipeline._check_freshness(df, period_hours=1, max_age_hours=1.5)

    def test_passes_when_last_closed_bar_recent(self, monkeypatch):
        from live import pipeline

        fake_now = pd.Timestamp("2026-06-01T12:05:00+00:00")
        monkeypatch.setattr(
            pipeline.pd.Timestamp, "now",
            classmethod(lambda cls, tz="UTC": fake_now),
        )

        df = pd.DataFrame({
            "timestamp": pd.to_datetime(["2026-06-01T11:00:00+00:00"], utc=True),
            "close": [100.0],
        })

        # No exception
        pipeline._check_freshness(df, period_hours=1, max_age_hours=1.5)
