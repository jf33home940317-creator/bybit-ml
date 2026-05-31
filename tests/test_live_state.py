import json
import pytest
import pandas as pd
from pathlib import Path


# ─── state.py 測試 ────────────────────────────────────────────────────────────

class TestLoadState:

    def test_returns_empty_when_no_file(self, tmp_path):
        from live.state import load_state
        result = load_state(state_file=tmp_path / "active_positions.json")
        assert result == {"positions": []}

    def test_load_returns_saved_data(self, tmp_path):
        from live.state import save_state, load_state
        sf = tmp_path / "active_positions.json"
        data = {"positions": [{"symbol": "ETHUSDT", "exit_time": "2030-01-01T00:00:00+00:00"}]}
        save_state(data, state_file=sf)
        assert load_state(state_file=sf) == data


class TestExpirePositions:

    def test_removes_past_positions(self):
        from live.state import expire_closed_positions
        now = pd.Timestamp.now('UTC')
        past   = (now - pd.Timedelta(hours=1)).isoformat()
        future = (now + pd.Timedelta(hours=1)).isoformat()
        s = {"positions": [{"symbol": "A", "exit_time": past},
                            {"symbol": "B", "exit_time": future}]}
        new_s, expired = expire_closed_positions(s, now)
        assert len(new_s["positions"]) == 1
        assert new_s["positions"][0]["symbol"] == "B"
        assert len(expired) == 1
        assert expired[0]["symbol"] == "A"

    def test_keeps_all_when_none_expired(self):
        from live.state import expire_closed_positions
        now    = pd.Timestamp.now('UTC')
        future = (now + pd.Timedelta(hours=1)).isoformat()
        s = {"positions": [{"symbol": "A", "exit_time": future}]}
        new_s, expired = expire_closed_positions(s, now)
        assert len(new_s["positions"]) == 1
        assert expired == []


class TestCountAndAdd:

    def test_count_active(self):
        from live.state import count_active
        s = {"positions": [{"symbol": "A"}, {"symbol": "B"}]}
        assert count_active(s) == 2

    def test_add_position_appends(self):
        from live.state import add_position
        s   = {"positions": []}
        pos = {"symbol": "ETHUSDT", "entry_price": 3000.0}
        new_s = add_position(s, pos)
        assert len(new_s["positions"]) == 1
        assert new_s["positions"][0]["entry_price"] == 3000.0
        assert len(s["positions"]) == 0, "add_position must not mutate the original state"


# ─── ledger.py 測試 ───────────────────────────────────────────────────────────

class TestLedger:

    def test_append_entry_creates_file(self, tmp_path):
        from live.ledger import append_entry, load_ledger
        lf = tmp_path / "ledger.json"
        append_entry({"symbol": "ETHUSDT", "status": "open"}, ledger_file=lf)
        records = load_ledger(ledger_file=lf)
        assert len(records) == 1
        assert records[0]["symbol"] == "ETHUSDT"

    def test_multiple_entries_accumulate(self, tmp_path):
        from live.ledger import append_entry, load_ledger
        lf = tmp_path / "ledger.json"
        append_entry({"id": 1}, ledger_file=lf)
        append_entry({"id": 2}, ledger_file=lf)
        records = load_ledger(ledger_file=lf)
        assert len(records) == 2

    def test_load_returns_empty_list_when_no_file(self, tmp_path):
        from live.ledger import load_ledger
        records = load_ledger(ledger_file=tmp_path / "nonexistent.json")
        assert records == []

    def test_existing_ledger_preserved_when_write_fails(self, tmp_path, monkeypatch):
        """Crash-safety: if json.dump raises mid-write, the previously-persisted
        ledger contents must remain intact (not truncated or corrupt)."""
        import json as _json
        from live import ledger
        lf = tmp_path / "ledger.json"
        # Pre-write a known-good ledger
        ledger.append_entry({"id": "first"}, ledger_file=lf)
        ledger.append_entry({"id": "second"}, ledger_file=lf)

        # Force the next dump to raise
        def _boom(*args, **kwargs):
            raise OSError("simulated disk failure")
        monkeypatch.setattr(ledger.json, "dump", _boom)

        with pytest.raises(OSError):
            ledger.append_entry({"id": "third"}, ledger_file=lf)

        # Original file must still be readable with original contents
        with open(lf, encoding="utf-8") as fp:
            recovered = _json.load(fp)
        assert recovered == [{"id": "first"}, {"id": "second"}]

    def test_no_tmp_files_left_after_successful_write(self, tmp_path):
        from live import ledger
        lf = tmp_path / "ledger.json"
        ledger.append_entry({"id": 1}, ledger_file=lf)
        ledger.append_entry({"id": 2}, ledger_file=lf)
        leftover = list(tmp_path.glob("*.tmp"))
        assert leftover == [], f"Temp files not cleaned up: {leftover}"
