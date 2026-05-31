import json
import os
import tempfile
from pathlib import Path
import pandas as pd
import config

STATE_FILE = config.STORAGE_LIVE / "active_positions.json"


def load_state(state_file: Path = None) -> dict:
    f = Path(state_file) if state_file else STATE_FILE
    if not f.exists():
        return {"positions": []}
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def save_state(state: dict, state_file: Path = None) -> None:
    f = Path(state_file) if state_file else STATE_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=f.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(state, fp, indent=2)
        os.replace(tmp_path, f)   # crash-safe; on Windows raises PermissionError if target is open
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise


def count_active(state: dict) -> int:
    return len(state["positions"])


def add_position(state: dict, position: dict) -> dict:
    return {"positions": state["positions"] + [position]}


def expire_closed_positions(state: dict, now: pd.Timestamp) -> tuple[dict, list]:
    """Remove positions whose exit_time <= now. Returns (new_state, expired_list).

    'now' must be UTC-aware. exit_time strings are treated as UTC if they
    carry no explicit offset.
    """
    def _as_utc(ts_str: str) -> pd.Timestamp:
        t = pd.Timestamp(ts_str)
        if t.tzinfo is None:
            t = t.tz_localize("UTC")
        return t

    expired = [p for p in state["positions"] if _as_utc(p["exit_time"]) <= now]
    kept    = [p for p in state["positions"] if _as_utc(p["exit_time"]) >  now]
    return {"positions": kept}, expired
