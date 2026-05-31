import json
import os
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
    tmp = f.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as fp:
        json.dump(state, fp, indent=2)
    os.replace(tmp, f)   # atomic rename — Crash Recovery


def count_active(state: dict) -> int:
    return len(state["positions"])


def add_position(state: dict, position: dict) -> dict:
    state["positions"].append(position)
    return state


def expire_closed_positions(state: dict, now: pd.Timestamp) -> tuple:
    """Remove positions whose exit_time <= now. Returns (new_state, expired_list)."""
    expired = [p for p in state["positions"] if pd.Timestamp(p["exit_time"]) <= now]
    kept    = [p for p in state["positions"] if pd.Timestamp(p["exit_time"]) >  now]
    return {"positions": kept}, expired
