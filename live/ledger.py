import json
import os
import tempfile
from pathlib import Path
import config

LEDGER_FILE = config.STORAGE_LIVE / "paper_trading_ledger.json"


def load_ledger(ledger_file: Path = None) -> list:
    f = Path(ledger_file) if ledger_file else LEDGER_FILE
    if not f.exists():
        return []
    with open(f, encoding="utf-8") as fp:
        return json.load(fp)


def append_entry(trade: dict, ledger_file: Path = None) -> None:
    """Append a single trade record to the ledger JSON file.

    Crash-safe: writes to a sibling temp file first, then atomically renames.
    NOT safe for concurrent writers — the single-threaded heartbeat loop
    makes this safe in practice.
    """
    f = Path(ledger_file) if ledger_file else LEDGER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    records = load_ledger(ledger_file=f)
    records.append(trade)

    fd, tmp_path = tempfile.mkstemp(dir=f.parent, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fp:
            json.dump(records, fp, indent=2)
        os.replace(tmp_path, f)
    except Exception:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
        raise
