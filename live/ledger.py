import json
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
    f = Path(ledger_file) if ledger_file else LEDGER_FILE
    f.parent.mkdir(parents=True, exist_ok=True)
    records = load_ledger(ledger_file=f)
    records.append(trade)
    with open(f, "w", encoding="utf-8") as fp:
        json.dump(records, fp, indent=2)
