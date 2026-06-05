"""Side-by-side terminal table comparing long vs short threshold-scan + combined metrics.

Reads:
  storage/backtest/{SYMBOL}_target_atr_threshold_scan.json
  storage/backtest/{SYMBOL}_target_atr_short_threshold_scan.json
  storage/backtest/{SYMBOL}_long_short_combined.json

Prints a single table. No file output.
"""
import json
import sys
from pathlib import Path

import config

SYMBOLS = ("ETHUSDT", "BTCUSDT")


def _load(path: Path) -> dict | None:
    if not path.exists():
        return None
    return json.loads(path.read_text())


def _fmt(val, suffix: str = "", precision: int = 2) -> str:
    if val is None:
        return "—"
    if isinstance(val, (int, float)):
        return f"{val:.{precision}f}{suffix}"
    return f"{val}{suffix}"


def main() -> int:
    backtest_dir = config.STORAGE_BACKTEST

    rows = []
    for symbol in SYMBOLS:
        long_scan  = _load(backtest_dir / f"{symbol}_target_atr_threshold_scan.json")
        short_scan = _load(backtest_dir / f"{symbol}_target_atr_short_threshold_scan.json")
        combined   = _load(backtest_dir / f"{symbol}_long_short_combined.json")

        long_opt  = (long_scan  or {}).get('optimal_metrics') or {}
        short_opt = (short_scan or {}).get('optimal_metrics') or {}
        comb_m    = (combined   or {}).get('combined_metrics') or {}

        rows.append({
            'symbol': symbol,
            'long_sharpe':   long_opt.get('sharpe_ratio'),
            'long_winrate':  long_opt.get('win_rate'),
            'long_n':        long_opt.get('n_trades'),
            'short_sharpe':  short_opt.get('sharpe_ratio'),
            'short_winrate': short_opt.get('win_rate'),
            'short_n':       short_opt.get('n_trades'),
            'combined_sharpe': comb_m.get('sharpe_ratio'),
            'combined_mdd':    comb_m.get('max_drawdown_pct'),
            'correlation':     (combined or {}).get('daily_return_correlation'),
        })

    if not rows:
        print("No reports found.")
        return 1

    label_w = 20
    col_w   = 14
    headers = ['Metric'] + [r['symbol'] for r in rows]
    print()
    print(headers[0].ljust(label_w) + ''.join(h.ljust(col_w) for h in headers[1:]))
    print('-' * (label_w + col_w * len(rows)))
    metric_rows = [
        ('Long  Sharpe',     'long_sharpe',     '',  2),
        ('Long  Win%',       'long_winrate',    '',  4),
        ('Long  Trades',     'long_n',          '',  0),
        ('Short Sharpe',     'short_sharpe',    '',  2),
        ('Short Win%',       'short_winrate',   '',  4),
        ('Short Trades',     'short_n',         '',  0),
        ('Combined Sharpe',  'combined_sharpe', '',  2),
        ('Combined MDD%',    'combined_mdd',    '%', 2),
        ('Long-Short ρ',     'correlation',     '',  3),
    ]
    for label, key, suffix, prec in metric_rows:
        line = label.ljust(label_w)
        for r in rows:
            line += _fmt(r[key], suffix, prec).ljust(col_w)
        print(line)
    print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
