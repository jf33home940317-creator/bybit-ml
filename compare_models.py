"""Compare old vs new model backtest results. Auto-rollback if Sharpe drops."""
import json
import shutil
import logging
from pathlib import Path
import config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYMBOL = "ETHUSDT"
TARGET = "target_atr"


def main():
    old_report = Path("storage/backtest_v1") / f"{SYMBOL}_{TARGET}_portfolio_report.json"
    new_report = config.STORAGE_BACKTEST / f"{SYMBOL}_{TARGET}_portfolio_report.json"

    if not old_report.exists() or not new_report.exists():
        logger.error("Cannot compare — missing report files")
        return

    old = json.loads(old_report.read_text())
    new = json.loads(new_report.read_text())

    old_sharpe = old["metrics"]["sharpe_ratio"]
    new_sharpe = new["metrics"]["sharpe_ratio"]
    old_return = old["metrics"]["total_return_pct"]
    new_return = new["metrics"]["total_return_pct"]
    old_mdd = old["metrics"]["max_drawdown_pct"]
    new_mdd = new["metrics"]["max_drawdown_pct"]

    print("=" * 60)
    print(f"  Phase 6 Model Comparison: {SYMBOL} {TARGET}")
    print("=" * 60)
    print(f"  {'Metric':<25} {'Old':>12} {'New':>12} {'Delta':>12}")
    print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*12}")
    print(f"  {'Sharpe Ratio':<25} {old_sharpe:>12.4f} {new_sharpe:>12.4f} {new_sharpe-old_sharpe:>+12.4f}")
    print(f"  {'Total Return %':<25} {old_return:>12.2f} {new_return:>12.2f} {new_return-old_return:>+12.2f}")
    print(f"  {'Max Drawdown %':<25} {old_mdd:>12.2f} {new_mdd:>12.2f} {new_mdd-old_mdd:>+12.2f}")
    print("=" * 60)

    if new_sharpe >= old_sharpe:
        print(f"\n  NEW MODEL WINS (Sharpe {old_sharpe:.4f} -> {new_sharpe:.4f})")
        print("  Keeping new models. Safe to deploy.")
    else:
        print(f"\n  OLD MODEL BETTER (Sharpe {new_sharpe:.4f} < {old_sharpe:.4f})")
        print("  Rolling back to v1 models...")
        shutil.rmtree(config.STORAGE_MODELS)
        shutil.copytree("storage/models_v1", str(config.STORAGE_MODELS))
        shutil.rmtree(config.STORAGE_FEATURES)
        shutil.copytree("storage/features_v1", str(config.STORAGE_FEATURES))
        shutil.rmtree(config.STORAGE_BACKTEST)
        shutil.copytree("storage/backtest_v1", str(config.STORAGE_BACKTEST))
        print("  Rollback complete. Old models restored.")


if __name__ == "__main__":
    main()
