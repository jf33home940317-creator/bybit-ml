#!/usr/bin/env python3
"""顯示 paper trading 正確率與資金曲線。執行:python show_results.py"""
import json
import sys
import io
import config

# Windows 終端機 UTF-8 輸出
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")


def main() -> None:
    lf = config.STORAGE_LIVE / "paper_trading_ledger.json"
    if not lf.exists():
        print("還沒有任何紀錄。等訊號觸發後再執行。")
        return

    records = json.loads(lf.read_text(encoding="utf-8"))

    opens  = [r for r in records if r.get("status") == "open"]
    closes = [r for r in records if "outcome" in r]

    print("=" * 55)
    print("  BYBIT_ML Paper Trading 正確率報告")
    print("=" * 55)
    print(f"  總共觸發訊號:{len(opens)} 次")
    print(f"  已結算筆數:  {len(closes)} 筆")
    print(f"  目前持倉中:  {len(opens) - len(closes)} 筆(等待 24h 結算)")

    if not closes:
        print("\n  尚無結算紀錄,請等訊號觸發後 24 小時再查。")
        return

    wins        = [r for r in closes if r.get("outcome") == "win"]
    losses      = [r for r in closes if r.get("outcome") == "loss"]
    no_exit_px  = [r for r in closes if r.get("outcome") == "timeout"]

    accuracy  = len(wins) / len(closes) * 100
    pnls      = [r["pnl_pct"] for r in closes if r.get("pnl_pct") is not None]
    avg_pnl   = sum(pnls) / len(pnls) if pnls else 0
    total_pnl = sum(pnls)

    print()
    print(f"  ✅ 漲(正確):{len(wins)} 次")
    print(f"  ❌ 跌(錯誤):{len(losses)} 次")
    if no_exit_px:
        print(f"  ⚠️  出場價抓取失敗:{len(no_exit_px)} 次")
    print(f"  ─────────────────────────")
    print(f"  >>> 方向正確率:{accuracy:.1f}% <<<")
    print(f"  平均每筆 P&L(已扣費):{avg_pnl:+.4f}%")
    print(f"  累計 P&L(已扣費):    {total_pnl:+.4f}%")

    # ── 實際資金曲線(用 ledger 真實記錄的 pnl_usd) ─────────────────
    # 舊版 timeout 紀錄沒寫 pnl_usd,從 position_usd × pnl_pct 反推
    def _pnl_usd_of(r: dict) -> float:
        if r.get("pnl_usd") is not None:
            return r["pnl_usd"]
        pu, pp = r.get("position_usd"), r.get("pnl_pct")
        if pu is not None and pp is not None:
            return pu * pp / 100
        return 0.0

    realised_pnl_usd = sum(_pnl_usd_of(r) for r in closes)
    final_equity = config.INITIAL_EQUITY + realised_pnl_usd
    total_return = realised_pnl_usd / config.INITIAL_EQUITY * 100

    print()
    print(f"  實際資金曲線(起始 ${config.INITIAL_EQUITY:,.0f} USD,每筆冒 {config.RISK_PCT*100:.0f}% 風險)")
    print(f"  ─────────────────────────")
    print(f"  起始資金:  ${config.INITIAL_EQUITY:>12,.2f}")
    print(f"  現在資產:  ${final_equity:>12,.2f}")
    print(f"  總損益:    ${realised_pnl_usd:>+12,.2f}  ({total_return:+.2f}%)")

    print()
    print("  交易明細:")
    print(f"  {'日期':<17} {'幣種':<10} {'進場':>10} {'出場':>10} {'P&L':>10}  結果")
    print(f"  {'-'*17} {'-'*10} {'-'*10} {'-'*10} {'-'*10}  ----")
    for r in closes:
        ep      = r.get("exit_price")
        pnl     = r.get("pnl_pct")
        ep_str  = f"{ep:.2f}"   if ep  is not None else "N/A"
        pnl_str = f"{pnl:+.4f}%" if pnl is not None else "N/A"
        mark    = "✅" if r["outcome"] == "win" else ("❌" if r["outcome"] == "loss" else "⚠️")
        print(f"  {r['entry_time'][:16]}  {r['symbol']:<10} "
              f"{r['entry_price']:>10.2f} {ep_str:>10} {pnl_str:>10}  {mark}")

    print("=" * 55)


if __name__ == "__main__":
    main()
