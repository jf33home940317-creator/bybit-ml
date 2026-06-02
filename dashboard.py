"""BYBIT_ML Streamlit Dashboard — real-time monitoring for the live signal daemon."""
import json
import requests
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from pathlib import Path
from datetime import datetime, timezone

# ── Config ───────────────────────────────────────────────────────────────────
STORAGE_LIVE = Path(__file__).parent / "storage" / "live"
STORAGE_BACKTEST = Path(__file__).parent / "storage" / "backtest"
PROB_CSV = STORAGE_LIVE / "prob_history.csv"
LEDGER_FILE = STORAGE_LIVE / "paper_trading_ledger.json"
INITIAL_EQUITY = 1_000_000.0

st.set_page_config(page_title="BYBIT_ML Dashboard", page_icon="📈", layout="wide")


# ── Data loaders ─────────────────────────────────────────────────────────────
@st.cache_data(ttl=300)
def load_kline(symbol: str = "ETHUSDT", interval: str = "60", limit: int = 168):
    """Fetch recent K-line data from Bybit V5 (168 bars = 7 days of 1h)."""
    try:
        resp = requests.get(
            "https://api.bybit.com/v5/market/kline",
            params={"symbol": symbol, "interval": interval, "limit": limit},
            timeout=10,
        )
        body = resp.json()
        if body.get("retCode", 0) != 0:
            return pd.DataFrame()
        rows = list(reversed(body["result"]["list"]))
        df = pd.DataFrame(rows, columns=["timestamp", "open", "high", "low", "close", "volume", "turnover"])
        df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
        for col in ["open", "high", "low", "close", "volume"]:
            df[col] = df[col].astype(float)
        return df
    except Exception:
        return pd.DataFrame()


@st.cache_data(ttl=60)
def load_prob_history():
    if not PROB_CSV.exists():
        return pd.DataFrame(columns=["timestamp", "symbol", "probability", "signal", "close"])
    df = pd.read_csv(PROB_CSV, parse_dates=["timestamp"])
    df = df.drop_duplicates(subset=["timestamp", "symbol"], keep="last")
    return df.sort_values("timestamp").reset_index(drop=True)


@st.cache_data(ttl=60)
def load_ledger():
    if not LEDGER_FILE.exists():
        return []
    return json.loads(LEDGER_FILE.read_text(encoding="utf-8"))


# ── Sidebar ──────────────────────────────────────────────────────────────────
st.sidebar.title("BYBIT_ML")
page = st.sidebar.radio("", ["概覽", "交易紀錄", "資金曲線", "Prob 分析", "系統狀態"])
st.sidebar.markdown("---")
st.sidebar.caption(f"Auto-refresh: 60s")


# ── Page: 概覽 ───────────────────────────────────────────────────────────────
if page == "概覽":
    st.title("📈 BYBIT_ML Dashboard")

    prob_df = load_prob_history()
    records = load_ledger()

    opens = [r for r in records if r.get("status") == "open"]
    shadows = [r for r in records if r.get("status") == "shadow"]
    closes = [r for r in records if "outcome" in r and r.get("status") != "shadow_closed"]
    active = [r for r in records if r.get("status") == "open" and "outcome" not in r]

    # Compute equity
    realised_pnl = sum(
        r.get("pnl_usd", 0) or 0 for r in records
        if "outcome" in r and r.get("status") != "shadow_closed"
    )
    equity = INITIAL_EQUITY + realised_pnl

    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    latest_prob = prob_df["probability"].iloc[-1] if not prob_df.empty else 0
    col1.metric("最新 Prob", f"{latest_prob:.4f}")
    col2.metric("帳戶淨值", f"${equity:,.0f}")
    col3.metric("持倉中", f"{len(active)} 筆")

    if not prob_df.empty:
        hours_running = (pd.Timestamp.now("UTC") - prob_df["timestamp"].iloc[0]).total_seconds() / 3600
        col4.metric("已運行", f"{hours_running:.0f}h")
    else:
        col4.metric("已運行", "0h")

    st.markdown("---")

    # Prob chart
    if not prob_df.empty:
        st.subheader("Prob 趨勢")
        chart_df = prob_df[["timestamp", "probability"]].set_index("timestamp")
        chart_df["0.75 門檻"] = 0.75
        chart_df["0.70 影子"] = 0.70
        st.line_chart(chart_df, height=350)

    # K-line candlestick chart
    st.subheader("ETHUSDT K 線圖（1h）")
    kline_df = load_kline()
    if not kline_df.empty:
        fig = go.Figure(data=[go.Candlestick(
            x=kline_df["timestamp"],
            open=kline_df["open"],
            high=kline_df["high"],
            low=kline_df["low"],
            close=kline_df["close"],
            increasing_line_color="#26a69a",
            decreasing_line_color="#ef5350",
        )])
        fig.update_layout(
            height=400,
            margin=dict(l=0, r=0, t=10, b=0),
            xaxis_rangeslider_visible=False,
            yaxis_title="USD",
            xaxis_title="",
        )

        # Overlay prob as secondary y-axis if data available
        if not prob_df.empty:
            fig.add_trace(go.Scatter(
                x=prob_df["timestamp"],
                y=prob_df["probability"],
                name="Prob",
                yaxis="y2",
                line=dict(color="rgba(255,165,0,0.7)", width=1.5),
            ))
            fig.add_hline(y=0.75, line_dash="dash", line_color="red",
                          annotation_text="0.75", yref="y2")
            fig.add_hline(y=0.70, line_dash="dot", line_color="orange",
                          annotation_text="0.70 shadow", yref="y2")
            fig.update_layout(
                yaxis2=dict(
                    title="Prob",
                    overlaying="y",
                    side="right",
                    range=[0.4, 1.0],
                ),
            )

        st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("K 線資料載入失敗")


# ── Page: 交易紀錄 ───────────────────────────────────────────────────────────
elif page == "交易紀錄":
    st.title("📋 交易紀錄")
    records = load_ledger()

    # Real trades
    closes = [r for r in records if "outcome" in r and r.get("status") != "shadow_closed"]
    if closes:
        st.subheader(f"正式訊號（>= 0.75）— {len(closes)} 筆")
        wins = sum(1 for r in closes if r.get("outcome") == "win")
        losses = sum(1 for r in closes if r.get("outcome") == "loss")
        st.markdown(f"**勝率: {wins/(wins+losses)*100:.1f}%** ({wins} 贏 / {losses} 輸)" if wins + losses > 0 else "")

        df = pd.DataFrame(closes)
        display_cols = ["entry_time", "symbol", "entry_price", "exit_price", "pnl_pct", "outcome"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)
    else:
        st.info("尚無正式訊號觸發")

    # Shadow trades
    shadow_closes = [r for r in records if r.get("status") == "shadow_closed"]
    if shadow_closes:
        st.subheader(f"影子訊號（0.70-0.74）— {len(shadow_closes)} 筆")
        s_wins = sum(1 for r in shadow_closes if r.get("outcome") == "win")
        s_losses = sum(1 for r in shadow_closes if r.get("outcome") == "loss")
        st.markdown(f"**勝率: {s_wins/(s_wins+s_losses)*100:.1f}%** ({s_wins} 贏 / {s_losses} 輸)" if s_wins + s_losses > 0 else "")

        df = pd.DataFrame(shadow_closes)
        display_cols = ["entry_time", "symbol", "entry_price", "exit_price", "pnl_pct", "outcome"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

    # Pending shadows
    pending_shadows = [r for r in records if r.get("status") == "shadow" and "outcome" not in r]
    if pending_shadows:
        st.subheader(f"等待結算的影子訊號 — {len(pending_shadows)} 筆")
        df = pd.DataFrame(pending_shadows)
        display_cols = ["entry_time", "symbol", "entry_price", "probability", "sl_price", "tp_price", "exit_time"]
        available = [c for c in display_cols if c in df.columns]
        st.dataframe(df[available], use_container_width=True)

    if not closes and not shadow_closes and not pending_shadows:
        st.info("尚無任何交易紀錄。等 prob >= 0.70 觸發後就會有。")


# ── Page: 資金曲線 ───────────────────────────────────────────────────────────
elif page == "資金曲線":
    st.title("💰 資金曲線")
    records = load_ledger()
    closes = [
        r for r in records
        if "outcome" in r and r.get("status") != "shadow_closed"
        and r.get("pnl_usd") is not None
    ]

    if closes:
        equity_points = [{"time": "start", "equity": INITIAL_EQUITY}]
        eq = INITIAL_EQUITY
        for r in closes:
            eq += r["pnl_usd"]
            equity_points.append({
                "time": r.get("exit_time_actual", r.get("entry_time", "")),
                "equity": eq,
            })

        eq_df = pd.DataFrame(equity_points)
        eq_df["time"] = pd.to_datetime(eq_df["time"], errors="coerce")
        eq_df = eq_df.dropna(subset=["time"]).set_index("time")

        st.line_chart(eq_df["equity"], height=400)

        # MDD
        peak = eq_df["equity"].cummax()
        dd = (eq_df["equity"] - peak) / peak * 100
        st.metric("最大回撤 (MDD)", f"{dd.min():.2f}%")
    else:
        st.info("尚無結算紀錄。資金曲線需要至少 1 筆交易。")


# ── Page: Prob 分析 ──────────────────────────────────────────────────────────
elif page == "Prob 分析":
    st.title("🔬 Prob 分析")
    prob_df = load_prob_history()

    if prob_df.empty:
        st.info("尚無 prob 紀錄。")
    else:
        st.subheader("Prob 分佈")
        st.bar_chart(prob_df["probability"].value_counts(bins=20).sort_index())

        st.subheader("統計")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("筆數", f"{len(prob_df)}")
        col2.metric("平均", f"{prob_df['probability'].mean():.4f}")
        col3.metric("最高", f"{prob_df['probability'].max():.4f}")
        col4.metric("最低", f"{prob_df['probability'].min():.4f}")

        st.subheader("每日最高 Prob")
        prob_df["date"] = prob_df["timestamp"].dt.date
        daily_max = prob_df.groupby("date")["probability"].max().reset_index()
        daily_max.columns = ["日期", "最高 Prob"]
        st.dataframe(daily_max.sort_values("日期", ascending=False), use_container_width=True)


# ── Page: 系統狀態 ───────────────────────────────────────────────────────────
elif page == "系統狀態":
    st.title("🖥️ 系統狀態")

    prob_df = load_prob_history()

    if not prob_df.empty:
        last_heartbeat = prob_df["timestamp"].iloc[-1]
        age = pd.Timestamp.now("UTC") - last_heartbeat
        age_minutes = age.total_seconds() / 60

        if age_minutes < 70:
            st.success(f"✅ Daemon 正常 — 最後心跳 {age_minutes:.0f} 分鐘前")
        elif age_minutes < 130:
            st.warning(f"⚠️ 心跳延遲 — {age_minutes:.0f} 分鐘前（正常 < 65 分鐘）")
        else:
            st.error(f"🚨 心跳中斷 — 已 {age_minutes:.0f} 分鐘未更新！")

        st.metric("最後心跳", last_heartbeat.strftime("%Y-%m-%d %H:%M UTC"))
        st.metric("總心跳次數", f"{len(prob_df)} 次")
    else:
        st.warning("尚無心跳紀錄")

    # Backtest baseline
    st.markdown("---")
    st.subheader("回測基準")
    report_path = STORAGE_BACKTEST / "ETHUSDT_target_atr_portfolio_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text())
        m = report["metrics"]
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Sharpe", f"{m['sharpe_ratio']:.4f}")
        col2.metric("回測報酬", f"{m['total_return_pct']:+.2f}%")
        col3.metric("MDD", f"{m['max_drawdown_pct']:.2f}%")
        col4.metric("勝率", f"{m['win_rate']:.2%}")
