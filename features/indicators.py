# features/indicators.py
import pandas as pd
import pandas_ta as ta


def _find_ppo_col(ppo_df: "pd.DataFrame", prefix: str) -> str:
    matches = [c for c in ppo_df.columns if c.startswith(prefix)]
    if len(matches) != 1:
        raise ValueError(
            f"Expected exactly 1 PPO column with prefix '{prefix}', "
            f"got {matches}. pandas-ta columns: {list(ppo_df.columns)}"
        )
    return matches[0]


def compute(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    df = hourly_df.copy()

    # RSI: [7, 14, 50] — 短/標準/中長，拉開級距避免 rsi_14↔rsi_24 高共線性
    for period in [7, 14, 50]:
        df[f"rsi_{period}"] = ta.rsi(df["close"], length=period)

    # PPO replaces MACD — output is percentage-based, no absolute-value trap
    # pandas-ta column order varies by version; use prefix matching for safety
    ppo = ta.ppo(df["close"], fast=12, slow=26, signal=9)
    ppo_col      = _find_ppo_col(ppo, "PPO_")
    ppo_sig_col  = _find_ppo_col(ppo, "PPOs")
    ppo_hist_col = _find_ppo_col(ppo, "PPOh")
    df["ppo"]        = ppo[ppo_col]
    df["ppo_signal"] = ppo[ppo_sig_col]
    df["ppo_hist"]   = ppo[ppo_hist_col]

    # ATR: [14, 24]
    for period in [14, 24]:
        df[f"atr_{period}"] = ta.atr(df["high"], df["low"], df["close"], length=period)

    # Bollinger Band width = (upper - lower) / middle
    # pandas-ta bbands columns order: BBL, BBM, BBU, BBB, BBP
    for length, std_val in [(20, 2.0), (50, 2.5)]:
        bb = ta.bbands(df["close"], length=length, lower_std=std_val, upper_std=std_val)
        bbl_col = next(c for c in bb.columns if c.startswith("BBL"))
        bbm_col = next(c for c in bb.columns if c.startswith("BBM"))
        bbu_col = next(c for c in bb.columns if c.startswith("BBU"))
        lower, middle, upper = bb[bbl_col], bb[bbm_col], bb[bbu_col]
        df[f"bband_width_{length}"] = (upper - lower) / middle

    # MA Bias = (close - SMA_N) / SMA_N
    for period in [20, 50, 200]:
        sma = ta.sma(df["close"], length=period)
        df[f"ma_bias_{period}"] = (df["close"] - sma) / sma

    # Turnover ratio only — vol_ratio 與 turnover_ratio r=1.00 完全重複，turnover 含價格權重更能反映資金力道
    for period in [12, 24]:
        df[f"turnover_ratio_{period}"] = df["turnover"] / df["turnover"].rolling(period).mean()

    # Daily indicators — computed independently, then merged with look-ahead prevention
    df = _attach_daily_features(df, daily_df)
    return df


def _attach_daily_features(
    hourly_df: pd.DataFrame,
    daily_df: pd.DataFrame,
) -> pd.DataFrame:
    daily = daily_df.copy()

    daily["daily_rsi_14"] = ta.rsi(daily["close"], length=14)
    daily["daily_atr_14"] = ta.atr(daily["high"], daily["low"], daily["close"], length=14)
    for period in [20, 50, 200]:
        sma = ta.sma(daily["close"], length=period)
        daily[f"daily_ma_bias_{period}"] = (daily["close"] - sma) / sma

    # Shift forward by 1 day to prevent look-ahead bias (same logic as Phase 1)
    daily["date_available"] = daily["timestamp"] + pd.Timedelta(days=1)

    feat_cols = [
        "date_available", "daily_rsi_14", "daily_atr_14",
        "daily_ma_bias_20", "daily_ma_bias_50", "daily_ma_bias_200",
    ]
    hourly_sorted = hourly_df.sort_values("timestamp").copy()
    daily_sorted = daily[feat_cols].sort_values("date_available").copy()
    # Normalize precision: Parquet reads as ms, in-memory ops may produce us
    hourly_sorted["timestamp"] = hourly_sorted["timestamp"].astype("datetime64[us, UTC]")
    daily_sorted["date_available"] = daily_sorted["date_available"].astype("datetime64[us, UTC]")

    merged = pd.merge_asof(
        hourly_sorted,
        daily_sorted,
        left_on="timestamp",
        right_on="date_available",
        direction="backward",
    )
    return merged.drop(columns=["date_available"], errors="ignore")
