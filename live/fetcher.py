import requests
import pandas as pd

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


def fetch_latest(symbol: str, interval: str, n: int = 300) -> pd.DataFrame:
    """
    Fetch n most recent candles from Bybit V5 public endpoint (no API key needed).
    Returns chronological DataFrame with UTC-aware timestamps.
    """
    resp = requests.get(
        BYBIT_KLINE_URL,
        params={"symbol": symbol, "interval": interval, "limit": n},
        timeout=10,
    )
    resp.raise_for_status()
    rows = resp.json()["result"]["list"]   # Bybit returns newest first
    rows = list(reversed(rows))            # convert to chronological
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df
