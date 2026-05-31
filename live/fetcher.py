import logging
import time
import requests
import pandas as pd

logger = logging.getLogger(__name__)

BYBIT_KLINE_URL = "https://api.bybit.com/v5/market/kline"
_COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]

MAX_RETRIES = 3
_BACKOFF_BASE_SEC = 1.0


def _sleep(seconds: float) -> None:
    """Indirection so tests can patch it without sleeping."""
    time.sleep(seconds)


def fetch_latest(symbol: str, interval: str, n: int = 300) -> pd.DataFrame:
    """
    Fetch n most recent candles from Bybit V5 public endpoint (no API key needed).
    Returns chronological DataFrame with UTC-aware timestamps.

    Retries transient network failures (requests exceptions) with exponential
    backoff; HTTP-4xx responses and Bybit business errors are NOT retried.
    """
    body = None
    last_exc: Exception = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = requests.get(
                BYBIT_KLINE_URL,
                params={"symbol": symbol, "interval": interval, "limit": n},
                timeout=10,
            )
            resp.raise_for_status()
            body = resp.json()
            break
        except requests.exceptions.RequestException as exc:
            last_exc = exc
            if attempt == MAX_RETRIES - 1:
                logger.error(f"fetch_latest {symbol} {interval}: gave up after {MAX_RETRIES} attempts: {exc}")
                raise
            wait = _BACKOFF_BASE_SEC * (2 ** attempt)
            logger.warning(f"fetch_latest {symbol} {interval}: attempt {attempt + 1}/{MAX_RETRIES} failed ({exc}); retry in {wait}s")
            _sleep(wait)

    if body.get("retCode", 0) != 0:
        raise ValueError(
            f"Bybit API error {body.get('retCode')}: {body.get('retMsg')}"
        )
    rows = body["result"]["list"]   # Bybit returns newest first
    rows = list(reversed(rows))            # convert to chronological
    df = pd.DataFrame(rows, columns=_COLUMNS)
    df["timestamp"] = pd.to_datetime(df["timestamp"].astype("int64"), unit="ms", utc=True)
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = df[col].astype(float)
    return df
