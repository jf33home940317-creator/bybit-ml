# data/fetcher.py
import time
import logging
from datetime import datetime, timezone

import pandas as pd
from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]


def fetch_ohlcv(
    symbol: str,
    interval: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    session = HTTP(testnet=False, api_key=config.API_KEY, api_secret=config.API_SECRET)
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches = []
    current_start = start_dt

    while current_start < end_dt:
        rows = _fetch_batch(session, symbol, interval, _to_ms(current_start), _to_ms(end_dt))
        if not rows:
            break

        rows = list(reversed(rows))  # Bybit returns newest first; reverse to chronological
        batch_df = _parse_rows(rows)
        batch_df = batch_df[batch_df["timestamp"] <= end_dt]

        if batch_df.empty:
            break

        all_batches.append(batch_df)
        last_ts_ms = int(batch_df["timestamp"].iloc[-1].timestamp() * 1000)
        current_start = _from_ms(last_ts_ms + 1)
        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(all_batches, ignore_index=True)


def _fetch_batch(session, symbol: str, interval: str, start_ms: int, end_ms: int) -> list:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval,
                start=start_ms,
                end=end_ms,
                limit=200,
            )
            return resp["result"]["list"]
        except Exception as exc:
            if attempt == config.MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(f"API error (attempt {attempt + 1}/{config.MAX_RETRIES}), retry in {wait}s: {exc}")
            time.sleep(wait)


def _parse_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(rows, columns=COLUMNS)
    df["timestamp"] = pd.to_datetime(
        pd.to_numeric(df["timestamp"], downcast=None).astype("int64"), unit="ms", utc=True
    )
    for col in ["open", "high", "low", "close", "volume", "turnover"]:
        df[col] = pd.to_numeric(df[col]).astype("float64")
    return df


def _parse_date(date_str: str) -> datetime:
    dt = datetime.fromisoformat(date_str)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt


def _to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def _from_ms(ms: int) -> datetime:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
