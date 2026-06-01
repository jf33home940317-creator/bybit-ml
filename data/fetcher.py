# data/fetcher.py
import time
import logging
from datetime import datetime, timedelta, timezone

import pandas as pd
from pybit.unified_trading import HTTP

import config

logger = logging.getLogger(__name__)

COLUMNS = ["timestamp", "open", "high", "low", "close", "volume", "turnover"]

_INTERVAL_DELTA = {"60": timedelta(hours=1), "D": timedelta(days=1)}


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
    batch_count = 0

    while current_start < end_dt:
        rows = _fetch_batch(session, symbol, interval, _to_ms(current_start))
        if not rows:
            break

        rows = list(reversed(rows))  # Bybit returns newest first; reverse to chronological
        batch_df = _parse_rows(rows)
        batch_df = batch_df[batch_df["timestamp"] <= end_dt]

        if batch_df.empty:
            break

        all_batches.append(batch_df)
        batch_count += 1
        last_ts = batch_df["timestamp"].iloc[-1].to_pydatetime()
        current_start = last_ts + _INTERVAL_DELTA[interval]

        if batch_count % 10 == 0:
            total_rows = sum(len(b) for b in all_batches)
            logger.info(f"  [{symbol} {interval}] batch {batch_count} | 已取得 {total_rows:,} 筆 | 進度至 {batch_df['timestamp'].iloc[-1].strftime('%Y-%m-%d')}")

        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=COLUMNS)

    return pd.concat(all_batches, ignore_index=True)


FR_COLUMNS = ["timestamp", "funding_rate"]
OI_COLUMNS = ["timestamp", "open_interest"]


def fetch_funding_rate(
    symbol: str,
    start_date: str,
    end_date: str,
) -> pd.DataFrame:
    """Fetch funding-rate history from Bybit and return a chronological DataFrame.

    Bybit's ``get_funding_rate_history`` returns rows newest-first and accepts an
    ``endTime`` cursor.  We paginate backward from end_date until we pass start_date.
    """
    session = HTTP(testnet=False, api_key=config.API_KEY, api_secret=config.API_SECRET)
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches: list[pd.DataFrame] = []
    current_end_ms = _to_ms(end_dt)

    while True:
        rows = _fetch_fr_batch(session, symbol, current_end_ms)
        if not rows:
            break

        # rows are newest-first; parse and filter
        batch_df = _parse_fr_rows(rows)
        batch_df = batch_df[batch_df["timestamp"] >= start_dt]

        if not batch_df.empty:
            all_batches.append(batch_df)

        # Oldest timestamp in this batch (rows are newest-first, so last after parsing)
        oldest_ts_ms = int(pd.to_numeric(rows[-1]["fundingRateTimestamp"]))
        if oldest_ts_ms <= _to_ms(start_dt):
            break

        # Move cursor to just before the oldest timestamp we received
        current_end_ms = oldest_ts_ms - 1

        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=FR_COLUMNS)

    df = pd.concat(all_batches, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _fetch_fr_batch(session, symbol: str, end_ms: int) -> list:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get_funding_rate_history(
                category="linear",
                symbol=symbol,
                endTime=end_ms,
                limit=200,
            )
            return resp["result"]["list"]
        except Exception as exc:
            if attempt == config.MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                f"FR API error (attempt {attempt + 1}/{config.MAX_RETRIES}), retry in {wait}s: {exc}"
            )
            time.sleep(wait)


def _parse_fr_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                pd.to_numeric([r["fundingRateTimestamp"] for r in rows]), unit="ms", utc=True
            ),
            "funding_rate": pd.to_numeric(
                [r["fundingRate"] for r in rows]
            ).astype("float64"),
        }
    )
    return df


def fetch_open_interest(
    symbol: str,
    start_date: str,
    end_date: str,
    interval_time: str = "1h",
) -> pd.DataFrame:
    """Fetch open-interest data from Bybit REST API and return a chronological DataFrame.

    Uses ``requests.get`` directly (not pybit) because the OI endpoint requires
    ``startTime``/``endTime`` window pagination.
    """
    import requests as _requests  # local import to avoid module-level name clash

    OI_URL = "https://api.bybit.com/v5/market/open-interest"
    start_dt = _parse_date(start_date)
    end_dt = _parse_date(end_date)

    all_batches: list[pd.DataFrame] = []
    start_ms = _to_ms(start_dt)
    current_end_ms = _to_ms(end_dt)
    batch_count = 0

    # Bybit OI API returns newest-first regardless of startTime, so we
    # paginate BACKWARD from endTime (same strategy as fetch_funding_rate).
    while current_end_ms > start_ms:
        rows = _fetch_oi_batch(
            _requests, OI_URL, symbol, start_ms, current_end_ms, interval_time
        )
        if not rows:
            break

        batch_df = _parse_oi_rows(rows)
        batch_df = batch_df[batch_df["timestamp"] >= start_dt]

        if batch_df.empty:
            break

        all_batches.append(batch_df)
        batch_count += 1

        oldest_ts_ms = int(batch_df["timestamp"].min().value) // 1_000_000
        current_end_ms = oldest_ts_ms - 1

        if batch_count % 10 == 0:
            total = sum(len(b) for b in all_batches)
            logger.info(f"  [{symbol} OI] batch {batch_count} | {total:,} records")

        time.sleep(config.RATE_LIMIT_SLEEP)

    if not all_batches:
        return pd.DataFrame(columns=OI_COLUMNS)

    df = pd.concat(all_batches, ignore_index=True)
    df = df.drop_duplicates(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)
    return df


def _fetch_oi_batch(
    requests_mod, url: str, symbol: str, start_ms: int, end_ms: int, interval_time: str
) -> list:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = requests_mod.get(
                url,
                params={
                    "category": "linear",
                    "symbol": symbol,
                    "intervalTime": interval_time,
                    "startTime": start_ms,
                    "endTime": end_ms,
                    "limit": 200,
                },
                timeout=10,
            )
            resp.raise_for_status()
            data = resp.json()
            if data.get("retCode", -1) != 0:
                raise ValueError(f"Bybit OI error: {data}")
            return data["result"]["list"]
        except Exception as exc:
            if attempt == config.MAX_RETRIES - 1:
                raise
            wait = 2 ** attempt
            logger.warning(
                f"OI API error (attempt {attempt + 1}/{config.MAX_RETRIES}), retry in {wait}s: {exc}"
            )
            time.sleep(wait)


def _parse_oi_rows(rows: list) -> pd.DataFrame:
    df = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                pd.to_numeric([r["timestamp"] for r in rows]), unit="ms", utc=True
            ),
            "open_interest": pd.to_numeric(
                [r["openInterest"] for r in rows]
            ).astype("float64"),
        }
    )
    return df


def _fetch_batch(session, symbol: str, interval: str, start_ms: int) -> list:
    for attempt in range(config.MAX_RETRIES):
        try:
            resp = session.get_kline(
                category="spot",
                symbol=symbol,
                interval=interval,
                start=start_ms,
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
