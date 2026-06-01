# Phase 6: Funding Rate + Open Interest Features

## Goal

Add 6 new features derived from Bybit perpetual contract market data (Funding Rate
and Open Interest) to the existing 30+ feature set. Retrain XGBoost models and deploy
only if backtest Sharpe ratio improves or holds. Otherwise auto-rollback to current models.

## Decision Log

- Scope: **FR + OI only** (no multi-timeframe confluence or adaptive threshold)
- Data range: **2022-01-01 to present** (aligned with existing K-line history)
- Safety net: **Auto-rollback** if new Sharpe < old Sharpe

## New Features (6)

| Feature | Computation | Intuition |
|---|---|---|
| `funding_rate` | Raw 8h funding rate, forward-filled to hourly | Direct long/short pressure indicator |
| `funding_rate_ma_24` | 24h rolling mean of funding_rate | Smooth short-term noise |
| `funding_zscore_30d` | (FR - 30d_mean) / 30d_std | Extreme FR = reversal signal |
| `oi_change_1h` | OI hourly pct_change | New money entering/leaving |
| `oi_change_24h` | OI 24h pct_change | Medium-term fund flow trend |
| `oi_price_divergence` | (sign(oi_change_24h) * sign(roc_24) < 0).astype(int) | Price-volume divergence = reversal risk |

## Data Sources

| Data | Bybit V5 Endpoint | Frequency | Pagination |
|---|---|---|---|
| Funding Rate | `GET /v5/market/funding/history` | Every 8h (3x/day) | limit=200, decrement endTime in while loop |
| Open Interest | `GET /v5/market/open-interest` | 1h snapshots | limit=200, use cursor pagination |

## Look-Ahead Prevention

- **Funding Rate**: `fundingRateTimestamp` is the settlement time. Use `merge_asof(direction="backward")`
  so each hourly bar only sees the most recently settled FR. Forward-fill between settlements.
- **Open Interest**: Timestamp-aligned with hourly bars. Same as K-line data — no shift needed.
- **Both**: DataFrames normalized to `datetime64[us, UTC]` and sorted before merge_asof (existing pattern).

## Live Pipeline Lookback Requirements

Live fetching needs enough history to compute rolling features:

| Data | Feature Requiring Most History | Lookback Needed |
|---|---|---|
| Funding Rate | `funding_zscore_30d` (30d * 3/day = 90 records) | FR_LOOKBACK = 100 |
| Open Interest | `oi_change_24h` (24 hourly records) | OI_LOOKBACK = 30 |

These constants go in `live/pipeline.py` alongside existing `HOURLY_LOOKBACK = 300`.

## Implementation Flow

### Batch 1: Data + Feature Engineering

1. `data/fetcher.py`: Add `fetch_funding_rate(symbol, start, end)` and `fetch_open_interest(symbol, start, end)`.
   - Both use while-loop pagination (same pattern as `fetch_ohlcv`).
   - Store as `storage/raw/{symbol}_funding_rate.parquet` and `{symbol}_open_interest.parquet`.

2. `main.py`: Add FR/OI fetching after K-line fetching.

3. `features/indicators.py`: Add `_attach_funding_features(df, fr_df)` and `_attach_oi_features(df, oi_df)`.
   - FR: merge_asof backward, then forward-fill, then compute ma_24, zscore_30d.
   - OI: merge on timestamp, then compute pct_change(1), pct_change(24), divergence.

4. `features/builder.py`: Load FR/OI parquets and pass to `indicators.compute()`.

5. `build_features.py`: Rerun. Validate no unexpected NaN in new columns (beyond warmup period).

### Batch 2: Retrain + A/B Backtest

6. Backup current models: `storage/models/` -> `storage/models_v1/`.

7. `train_models.py`: Rerun (picks up new features automatically via validation_report.json).

8. `run_backtest.py` + `run_portfolio_backtest.py`: Rerun for ETHUSDT target_atr.

9. Compare: if new `sharpe_ratio >= old_sharpe_ratio`, keep new models. Otherwise copy
   `storage/models_v1/` back to `storage/models/` and revert `storage/features/` validation reports.

### Batch 3: Live Deployment

10. `live/fetcher.py`: Add `fetch_funding_rate_latest(symbol, n)` and `fetch_oi_latest(symbol, n)`.
    - With retry (existing pattern).

11. `live/pipeline.py`:
    - `compute_signal()` fetches FR + OI alongside hourly/daily K-lines.
    - Passes FR/OI DataFrames to `indicators.compute()`.
    - New constants: `FR_LOOKBACK = 100`, `OI_LOOKBACK = 30`.

12. Deploy to VM. Verify heartbeat runs without error.

## Rollback Procedure

```
storage/
  models_v1/          # backup of current (known-good) models
  models/             # new models (may or may not be better)
  features_v1/        # backup of current validation reports
  features/           # new validation reports (with FR/OI columns)
```

If new Sharpe < old Sharpe:
1. Copy `models_v1/*.pkl` back to `models/`
2. Copy `features_v1/*_validation_report.json` back to `features/`
3. Do NOT deploy Batch 3 changes to live/pipeline.py
4. Feature columns in validation_report.json drive which features the model expects —
   reverting the report automatically excludes FR/OI from inference.

## Testing

- Unit tests for new fetcher functions (mock API responses)
- Unit tests for new indicator computations (known input -> expected output)
- Integration: `build_features.py` produces valid parquet with no unexpected NaN
- Backtest comparison: automated Sharpe comparison script

## Config Changes

```python
# config.py — no new constants needed for Phase 6
# (FR_LOOKBACK and OI_LOOKBACK live in live/pipeline.py alongside HOURLY_LOOKBACK)
```

## Risk Assessment

- **Low risk**: FR/OI are additive features. If they're noise, XGBoost will assign them zero importance
  and Sharpe stays flat. Auto-rollback catches any regression.
- **Medium risk**: FR/OI history before 2022 may be incomplete on Bybit. Mitigated by starting at 2022-01-01.
- **No risk to live system**: Batch 3 only deploys if Batch 2 passes the Sharpe gate.
