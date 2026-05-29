# Phase 1：資料抓取與時間對齊 — 設計文件

**日期：** 2026-05-29  
**專案：** BYBIT_ML（虛擬貨幣價格趨勢預測系統）  
**階段：** Phase 1 / 5

---

## 目標

從 Bybit 抓取 3-4 年的雙週期（1H + 1D）OHLCV + turnover 歷史資料，
進行清洗、時間對齊（無 look-ahead bias），並持久化為 Parquet 主格式 + Excel 供人工查閱，
同時支援每日增量更新（含重疊覆寫護城河）。

---

## 範圍

- **幣對：** BTCUSDT、ETHUSDT（驗證後再擴充）
- **週期：** 1H（核心）、1D（環境濾網）
- **歷史深度：** 2022-01-01 至今（約 4.5 年，涵蓋完整牛熊週期）
- **儲存格式：** Parquet（主要）+ Excel（人工查閱）

---

## 目錄結構

```
BYBIT_ML/
├── config.py                  # 幣對、時間區間、路徑等全域設定
├── main.py                    # 手動執行入口（歷史全量拉取）
├── scheduler.py               # 每日增量更新入口
│
├── data/
│   ├── __init__.py
│   ├── fetcher.py             # Bybit API 分頁抓取（唯一接觸 API 的地方）
│   ├── cleaner.py             # 去重、補缺、1H↔1D 無洩漏對齊（唯一處理資料品質的地方）
│   └── exporter.py            # Parquet append + Excel 重寫（唯一碰磁碟的地方）
│
├── storage/
│   ├── raw/                   # Parquet 主檔
│   │   ├── BTCUSDT_1h.parquet
│   │   ├── BTCUSDT_1d.parquet
│   │   ├── ETHUSDT_1h.parquet
│   │   └── ETHUSDT_1d.parquet
│   └── excel/
│       ├── BTCUSDT.xlsx       # 含 1H、1D 兩個工作表
│       └── ETHUSDT.xlsx
│
├── tests/
│   ├── test_fetcher.py
│   ├── test_cleaner.py
│   └── test_exporter.py
│
├── requirements.txt
├── .env                       # API 金鑰（不進 git）
└── .gitignore
```

---

## 模組設計

### fetcher.py

**職責：** 唯一知道 Bybit API 的地方，回傳乾淨的原始 DataFrame。

**對外介面：**
```python
def fetch_ohlcv(
    symbol: str,        # "BTCUSDT"
    interval: str,      # "60"（1H）或 "D"（1D）
    start_date: str,    # "2022-01-01"
    end_date: str,      # "2026-05-29"
) -> pd.DataFrame:
    # 回傳欄位：timestamp(UTC), open, high, low, close, volume, turnover
```

**技術細節：**
- 使用 `pybit v5 unified_trading.HTTP`
- Bybit 每次最多回傳 200 根；採用 **Forward Fetching（由舊到新）**：
  從 `start_date` 開始，取最後一根時間戳 + 1ms 作為下次請求的 `start`，直到超過 `end_date`
- 資料天生正序，無需 `sort_index()`
- 每次請求間隔 0.2 秒，避免觸發 rate limit
- API 錯誤自動 retry 3 次（指數退避：1s / 2s / 4s）
- `timestamp` 統一轉為 `datetime64[UTC]`

**回傳資料欄位：**
```
timestamp(UTC)          open      high      low       close     volume    turnover
2024-01-01 00:00:00    42000.0   42500.0   41800.0   42300.0   1234.5    51987150.0
```

> **為何需要 turnover：** BTC 價格 3 年間從 $20k 漲到 $70k，相同的 volume（顆數）代表截然不同的資金規模。turnover（USDT 總價值）才是正確的資金動能基底，對後續特徵工程至關重要。

---

### cleaner.py

**職責：** 唯一處理資料品質的地方，不碰 API、不碰磁碟。

**對外介面：**
```python
def clean(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    # 回傳：去重、補缺、型別正確、timestamp 為 DatetimeTZDtype(UTC) 的 DataFrame

def align_daily_to_hourly(hourly_df: pd.DataFrame, daily_df: pd.DataFrame) -> pd.DataFrame:
    # 將日線特徵貼到小時線，嚴格防止 look-ahead bias
```

**清洗步驟：**

1. **去除重複（Duplicates）**
   ```python
   df = df.drop_duplicates(subset=["timestamp"], keep="last")
   ```
   分頁接縫、重疊抓取都可能產生重複；`keep="last"` 保留最新版本（含已收盤資料）。

2. **補缺失 K 線（Missing Candles）**
   ```python
   full_index = pd.date_range(start, end, freq="1H" or "1D", tz="UTC")
   df = df.set_index("timestamp").reindex(full_index)
   # OHLC：forward fill（延續上根收盤價）
   # volume / turnover：填 0（無交易發生）
   ```

3. **資料品質 log**
   ```
   [BTCUSDT 1H] 總筆數: 26,280 | 補缺: 3 根 | 重複移除: 0
   ```

**雙週期時間對齊（無 look-ahead bias）：**

> **致命陷阱：** 直接用 `timestamp.floor("1D")` 把當天日線特徵貼到當天小時線，
> 會讓模型在早上 8 點「偷看」到當天 23:59 才收盤的日線結果，
> 導致回測勝率虛高，實盤上線後立即崩潰。

**正確做法：**
```python
# 日線特徵向後位移一天，確保只能看到「已收盤的」日線
daily_df["date_available"] = daily_df["timestamp"] + pd.Timedelta(days=1)

# merge_asof 只匹配 date_available <= hourly timestamp 的最新日線
merged = pd.merge_asof(
    hourly_df.sort_values("timestamp"),
    daily_df.sort_values("date_available"),
    left_on="timestamp",
    right_on="date_available",
    direction="backward"
)
```

對齊時間軸示意：
```
日線 2024-01-01（收盤 23:59）→ date_available = 2024-01-02 00:00
貼到 2024-01-02 00:00 ✅  貼到 2024-01-02 12:00 ✅
不貼到 2024-01-01 任何一根 ✅
```

---

### exporter.py

**職責：** 唯一碰磁碟的地方，負責寫 Parquet 與 Excel。

**Parquet 增量 append：**
```python
def append_parquet(path: Path, new_df: pd.DataFrame):
    if path.exists():
        old_df = pd.read_parquet(path)
        # 舊資料放前，新資料放後，keep="last" 保留最新版本
        combined = pd.concat([old_df, new_df])
        combined = combined.drop_duplicates(subset=["timestamp"], keep="last")
        combined = combined.sort_values("timestamp").reset_index(drop=True)
    else:
        combined = new_df
    combined.to_parquet(path, index=False)
```

**Excel 輸出：**
- 每個幣對一個 `.xlsx`，內含 `1H`、`1D` 兩個工作表
- 每次增量更新後整份重寫（避免 openpyxl append 造成格式錯亂）
- 首列 freeze，數字欄位格式化（千分位、小數點）

---

### scheduler.py

**職責：** 每日增量更新，讀取現有資料的最後時間戳，接著抓取缺口。

**增量更新流程：**
```python
OVERLAP_CANDLES = 3  # 往回重疊抓取的根數

def incremental_update(symbol: str, interval: str):
    last_ts = get_last_timestamp(parquet_path)
    overlap = timedelta(hours=3) if interval == "60" else timedelta(days=3)
    start = last_ts - overlap          # 往回推，覆蓋可能未收盤的 K 線
    end = datetime.utcnow()
    new_df = fetcher.fetch_ohlcv(symbol, interval, start, end)
    new_df = cleaner.clean(new_df, interval)
    exporter.append_parquet(parquet_path, new_df)  # 重疊部分自動覆寫
    exporter.write_excel(symbol)
```

> **重疊覆寫護城河：** 若上次排程在 K 線未收盤時執行，
> 會抓到不完整的收盤價並存入 Parquet。
> 下次從 `last_ts - 3根` 重新抓取，`keep="last"` 自動用已收盤的最終價格覆蓋舊記錄。

**執行方式：**
```
python scheduler.py
```
使用 Windows 工作排程器，設定每天 UTC 01:00 執行（確保前一天日線已收盤）。

---

## 資料流總覽

```
Bybit API
   └─ fetcher.py（分頁 forward fetch，含 turnover）
        └─ cleaner.py（去重 → 補缺 → 型別轉換）
             └─ exporter.py
                  ├─ storage/raw/*.parquet（增量 append，重疊覆寫）
                  └─ storage/excel/*.xlsx（整份重寫）
```

---

## 測試策略

| 測試檔案 | 測試重點 |
|---|---|
| `test_fetcher.py` | Mock API 回應，驗證分頁邏輯與 retry |
| `test_cleaner.py` | 補缺、去重、look-ahead bias 對齊正確性 |
| `test_exporter.py` | Parquet append 去重、Excel 工作表結構 |

---

## 依賴套件

```
pybit>=5.0.0
pandas>=2.0.0
pyarrow>=14.0.0       # Parquet 讀寫
openpyxl>=3.1.0       # Excel 輸出
python-dotenv>=1.0.0
pytest>=7.0.0
numpy<2               # 避免 Anaconda 環境 NumPy 2.x 相容性問題
```

---

## 後續階段接口

Phase 2（特徵工程）直接讀取 `storage/raw/*.parquet`，
呼叫 `cleaner.align_daily_to_hourly()` 取得已對齊的雙週期 DataFrame，
在此基礎上計算 RSI、布林通道、未平倉量變化率等特徵。
