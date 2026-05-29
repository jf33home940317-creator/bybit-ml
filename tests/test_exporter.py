# tests/test_exporter.py
import pytest
import pandas as pd
from pathlib import Path


def _make_df(start: str = "2024-01-01", periods: int = 5) -> pd.DataFrame:
    ts = pd.date_range(start=start, periods=periods, freq="1h", tz="UTC")
    return pd.DataFrame({
        "timestamp": ts,
        "open":  [100.0] * periods,
        "high":  [101.0] * periods,
        "low":   [99.0]  * periods,
        "close": [100.5] * periods,
        "volume":   [10.0]   * periods,
        "turnover": [1005.0] * periods,
    })


class TestAppendParquet:
    def test_creates_file_if_not_exists(self, tmp_path):
        """Parquet 不存在時建立新檔"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        append_parquet(path, _make_df())

        assert path.exists()
        assert len(pd.read_parquet(path)) == 5

    def test_appends_non_overlapping_rows(self, tmp_path):
        """無重疊時 append 後總筆數 = 舊 + 新"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        df1 = _make_df("2024-01-01 00:00", periods=5)
        df2 = _make_df("2024-01-01 05:00", periods=5)

        append_parquet(path, df1)
        append_parquet(path, df2)

        assert len(pd.read_parquet(path)) == 10

    def test_overlap_keeps_latest_version(self, tmp_path):
        """重疊 timestamp 保留最新版本（已收盤價格覆蓋舊的未收盤價格）"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        ts = pd.Timestamp("2024-01-01 00:00:00", tz="UTC")

        df_old = pd.DataFrame({
            "timestamp": [ts],
            "open": [100.0], "high": [101.0], "low": [99.0],
            "close": [100.5],      # 未收盤時抓到的臨時價格
            "volume": [5.0], "turnover": [502.5],
        })
        df_new = pd.DataFrame({
            "timestamp": [ts],
            "open": [100.0], "high": [102.0], "low": [98.0],
            "close": [101.5],      # 收盤後的最終價格
            "volume": [10.0], "turnover": [1015.0],
        })

        append_parquet(path, df_old)
        append_parquet(path, df_new)

        result = pd.read_parquet(path)
        assert len(result) == 1
        assert result.iloc[0]["close"] == 101.5

    def test_result_sorted_by_timestamp(self, tmp_path):
        """儲存後 Parquet 按 timestamp 升序排列"""
        from data.exporter import append_parquet

        path = tmp_path / "test.parquet"
        append_parquet(path, _make_df("2024-01-01", periods=5))

        result = pd.read_parquet(path)
        diffs = result["timestamp"].diff().dropna()
        assert (diffs > pd.Timedelta(0)).all()


class TestWriteExcel:
    def test_creates_two_sheets(self, tmp_path):
        """Excel 包含 1H 和 1D 兩個工作表"""
        from data.exporter import write_excel

        path = tmp_path / "BTCUSDT.xlsx"
        write_excel(path, _make_df(periods=10), _make_df(periods=3))

        xl = pd.ExcelFile(path)
        assert "1H" in xl.sheet_names
        assert "1D" in xl.sheet_names

    def test_sheet_row_counts_match_data(self, tmp_path):
        """Excel 各工作表行數與輸入資料一致"""
        from data.exporter import write_excel

        path = tmp_path / "BTCUSDT.xlsx"
        write_excel(path, _make_df(periods=10), _make_df(periods=3))

        assert len(pd.read_excel(path, sheet_name="1H")) == 10
        assert len(pd.read_excel(path, sheet_name="1D")) == 3
