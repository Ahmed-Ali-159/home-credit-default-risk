# tests/test_features.py

"""
tests/test_features.py

Minimal smoke tests for feature engineering modules.

Philosophy: not exhaustive coverage — just enough to catch the
expensive, silent bugs (wrong math, broken aggregation, crashes)
that wouldn't surface until the full pipeline runs on real data.
"""

import pandas as pd
import pytest

from src.features.application import clean_column_names, engineer_features
from src.features.bureau import build_bureau_features
from src.features.cross_table import build_cross_table_features


class TestEngineerFeatures:
    """Catches: wrong ratio math — the most common silent bug type."""

    @pytest.fixture
    def minimal_df(self):
        return pd.DataFrame(
            {
                "DAYS_BIRTH": [-12000],
                "DAYS_EMPLOYED": [-2000],
                "AMT_CREDIT": [200000.0],
                "AMT_INCOME_TOTAL": [100000.0],
                "AMT_ANNUITY": [10000.0],
                "AMT_GOODS_PRICE": [200000.0],
                "CNT_FAM_MEMBERS": [2.0],
                "EXT_SOURCE_1": [0.5],
                "EXT_SOURCE_2": [0.6],
                "EXT_SOURCE_3": [0.4],
                "OBS_30_CNT_SOCIAL_CIRCLE": [5.0],
                "DEF_30_CNT_SOCIAL_CIRCLE": [1.0],
                "OBS_60_CNT_SOCIAL_CIRCLE": [5.0],
                "DEF_60_CNT_SOCIAL_CIRCLE": [1.0],
                "FLAG_DOCUMENT_2": [1],
                "FLAG_DOCUMENT_3": [1],
            }
        )

    def test_credit_income_ratio_correct(self, minimal_df):
        """200000 / 100000 should be exactly 2.0 — catches swapped division order."""
        result = engineer_features(minimal_df)
        assert result["CREDIT_INCOME_RATIO"].iloc[0] == 2.0

    def test_ext_source_mean_correct(self, minimal_df):
        """(0.5+0.6+0.4)/3 = 0.5 — catches wrong aggregation function."""
        result = engineer_features(minimal_df)
        assert result["EXT_SOURCE_MEAN"].iloc[0] == pytest.approx(0.5)


class TestCleanColumnNames:
    """Catches: special characters that crash LightGBM at training time."""

    def test_special_chars_replaced(self):
        df = pd.DataFrame({"a/b": [1], "c (d)": [2]})
        result = clean_column_names(df)
        assert list(result.columns) == ["a_b", "c_d"]


class TestBureauFeatures:
    """Catches: broken groupby producing duplicate or missing client rows."""

    def test_one_row_per_client(self):
        bureau = pd.DataFrame(
            {
                "SK_ID_CURR": [100, 100, 200],
                "SK_ID_BUREAU": [1, 2, 3],
                "CREDIT_ACTIVE": ["Active", "Closed", "Active"],
                "DAYS_CREDIT": [-100, -200, -300],
                "CREDIT_DAY_OVERDUE": [0, 0, 0],
                "AMT_CREDIT_SUM": [1000.0, 2000.0, 3000.0],
                "AMT_CREDIT_SUM_DEBT": [500.0, 0.0, 1500.0],
                "AMT_CREDIT_SUM_OVERDUE": [0.0, 0.0, 0.0],
                "CNT_CREDIT_PROLONG": [0, 0, 0],
            }
        )
        bureau_balance = pd.DataFrame(
            {
                "SK_ID_BUREAU": [1, 2, 3],
                "MONTHS_BALANCE": [-1, -1, -1],
                "STATUS": ["0", "C", "0"],
            }
        )
        result = build_bureau_features(bureau, bureau_balance)
        # 2 unique clients (100, 200) → exactly 2 rows, not 3
        assert len(result) == 2


class TestCrossTableFeatures:
    """Catches: crashes when source columns are missing from upstream merges."""

    def test_handles_missing_columns_gracefully(self):
        df = pd.DataFrame({"SK_ID_CURR": [1, 2], "AGE_YEARS": [30, 40]})
        # Most cross-table source columns are missing — should not crash
        result = build_cross_table_features(df)
        assert "age_x_buro_clean" in result.columns
