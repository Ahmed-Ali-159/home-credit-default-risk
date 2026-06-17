# tests/test_loader.py

"""
tests/test_loader.py

Minimal smoke test for the most expensive silent bug in this dataset:
the DAYS_EMPLOYED=365243 sentinel that creates "employed for 1000 years"
features if left unfixed.
"""

import pandas as pd

from src.data.loader import fix_application_anomalies


class TestFixApplicationAnomalies:
    def test_days_employed_sentinel_replaced(self):
        df = pd.DataFrame(
            {
                "DAYS_EMPLOYED": [-1000, 365243],
                "DAYS_LAST_PHONE_CHANGE": [-100, -200],
                "CODE_GENDER": ["M", "F"],
            }
        )
        result = fix_application_anomalies(df)
        assert result["DAYS_EMPLOYED"].isna().sum() == 1
        assert "DAYS_EMPLOYED_ANOMALY" in result.columns
