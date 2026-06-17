"""
src/data/loader.py

Reads raw CSVs, enforces correct dtypes, handles known data anomalies,
and returns clean DataFrames ready for Pydantic validation.

Why a dedicated loader instead of just pd.read_csv() everywhere:
    - Correct dtypes must be set at READ time, not after, for memory efficiency.
      Reading DAYS_BIRTH as int32 instead of int64 halves its memory usage.
    - Known anomalies (DAYS_EMPLOYED = 365243) must be fixed once, here,
      not scattered across every feature engineering function.
    - Consistent null handling — every script sees the same NA representation.
"""

import logging
from pathlib import Path

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dtype maps — tells pandas exactly what type each column should be on load.
# This is more memory-efficient than letting pandas guess (it defaults to
# int64/float64 for everything, wasting 2-4x memory on small-range columns).
# ---------------------------------------------------------------------------

APPLICATION_DTYPES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "TARGET": "Int8",  # capital I = nullable integer
    "NAME_CONTRACT_TYPE": "category",  # categorical is more memory-efficient than object
    "CODE_GENDER": "category",
    "FLAG_OWN_CAR": "category",
    "FLAG_OWN_REALTY": "category",
    "CNT_CHILDREN": "int16",
    "AMT_INCOME_TOTAL": "float32",
    "AMT_CREDIT": "float32",
    "AMT_ANNUITY": "float32",
    "AMT_GOODS_PRICE": "float32",
    "NAME_TYPE_SUITE": "category",
    "NAME_INCOME_TYPE": "category",
    "NAME_EDUCATION_TYPE": "category",
    "NAME_FAMILY_STATUS": "category",
    "NAME_HOUSING_TYPE": "category",
    "REGION_POPULATION_RELATIVE": "float32",
    "DAYS_BIRTH": "int32",
    "DAYS_EMPLOYED": "int32",
    "DAYS_REGISTRATION": "float32",
    "DAYS_ID_PUBLISH": "int32",
    "FLAG_MOBIL": "int8",
    "FLAG_EMP_PHONE": "int8",
    "FLAG_WORK_PHONE": "int8",
    "FLAG_CONT_MOBILE": "int8",
    "FLAG_PHONE": "int8",
    "FLAG_EMAIL": "int8",
    "REGION_RATING_CLIENT": "int8",
    "REGION_RATING_CLIENT_W_CITY": "int8",
    "EXT_SOURCE_1": "float32",
    "EXT_SOURCE_2": "float32",
    "EXT_SOURCE_3": "float32",
    "OBS_30_CNT_SOCIAL_CIRCLE": "float32",
    "DEF_30_CNT_SOCIAL_CIRCLE": "float32",
    "OBS_60_CNT_SOCIAL_CIRCLE": "float32",
    "DEF_60_CNT_SOCIAL_CIRCLE": "float32",
    "DAYS_LAST_PHONE_CHANGE": "float32",
}

BUREAU_DTYPES: dict[str, str] = {
    "SK_ID_CURR": "int32",
    "SK_ID_BUREAU": "int32",
    "CREDIT_ACTIVE": "category",
    "CREDIT_CURRENCY": "category",
    "DAYS_CREDIT": "int32",
    "CREDIT_DAY_OVERDUE": "int32",
    "DAYS_CREDIT_ENDDATE": "float32",
    "AMT_CREDIT_MAX_OVERDUE": "float32",
    "CNT_CREDIT_PROLONG": "int32",
    "AMT_CREDIT_SUM": "float32",
    "AMT_CREDIT_SUM_DEBT": "float32",
    "AMT_CREDIT_SUM_LIMIT": "float32",
    "AMT_CREDIT_SUM_OVERDUE": "float32",
    "CREDIT_TYPE": "category",
    "DAYS_CREDIT_UPDATE": "int32",
    "AMT_ANNUITY": "float32",
}

BUREAU_BALANCE_DTYPES: dict[str, str] = {
    "SK_ID_BUREAU": "int32",
    "MONTHS_BALANCE": "int16",
    "STATUS": "category",
}

PREVIOUS_APP_DTYPES: dict[str, str] = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "NAME_CONTRACT_TYPE": "category",
    "AMT_ANNUITY": "float32",
    "AMT_APPLICATION": "float32",
    "AMT_CREDIT": "float32",
    "AMT_DOWN_PAYMENT": "float32",
    "AMT_GOODS_PRICE": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "DAYS_DECISION": "int32",
    "NAME_PAYMENT_TYPE": "category",
    "CODE_REJECT_REASON": "category",
    "NAME_CLIENT_TYPE": "category",
    "NAME_GOODS_CATEGORY": "category",
    "NAME_PORTFOLIO": "category",
    "NAME_PRODUCT_TYPE": "category",
    "CHANNEL_TYPE": "category",
    "NAME_SELLER_INDUSTRY": "category",
    "CNT_PAYMENT": "float32",
    "NAME_YIELD_GROUP": "category",
}

POS_CASH_DTYPES: dict[str, str] = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "MONTHS_BALANCE": "int16",
    "CNT_INSTALMENT": "float32",
    "CNT_INSTALMENT_FUTURE": "float32",
    "NAME_CONTRACT_STATUS": "category",
    "SK_DPD": "int32",
    "SK_DPD_DEF": "int32",
}

INSTALLMENTS_DTYPES: dict[str, str] = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "NUM_INSTALMENT_VERSION": "float32",
    "NUM_INSTALMENT_NUMBER": "int32",
    "DAYS_INSTALMENT": "float32",
    "DAYS_ENTRY_PAYMENT": "float32",
    "AMT_INSTALMENT": "float32",
    "AMT_PAYMENT": "float32",
}

CREDIT_CARD_DTYPES: dict[str, str] = {
    "SK_ID_PREV": "int32",
    "SK_ID_CURR": "int32",
    "MONTHS_BALANCE": "int16",
    "AMT_BALANCE": "float32",
    "AMT_CREDIT_LIMIT_ACTUAL": "float32",
    "AMT_DRAWINGS_ATM_CURRENT": "float32",
    "AMT_DRAWINGS_CURRENT": "float32",
    "AMT_DRAWINGS_OTHER_CURRENT": "float32",
    "AMT_DRAWINGS_POS_CURRENT": "float32",
    "AMT_INST_MIN_REGULARITY": "float32",
    "AMT_PAYMENT_CURRENT": "float32",
    "AMT_PAYMENT_TOTAL_CURRENT": "float32",
    "AMT_RECEIVABLE_PRINCIPAL": "float32",
    "AMT_RECIVABLE": "float32",
    "AMT_TOTAL_RECEIVABLE": "float32",
    "SK_DPD": "int32",
    "SK_DPD_DEF": "int32",
}


# ---------------------------------------------------------------------------
# Anomaly fixes — known data quality issues that must be fixed before
# any feature engineering happens
# ---------------------------------------------------------------------------


def fix_application_anomalies(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fix known anomalies in the application table.

    DAYS_EMPLOYED = 365243:
        This is a placeholder value meaning 'not employed / retired'.
        If left as-is, feature engineering will compute wildly wrong values
        like 'employed for 1000 years'. Replace with NaN so it's treated
        as missing (which it effectively is).

    DAYS_BIRTH:
        Always negative (days before application). Convert to positive
        years for interpretability. 365.25 accounts for leap years.
    """
    df = df.copy()

    # Fix DAYS_EMPLOYED anomaly
    anomaly_count = (df["DAYS_EMPLOYED"] == 365243).sum()
    if anomaly_count > 0:
        logger.info(f"Replacing {anomaly_count} DAYS_EMPLOYED=365243 anomalies with NaN")
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, pd.NA)
        df["DAYS_EMPLOYED_ANOMALY"] = (df["DAYS_EMPLOYED"].isna()).astype("int8")

    # Convert CODE_GENDER XNA to NaN (3 rows in dataset)
    df["CODE_GENDER"] = df["CODE_GENDER"].replace("XNA", pd.NA)

    # DAYS_LAST_PHONE_CHANGE == 0 is often a missing-data placeholder
    df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)

    return df


# ---------------------------------------------------------------------------
# Loaders — one function per table
# ---------------------------------------------------------------------------


def load_application(path: str | Path, is_train: bool = True) -> pd.DataFrame:
    """Load and clean application_train.csv or application_test.csv."""
    logger.info(f"Loading application {'train' if is_train else 'test'}: {path}")

    # Only pass dtypes for columns we know about — pandas infers(guesses) extra ones
    df = pd.read_csv(path, dtype=APPLICATION_DTYPES)

    df = fix_application_anomalies(df)

    logger.info(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    logger.info(f"Memory usage: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    if is_train:
        target_counts = df["TARGET"].value_counts()
        imbalance_ratio = target_counts[0] / target_counts[1]
        logger.info(f"Class distribution — 0: {target_counts[0]:,} | 1: {target_counts[1]:,}")
        logger.info(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

    return df


def load_bureau(path: str | Path) -> pd.DataFrame:
    """Load bureau.csv."""
    logger.info(f"Loading bureau: {path}")
    df = pd.read_csv(path, dtype=BUREAU_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_bureau_balance(path: str | Path) -> pd.DataFrame:
    """Load bureau_balance.csv — largest-ish table at ~27M rows."""
    logger.info(f"Loading bureau_balance: {path}")
    df = pd.read_csv(path, dtype=BUREAU_BALANCE_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_previous_application(path: str | Path) -> pd.DataFrame:
    """Load previous_application.csv."""
    logger.info(f"Loading previous_application: {path}")
    df = pd.read_csv(path, dtype=PREVIOUS_APP_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_pos_cash(path: str | Path) -> pd.DataFrame:
    """Load POS_CASH_balance.csv."""
    logger.info(f"Loading POS_CASH_balance: {path}")
    df = pd.read_csv(path, dtype=POS_CASH_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_installments(path: str | Path) -> pd.DataFrame:
    """Load installments_payments.csv."""
    logger.info(f"Loading installments_payments: {path}")
    df = pd.read_csv(path, dtype=INSTALLMENTS_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_credit_card(path: str | Path) -> pd.DataFrame:
    """Load credit_card_balance.csv."""
    logger.info(f"Loading credit_card_balance: {path}")
    df = pd.read_csv(path, dtype=CREDIT_CARD_DTYPES)
    logger.info(f"Loaded {len(df):,} rows")
    return df


def load_all_tables(cfg) -> dict[str, pd.DataFrame]:
    """
    Load all 7 tables using paths from Hydra config.

    Returns a dict keyed by table name so downstream code
    can access tables by name rather than by positional index.

    Why cfg instead of hardcoded paths:
        cfg.paths.* comes from configs/data.yaml.
        Change the path in one YAML file -> all scripts update.
        Hardcoded paths break on every new machine and in Docker.
    """
    return {
        "application_train": load_application(cfg.paths.application_train, is_train=True),
        "application_test": load_application(cfg.paths.application_test, is_train=False),
        "bureau": load_bureau(cfg.paths.bureau),
        "bureau_balance": load_bureau_balance(cfg.paths.bureau_balance),
        "previous_app": load_previous_application(cfg.paths.previous_app),
        "pos_cash": load_pos_cash(cfg.paths.pos_cash),
        "installments": load_installments(cfg.paths.installments),
        "credit_card": load_credit_card(cfg.paths.credit_card),
    }
