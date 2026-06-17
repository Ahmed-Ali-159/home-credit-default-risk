"""
src/features/application.py

Feature engineering for the main application table.
Extracted from the EDA notebook's preprocess_application() function.

Why this is its own module:
    The application table is the "spine" of the dataset. Every row in the
    feature-engineered output corresponds to one application row.
    All other feature groups (bureau, previous_app, etc.) get aggregated
    and then merged ONTO this table.

Public API:
    preprocess_application(df, high_card_cols) -> pd.DataFrame
"""

import logging
import re

import pandas as pd
from sklearn.preprocessing import LabelEncoder

logger = logging.getLogger(__name__)

# Columns we treat as high cardinality — keep as strings for target encoding
# during model training (target encoding leaks the label if done outside CV)
DEFAULT_HIGH_CARD_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features to the application table.

    These are the features that EDA identified as predictive:
    - Age / employment converted to years
    - Loan-math ratios (debt-to-income, etc.)
    - EXT_SOURCE combinations (strongest individual predictors)
    - Document count rollup
    - Social circle default rates
    """
    df = df.copy()

    # ── Readable age features ─────────────────────────────────
    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365
    df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    # ── Core ratio features (loan math) ──────────────────────
    # Debt-to-income: how large is this loan relative to income?
    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(1)
    # Monthly payment burden: what fraction of monthly income is the payment?
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12).clip(1)
    # Credit relative to goods price: how much is financed vs price?
    df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].clip(1)
    # Annuity relative to credit: higher = shorter loan term
    df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].clip(1)
    # Income per family member
    df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(1)

    # ── EXT_SOURCE composites ────────────────────────────────
    # The 3 strongest individual features — combine them
    ext_cols = ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"]
    df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
    df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
    df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
    df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # ── Document flags — collapse 21 columns into a count ────
    doc_cols = [c for c in df.columns if "FLAG_DOCUMENT" in c]
    df["DOCS_PROVIDED"] = df[doc_cols].sum(axis=1)

    # ── Social circle — normalize by total observations ──────
    df["DEF_30_RATE"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"].clip(1)
    df["DEF_60_RATE"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / df["OBS_60_CNT_SOCIAL_CIRCLE"].clip(1)

    return df


def encode_categoricals(df: pd.DataFrame, high_card_cols: list[str]) -> pd.DataFrame:
    """
    Encode categorical features.

    Strategy by cardinality:
      - 2 unique values    → LabelEncoder (Y/N, M/F → 0/1)
      - high_card_cols     → leave as strings (target-encoded inside CV later)
      - ≤10 unique values  → one-hot encoded with pd.get_dummies
      - everything else    → drop (avoids exploding the feature space)

    Why high_card_cols are NOT encoded here:
        Target encoding must fit on training fold only to prevent leakage.
        If we encoded them here, the encoding would see the test set's
        distribution and we'd get an unrealistic AUC.
    """
    df = df.copy()

    for col in df.select_dtypes(include=["object", "category"]).columns:
        n_unique = df[col].nunique()

        if n_unique == 2:
            # Binary categorical (Y/N, M/F)
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))

        elif col in high_card_cols:
            # High-cardinality but in the safe list — keep as string for target encoding later
            if hasattr(df[col], "cat"):
                df[col] = df[col].astype(str)

            # Keep as string — handled by target encoding during CV
            df[col] = df[col].fillna("Unknown").astype(str)

        elif n_unique <= 10:
            # One-hot encode low-cardinality
            df = pd.get_dummies(df, columns=[col], prefix=col, dummy_na=False)

        else:
            # Drop anything high-cardinality not explicitly handled
            logger.info(
                f"Dropping high-cardinality column not in safe list: {col} ({n_unique} values)"
            )
            df = df.drop(columns=[col])

    return df


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Replace special characters in column names with underscores.

    Why: LightGBM stores feature names in JSON and rejects names with
    special characters like '/', ',', '(', ')', etc. One-hot encoding
    of categories like "Self-employed" produces such names.
    """
    new_cols = {}
    for col in df.columns:
        # Replace any special character with underscore
        new_col = re.sub(r"[^A-Za-z0-9_]", "_", col)
        # Remove consecutive underscores
        new_col = re.sub(r"_+", "_", new_col)
        # Strip leading/trailing underscores
        new_col = new_col.strip("_")
        new_cols[col] = new_col
    return df.rename(columns=new_cols)


def preprocess_application(
    df: pd.DataFrame,
    high_card_cols: list[str] | None = None,
) -> pd.DataFrame:
    """
    Full preprocessing pipeline for the application table.

    This is the public entry point — call this from feature build orchestrator.
    Internally chains: fix_anomalies → engineer_features → encode_categoricals.

    Args:
        df: Raw application_{train|test} DataFrame.
        high_card_cols: Columns to leave as strings for later target encoding.
                        Defaults to ['ORGANIZATION_TYPE', 'OCCUPATION_TYPE'].

    Returns:
        DataFrame with engineered features and encoded categoricals.
    """
    if high_card_cols is None:
        high_card_cols = DEFAULT_HIGH_CARD_COLS

    df = engineer_features(df)
    df = encode_categoricals(df, high_card_cols)
    df = clean_column_names(df)

    return df


def align_train_test(
    train: pd.DataFrame,
    test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Align train and test columns after preprocessing.

    Why this is needed:
        One-hot encoding may produce different dummy columns in train vs test
        if a category appears in one set but not the other. E.g. if
        ORGANIZATION_TYPE='Religion' has 5 rows in train but 0 in test,
        the dummy column 'ORGANIZATION_TYPE_Religion' exists only in train.

        align() with join='left' keeps all train columns and adds missing
        ones to test as zeros.
    """
    train_aligned, test_aligned = train.align(test, join="left", axis=1, fill_value=0)
    # Remove TARGET from test (align added it as 0s)
    test_aligned = test_aligned.drop(columns=["TARGET"], errors="ignore")
    return train_aligned, test_aligned
