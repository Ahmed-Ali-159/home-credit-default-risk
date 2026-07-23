# api/predictor.py

"""
Applies the SAME preprocessing as src/features/application.py to a single
incoming request, then predicts. Must stay in sync with that file — any
divergence is training-serving skew.
"""

import logging
import math
import re

import numpy as np
import pandas as pd

from api.model_loader import ModelArtifacts
from api.schemas import ExplainResponse, LoanApplicationRequest, PredictionResponse, ShapFeature
from src.models.explain import explain_single
from src.models.narrate import generate_narrative

logger = logging.getLogger(__name__)

HIGH_CARD_COLS = ["ORGANIZATION_TYPE", "OCCUPATION_TYPE"]

# Mirrors application.py's LabelEncoder step for binary string columns.
# LabelEncoder fits alphabetically, so the mapping is deterministic.
_BINARY_MAPS = {
    "CODE_GENDER": {"F": 0, "M": 1, "XNA": 0},
    "FLAG_OWN_CAR": {"N": 0, "Y": 1},
    "FLAG_OWN_REALTY": {"N": 0, "Y": 1},
    "NAME_CONTRACT_TYPE": {"Cash loans": 0, "Revolving loans": 1},
    "EMERGENCYSTATE_MODE": {"No": 0, "Yes": 1},
}

# One-hot columns produced by application.py's encode_categoricals()
# after clean_column_names(). Order doesn't matter — _align handles it.
_OHE_COLUMNS = {
    "NAME_INCOME_TYPE": [
        "NAME_INCOME_TYPE_Working",
    ],
    "NAME_EDUCATION_TYPE": [
        "NAME_EDUCATION_TYPE_Academic_degree",
        "NAME_EDUCATION_TYPE_Higher_education",
        "NAME_EDUCATION_TYPE_Incomplete_higher",
        "NAME_EDUCATION_TYPE_Lower_secondary",
        "NAME_EDUCATION_TYPE_Secondary_secondary_special",
    ],
    "NAME_FAMILY_STATUS": [
        "NAME_FAMILY_STATUS_Civil_marriage",
        "NAME_FAMILY_STATUS_Married",
        "NAME_FAMILY_STATUS_Separated",
        "NAME_FAMILY_STATUS_Single_not_married",
        "NAME_FAMILY_STATUS_Widow",
    ],
    "NAME_HOUSING_TYPE": [
        "NAME_HOUSING_TYPE_Co_op_apartment",
        "NAME_HOUSING_TYPE_House_apartment",
        "NAME_HOUSING_TYPE_Municipal_apartment",
        "NAME_HOUSING_TYPE_Office_apartment",
        "NAME_HOUSING_TYPE_Rented_apartment",
        "NAME_HOUSING_TYPE_With_parents",
    ],
    "WEEKDAY_APPR_PROCESS_START": [
        "WEEKDAY_APPR_PROCESS_START_FRIDAY",
        "WEEKDAY_APPR_PROCESS_START_MONDAY",
        "WEEKDAY_APPR_PROCESS_START_SATURDAY",
        "WEEKDAY_APPR_PROCESS_START_SUNDAY",
        "WEEKDAY_APPR_PROCESS_START_THURSDAY",
        "WEEKDAY_APPR_PROCESS_START_TUESDAY",
        "WEEKDAY_APPR_PROCESS_START_WEDNESDAY",
    ],
    "FONDKAPREMONT_MODE": [
        "FONDKAPREMONT_MODE_not_specified",
        "FONDKAPREMONT_MODE_org_spec_account",
        "FONDKAPREMONT_MODE_reg_oper_account",
        "FONDKAPREMONT_MODE_reg_oper_spec_account",
    ],
    "HOUSETYPE_MODE": [
        "HOUSETYPE_MODE_block_of_flats",
        "HOUSETYPE_MODE_specific_housing",
        "HOUSETYPE_MODE_terraced_house",
    ],
    "WALLSMATERIAL_MODE": [
        "WALLSMATERIAL_MODE_Block",
        "WALLSMATERIAL_MODE_Mixed",
        "WALLSMATERIAL_MODE_Monolithic",
        "WALLSMATERIAL_MODE_Others",
        "WALLSMATERIAL_MODE_Panel",
        "WALLSMATERIAL_MODE_Stone_brick",
        "WALLSMATERIAL_MODE_Wooden",
    ],
}


def _sigmoid(x: float) -> float:
    return 1 / (1 + math.exp(-x))


def _clean_col_name(name: str) -> str:
    """Mirrors application.py's clean_column_names() logic."""
    new = re.sub(r"[^A-Za-z0-9_]", "_", name)
    new = re.sub(r"_+", "_", new)
    return new.strip("_")


def _engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Mirrors application.py's engineer_features() — keep these in sync."""
    df = df.copy()

    # DAYS_EMPLOYED anomaly fix (mirrors loader.py + predictor contract)
    if "DAYS_EMPLOYED" in df.columns:
        df["DAYS_EMPLOYED"] = df["DAYS_EMPLOYED"].replace(365243, np.nan)
        df["DAYS_EMPLOYED_ANOMALY"] = df["DAYS_EMPLOYED"].isna().astype(int)
        df["EMPLOYED_YEARS"] = -df["DAYS_EMPLOYED"] / 365

    if "DAYS_LAST_PHONE_CHANGE" in df.columns:
        df["DAYS_LAST_PHONE_CHANGE"] = df["DAYS_LAST_PHONE_CHANGE"].replace(0, np.nan)

    df["AGE_YEARS"] = -df["DAYS_BIRTH"] / 365

    df["CREDIT_INCOME_RATIO"] = df["AMT_CREDIT"] / df["AMT_INCOME_TOTAL"].clip(lower=1)
    df["ANNUITY_INCOME_RATIO"] = df["AMT_ANNUITY"] / (df["AMT_INCOME_TOTAL"] / 12).clip(lower=1)

    if "AMT_GOODS_PRICE" in df.columns:
        df["CREDIT_GOODS_RATIO"] = df["AMT_CREDIT"] / df["AMT_GOODS_PRICE"].clip(lower=1)
        df["ANNUITY_CREDIT_RATIO"] = df["AMT_ANNUITY"] / df["AMT_CREDIT"].clip(lower=1)

    if "CNT_FAM_MEMBERS" in df.columns:
        df["INCOME_PER_PERSON"] = df["AMT_INCOME_TOTAL"] / df["CNT_FAM_MEMBERS"].clip(lower=1)

    ext_cols = [c for c in ["EXT_SOURCE_1", "EXT_SOURCE_2", "EXT_SOURCE_3"] if c in df.columns]
    if ext_cols:
        df["EXT_SOURCE_MEAN"] = df[ext_cols].mean(axis=1)
        df["EXT_SOURCE_MIN"] = df[ext_cols].min(axis=1)
        if len(ext_cols) == 3:
            df["EXT_SOURCE_PROD"] = df["EXT_SOURCE_1"] * df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]
            df["EXT_SOURCE_2_3"] = df["EXT_SOURCE_2"] * df["EXT_SOURCE_3"]

    # DOCS_PROVIDED — use provided FLAG_DOCUMENT cols if present
    doc_cols = [c for c in df.columns if "FLAG_DOCUMENT" in c]
    df["DOCS_PROVIDED"] = df[doc_cols].sum(axis=1) if doc_cols else 0

    # Social circle rates — now provided in full demo payload
    if all(c in df.columns for c in ["DEF_30_CNT_SOCIAL_CIRCLE", "OBS_30_CNT_SOCIAL_CIRCLE"]):
        df["DEF_30_RATE"] = df["DEF_30_CNT_SOCIAL_CIRCLE"] / df["OBS_30_CNT_SOCIAL_CIRCLE"].clip(
            lower=1
        )
    else:
        df["DEF_30_RATE"] = np.nan

    if all(c in df.columns for c in ["DEF_60_CNT_SOCIAL_CIRCLE", "OBS_60_CNT_SOCIAL_CIRCLE"]):
        df["DEF_60_RATE"] = df["DEF_60_CNT_SOCIAL_CIRCLE"] / df["OBS_60_CNT_SOCIAL_CIRCLE"].clip(
            lower=1
        )
    else:
        df["DEF_60_RATE"] = np.nan

    return df


def _encode_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode binary string columns — mirrors application.py LabelEncoder (alphabetical fit)."""
    df = df.copy()
    for col, mapping in _BINARY_MAPS.items():
        if col in df.columns:
            df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def _encode_ohe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Reproduce pd.get_dummies + clean_column_names for low-cardinality cols.
    Each source column is replaced by its dummy columns then dropped.
    Unknown category values produce all-zero dummies (same as unseen at train time).
    """
    df = df.copy()
    for src_col, dummy_cols in _OHE_COLUMNS.items():
        if src_col not in df.columns:
            for dummy in dummy_cols:
                df[dummy] = 0
            continue

        value = str(df[src_col].iloc[0]) if pd.notna(df[src_col].iloc[0]) else ""
        dummy_name = _clean_col_name(f"{src_col}_{value}")

        for dummy in dummy_cols:
            df[dummy] = int(dummy == dummy_name)

        df = df.drop(columns=[src_col])

    return df


def _encode_categoricals(df: pd.DataFrame, encoders: dict) -> pd.DataFrame:
    """Apply target-encoding maps saved during training. Unknown categories -> global mean."""
    df = df.copy()
    for col, enc in encoders.items():
        if col in df.columns:
            df[col] = df[col].map(enc["map"]).fillna(enc["global_mean"])
    return df


def _align_to_training_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Missing features (bureau, POS, etc.) -> NaN. LightGBM handles NaN natively."""
    for col in feature_cols:
        if col not in df.columns:
            df[col] = np.nan
    return df[feature_cols]


def _preprocess(request: LoanApplicationRequest, artifacts: ModelArtifacts) -> pd.DataFrame:
    df = pd.DataFrame([request.model_dump()])
    df = _engineer_features(df)
    df = _encode_binary_flags(df)
    df = _encode_ohe(df)
    df = _encode_categoricals(df, artifacts.encoders)
    return _align_to_training_features(df, artifacts.feature_cols)


def _risk_tier(probability: float) -> str:
    if probability < 0.05:
        return "LOW"
    if probability < 0.15:
        return "MEDIUM"
    if probability < 0.30:
        return "HIGH"
    return "VERY_HIGH"


def predict(request: LoanApplicationRequest, artifacts: ModelArtifacts) -> PredictionResponse:
    x_input = _preprocess(request, artifacts)
    probability = float(artifacts.model.predict_proba(x_input.values)[0, 1])
    return PredictionResponse.from_probability(probability, artifacts.model_version)


def predict_with_explanation(
    request: LoanApplicationRequest, artifacts: ModelArtifacts, n_top: int = 10
) -> ExplainResponse:
    if artifacts.explainer is None:
        raise ValueError("SHAP explainer not loaded. Run dvc repro train to generate it.")

    x_input = _preprocess(request, artifacts)
    probability = float(artifacts.model.predict_proba(x_input.values)[0, 1])
    top_features = explain_single(
        artifacts.explainer, x_input.values, artifacts.feature_cols, n_top=n_top
    )

    baseline = artifacts.explainer.expected_value
    if isinstance(baseline, list):
        baseline = baseline[1]
    baseline_prob = round(_sigmoid(float(baseline)), 6)

    narrative = generate_narrative(
        default_probability=probability,
        risk_tier=_risk_tier(probability),
        baseline_probability=baseline_prob,
        top_features=top_features,
    )

    return ExplainResponse(
        default_probability=round(probability, 6),
        risk_tier=_risk_tier(probability),
        model_version=artifacts.model_version,
        baseline_probability=baseline_prob,
        top_features=[ShapFeature(**f) for f in top_features],
        narrative_explanation=narrative,
    )
