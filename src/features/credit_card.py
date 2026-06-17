"""
src/features/credit_card.py

Feature engineering for credit_card_balance.csv.
Returns one row per SK_ID_CURR with the prefix 'cc_'.

Credit card has unique signals not present in other tables:
  - Utilization (balance / credit limit) — closer to 1 = more stressed
  - Min payment vs full payment behavior
  - ATM cash advances — known financial stress signal
  - Balance trend over time (improving vs deteriorating)

Public API:
    build_credit_card_features(credit_card, prev_app) -> pd.DataFrame
"""

import gc
import logging

import pandas as pd

logger = logging.getLogger(__name__)

RECENT_MONTHS = -12
UTIL_MAXED_THRESHOLD = 0.9
UTIL_OVER_LIMIT = 1.0


def add_row_level_features(cc: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-row credit card behavior signals.

    CRITICAL: sort by SK_ID_PREV + MONTHS_BALANCE before any diff() ops,
    otherwise balance_change will be computed on a shuffled time series.
    """
    cc = cc.copy()
    cc = cc.sort_values(["SK_ID_PREV", "MONTHS_BALANCE"])

    # Utilization — capped at 1.5 to limit influence of extreme outliers
    cc["utilization"] = (cc["AMT_BALANCE"] / cc["AMT_CREDIT_LIMIT_ACTUAL"].clip(lower=1)).clip(
        upper=1.5
    )

    # Payment behavior flags
    cc["paid_min"] = (cc["AMT_PAYMENT_TOTAL_CURRENT"] >= cc["AMT_INST_MIN_REGULARITY"]).astype(int)
    cc["paid_full"] = (cc["AMT_PAYMENT_TOTAL_CURRENT"] >= cc["AMT_BALANCE"]).astype(int)

    # ATM cash advance ratio — known financial stress signal
    cc["atm_ratio"] = (
        cc["AMT_DRAWINGS_ATM_CURRENT"] / cc["AMT_DRAWINGS_CURRENT"].clip(lower=1)
    ).fillna(0)

    # Balance trend — requires sorted data above
    cc["balance_change"] = cc.groupby("SK_ID_PREV")["AMT_BALANCE"].diff()

    # Stress flags
    cc["is_maxed"] = (cc["utilization"] > UTIL_MAXED_THRESHOLD).astype(int)
    cc["is_overlimit"] = (cc["utilization"] > UTIL_OVER_LIMIT).astype(int)

    return cc


def build_credit_card_features(
    credit_card: pd.DataFrame,
    prev_app: pd.DataFrame,
) -> pd.DataFrame:
    """Build all credit_card-derived features, one row per SK_ID_CURR."""
    cc = add_row_level_features(credit_card)

    # ── Step 1: card level (SK_ID_PREV) ───────────────────────
    cc_card = cc.groupby("SK_ID_PREV").agg(
        cc_months_count=("MONTHS_BALANCE", "count"),
        cc_utilization_mean=("utilization", "mean"),
        cc_utilization_max=("utilization", "max"),
        cc_maxed_count=("is_maxed", "sum"),
        cc_overlimit_count=("is_overlimit", "sum"),
        cc_paid_min_rate=("paid_min", "mean"),
        cc_paid_full_rate=("paid_full", "mean"),
        cc_atm_ratio_mean=("atm_ratio", "mean"),
        cc_atm_months=("AMT_DRAWINGS_ATM_CURRENT", lambda x: (x > 0).sum()),
        cc_drawings_mean=("AMT_DRAWINGS_CURRENT", "mean"),
        cc_balance_mean=("AMT_BALANCE", "mean"),
        cc_dpd_max=("SK_DPD", "max"),
        cc_dpd_mean=("SK_DPD", "mean"),
        cc_late_count=("SK_DPD", lambda x: (x > 0).sum()),
        cc_ever_demand=("NAME_CONTRACT_STATUS", lambda x: int((x == "Demand").any())),
        cc_balance_change_mean=("balance_change", "mean"),
    )

    # Recency window
    cc_recent = cc[cc["MONTHS_BALANCE"] >= RECENT_MONTHS]
    cc_recent_card = cc_recent.groupby("SK_ID_PREV").agg(
        cc_recent_utilization=("utilization", "mean"),
        cc_recent_dpd_max=("SK_DPD", "max"),
        cc_recent_paid_full=("paid_full", "mean"),
        cc_recent_atm_ratio=("atm_ratio", "mean"),
    )
    cc_card = cc_card.join(cc_recent_card, how="left")

    # Ratio and trajectory features
    cc_card["cc_maxed_rate"] = cc_card["cc_maxed_count"] / cc_card["cc_months_count"].clip(1)
    cc_card["cc_late_rate"] = cc_card["cc_late_count"] / cc_card["cc_months_count"].clip(1)
    # Is utilization getting worse recently?
    cc_card["cc_util_trend"] = cc_card["cc_recent_utilization"] - cc_card["cc_utilization_mean"]

    # ── Step 2: two-hop to applicant level ────────────────────
    prev_ids = prev_app[["SK_ID_PREV", "SK_ID_CURR"]]
    cc_with_curr = prev_ids.merge(cc_card.reset_index(), on="SK_ID_PREV", how="left")

    cc_agg = (
        cc_with_curr.groupby("SK_ID_CURR")
        .agg(
            cc_card_count=("SK_ID_PREV", "count"),
            cc_utilization_mean=("cc_utilization_mean", "mean"),
            cc_utilization_max=("cc_utilization_max", "max"),
            cc_maxed_rate_mean=("cc_maxed_rate", "mean"),
            cc_overlimit_ever=("cc_overlimit_count", lambda x: int((x > 0).any())),
            cc_paid_full_rate_mean=("cc_paid_full_rate", "mean"),
            cc_paid_min_rate_mean=("cc_paid_min_rate", "mean"),
            cc_atm_ratio_mean=("cc_atm_ratio_mean", "mean"),
            cc_dpd_max=("cc_dpd_max", "max"),
            cc_ever_demand=("cc_ever_demand", "max"),
            cc_util_trend_mean=("cc_util_trend", "mean"),
            cc_balance_change_mean=("cc_balance_change_mean", "mean"),
            cc_recent_utilization=("cc_recent_utilization", "mean"),
            cc_recent_dpd_max=("cc_recent_dpd_max", "max"),
        )
        .add_prefix("cc_")
        .rename_axis("SK_ID_CURR")
    )

    cc_agg.columns = [c.replace("cc_cc_", "cc_") for c in cc_agg.columns]

    logger.info(f"Credit card features: {cc_agg.shape[1]} columns")

    del cc, cc_card, cc_recent, cc_recent_card, prev_ids, cc_with_curr
    gc.collect()

    return cc_agg.reset_index()
