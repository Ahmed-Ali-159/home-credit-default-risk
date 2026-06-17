"""
src/features/bureau.py

Feature engineering for bureau + bureau_balance tables.
Returns one row per SK_ID_CURR with the prefix 'buro_'.

Why this is its own module:
    Bureau is the most complex feature group — it's a two-hop aggregation:
      bureau_balance → aggregate to loan level (SK_ID_BUREAU)
      bureau (joined with above) → aggregate to client level (SK_ID_CURR)

    Bureau also has the most variation in CREDIT_ACTIVE statuses, requiring
    separate aggregations for Active vs Closed credits to capture
    "current debt burden" separately from "historical behavior".

Public API:
    build_bureau_features(bureau, bureau_balance) -> pd.DataFrame
"""

import gc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# bureau_balance.STATUS encoding:
#   '0' = no DPD
#   '1' through '5' = DPD buckets (1-30, 31-60, 61-90, 91-120, 120+)
#   'C' = closed
#   'X' = status unknown
STATUS_MAP = {"0": 0, "1": 1, "2": 2, "3": 3, "4": 4, "5": 5, "C": -1, "X": np.nan}
DPD_STATUSES = ["1", "2", "3", "4", "5"]


def aggregate_bureau_balance(bureau_balance: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate bureau_balance to one row per SK_ID_BUREAU.

    Produces features capturing how the bureau credit performed over time:
    - How long has it been tracked
    - How many months it was delinquent
    - Worst status reached
    - Recent (last 12 months) behavior vs overall
    """
    bb = bureau_balance.copy()
    bb["STATUS_NUM"] = bb["STATUS"].map(STATUS_MAP)

    # ── Lifetime aggregations ────────────────────────────────
    bb_loan = bb.groupby("SK_ID_BUREAU").agg(
        bb_months_count=("MONTHS_BALANCE", "count"),
        bb_months_min=("MONTHS_BALANCE", "min"),
        bb_dpd_count=("STATUS", lambda x: x.isin(DPD_STATUSES).sum()),
        bb_status_C_count=("STATUS", lambda x: (x == "C").sum()),
        bb_status_num_max=("STATUS_NUM", "max"),
        bb_status_num_mean=("STATUS_NUM", "mean"),
    )

    # ── Recent 12 months — has behavior changed lately? ─────
    bb_recent = bb[bb["MONTHS_BALANCE"] >= -12]
    bb_recent_loan = bb_recent.groupby("SK_ID_BUREAU").agg(
        bb_recent_dpd_count=("STATUS", lambda x: x.isin(DPD_STATUSES).sum()),
        bb_recent_worst=("STATUS_NUM", "max"),
    )
    bb_loan = bb_loan.join(bb_recent_loan, how="left")

    # Delinquency rate (normalised by loan length so short and long loans compare)
    bb_loan["bb_dpd_rate"] = bb_loan["bb_dpd_count"] / bb_loan["bb_months_count"].clip(1)

    return bb_loan


def build_bureau_features(
    bureau: pd.DataFrame,
    bureau_balance: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build all bureau-derived features, one row per SK_ID_CURR.

    The output dataframe has columns prefixed with 'buro_' and a single
    SK_ID_CURR column suitable for merging into the application table.
    """
    # Step 1: aggregate bureau_balance to loan level
    bb_loan = aggregate_bureau_balance(bureau_balance)

    # Step 2: merge balance info into bureau, then aggregate to client level
    buro = bureau.copy()
    buro = buro.merge(bb_loan, on="SK_ID_BUREAU", how="left")

    # Active vs closed subsets
    buro_active = buro[buro["CREDIT_ACTIVE"] == "Active"]

    # ── Main aggregation ─────────────────────────────────────
    agg = buro.groupby("SK_ID_CURR").agg(
        # Volume
        loan_count=("SK_ID_BUREAU", "count"),
        active_count=("CREDIT_ACTIVE", lambda x: (x == "Active").sum()),
        closed_count=("CREDIT_ACTIVE", lambda x: (x == "Closed").sum()),
        prolonged_sum=("CNT_CREDIT_PROLONG", "sum"),
        # Amounts
        credit_sum_mean=("AMT_CREDIT_SUM", "mean"),
        credit_sum_max=("AMT_CREDIT_SUM", "max"),
        debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        overdue_sum=("AMT_CREDIT_SUM_OVERDUE", "sum"),
        overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
        # Delinquency
        dpd_max=("CREDIT_DAY_OVERDUE", "max"),
        ever_overdue=("CREDIT_DAY_OVERDUE", lambda x: int((x > 0).any())),
        # Time / recency
        days_credit_max=("DAYS_CREDIT", "max"),
        days_credit_mean=("DAYS_CREDIT", "mean"),
        # Bureau_balance rollup
        worst_status=("bb_status_num_max", "max"),
        dpd_total=("bb_dpd_count", "sum"),
        dpd_rate_mean=("bb_dpd_rate", "mean"),
        months_total=("bb_months_count", "sum"),
        recent_dpd_sum=("bb_recent_dpd_count", "sum"),
        recent_worst=("bb_recent_worst", "max"),
    )

    # ── Active-only aggregations (current debt burden) ──────
    agg_active = buro_active.groupby("SK_ID_CURR").agg(
        active_credit_mean=("AMT_CREDIT_SUM", "mean"),
        active_debt_sum=("AMT_CREDIT_SUM_DEBT", "sum"),
        active_overdue_max=("AMT_CREDIT_SUM_OVERDUE", "max"),
    )
    agg = agg.join(agg_active, how="left")

    # ── Ratio features ───────────────────────────────────────
    agg["active_ratio"] = agg["active_count"] / agg["loan_count"].clip(1)
    agg["overdue_ratio"] = (agg["overdue_sum"] > 0).astype(int)
    agg["debt_ratio"] = agg["debt_sum"] / agg["credit_sum_mean"].clip(1)

    # Trajectory: is recent behavior worse than historical?
    agg["recent_vs_overall"] = agg["recent_worst"] - agg["worst_status"]

    logger.info(f"Bureau features: {agg.shape[1]} columns for {agg.shape[0]:,} clients")

    # Free large intermediate tables
    del bb_loan, buro, buro_active, agg_active
    gc.collect()

    return agg.add_prefix("buro_").rename_axis("SK_ID_CURR").reset_index()
