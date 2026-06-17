"""
src/features/pos_cash.py

Feature engineering for POS_CASH_balance.csv.
Returns one row per SK_ID_CURR with the prefix 'pos_'.

Why two-hop aggregation:
    POS_CASH_balance has SK_ID_PREV (a loan ID) but no SK_ID_CURR (client ID).
    We have to:
      1. Aggregate POS_CASH at loan level (SK_ID_PREV).
      2. Look up SK_ID_CURR for each SK_ID_PREV via previous_application.
      3. Aggregate again at client level.

Public API:
    build_pos_features(pos_cash, prev_app) -> pd.DataFrame
"""

import gc
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Exclude Signed rows from DPD aggregations
# (no payment has been due yet — DPD=0 is not a reliability signal)
ACTIVE_STATUSES = ["Active", "Completed", "Demand"]
RECENT_MONTHS = -12


def build_pos_features(
    pos_cash: pd.DataFrame,
    prev_app: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build all POS_CASH-derived features, one row per SK_ID_CURR.
    """
    pos = pos_cash.copy()

    # Exclude Signed rows from DPD aggregations
    # (no payment has been due yet — DPD=0 is not a reliability signal)
    pos_active = pos[pos["NAME_CONTRACT_STATUS"].isin(ACTIVE_STATUSES)]

    # ── Step 1: aggregate to loan level (SK_ID_PREV) ──────────
    pos_loan = pos.groupby("SK_ID_PREV").agg(
        pos_months_count=("MONTHS_BALANCE", "count"),
        pos_instalment_total=("CNT_INSTALMENT", "max"),
        pos_instalment_future=("CNT_INSTALMENT_FUTURE", "min"),  # min = last snapshot
        pos_ever_demand=("NAME_CONTRACT_STATUS", lambda x: int((x == "Demand").any())),
        pos_completed=("NAME_CONTRACT_STATUS", lambda x: int((x == "Completed").any())),
    )

    pos_loan_dpd = pos_active.groupby("SK_ID_PREV").agg(
        pos_dpd_max=("SK_DPD", "max"),
        pos_dpd_mean=("SK_DPD", "mean"),
        pos_late_count=("SK_DPD", lambda x: (x > 0).sum()),
        pos_dpd_def_max=("SK_DPD_DEF", "max"),
    )
    pos_loan = pos_loan.join(pos_loan_dpd, how="left")

    # Recency window — last 12 months
    pos_recent = pos_active[pos_active["MONTHS_BALANCE"] >= RECENT_MONTHS]
    pos_recent_loan = pos_recent.groupby("SK_ID_PREV").agg(
        pos_recent_dpd_max=("SK_DPD", "max"),
        pos_recent_late_count=("SK_DPD", lambda x: (x > 0).sum()),
        pos_recent_demand=("NAME_CONTRACT_STATUS", lambda x: int((x == "Demand").any())),
    )
    pos_loan = pos_loan.join(pos_recent_loan, how="left")

    # ── Ratio features ────────────────────────────────────────
    pos_loan["pos_completion_ratio"] = 1 - pos_loan["pos_instalment_future"] / pos_loan[
        "pos_instalment_total"
    ].clip(1)
    pos_loan["pos_late_rate"] = pos_loan["pos_late_count"] / pos_loan["pos_months_count"].clip(1)

    # ── Step 2: two-hop to applicant level ────────────────────
    prev_ids = prev_app[["SK_ID_PREV", "SK_ID_CURR"]]
    pos_with_curr = prev_ids.merge(pos_loan.reset_index(), on="SK_ID_PREV", how="left")

    pos_agg = (
        pos_with_curr.groupby("SK_ID_CURR")
        .agg(
            pos_loan_count=("SK_ID_PREV", "count"),
            pos_dpd_max=("pos_dpd_max", "max"),
            pos_dpd_mean=("pos_dpd_mean", "mean"),
            pos_late_rate_mean=("pos_late_rate", "mean"),
            pos_ever_demand=("pos_ever_demand", "max"),
            pos_completed_count=("pos_completed", "sum"),
            pos_completion_mean=("pos_completion_ratio", "mean"),
            pos_recent_dpd_max=("pos_recent_dpd_max", "max"),
            pos_recent_demand=("pos_recent_demand", "max"),
        )
        .add_prefix("pos_")
        .rename_axis("SK_ID_CURR")
    )

    # Remove double prefix introduced by .add_prefix on pre-prefixed columns
    pos_agg.columns = [c.replace("pos_pos_", "pos_") for c in pos_agg.columns]

    logger.info(f"POS_CASH features: {pos_agg.shape[1]} columns")

    # Free intermediate tables
    del pos, pos_active, pos_loan, pos_loan_dpd, pos_recent, pos_recent_loan
    del prev_ids, pos_with_curr
    gc.collect()

    return pos_agg.reset_index()
