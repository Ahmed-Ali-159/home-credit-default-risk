"""
src/features/installments.py

Feature engineering for installments_payments.csv.
Returns one row per SK_ID_CURR with the prefix 'inst_'.

Why this module is unique:
    Installments tracks payment vs schedule at the row level.
    Before any groupby we derive row-level signals:
      - days_late = actual_pay_date - scheduled_pay_date
      - payment_ratio = amount_paid / amount_due
    Then aggregate these to loan level, then to client level.

Public API:
    build_installments_features(installments, prev_app) -> pd.DataFrame
"""

import gc
import logging

import pandas as pd

logger = logging.getLogger(__name__)

# Window for "recent behavior" — last year of installments
RECENT_DAYS = -365


def add_row_level_features(inst: pd.DataFrame) -> pd.DataFrame:
    """
    Derive per-row payment behavior signals BEFORE any groupby.

    These features capture individual installment performance:
    - Was the payment late, early, or on time?
    - Was the full amount paid?
    - Was the payment recorded at all (missing entry = payment never made)?
    """
    inst = inst.copy()

    # Positive = paid late, negative = paid early
    inst["days_late"] = inst["DAYS_ENTRY_PAYMENT"] - inst["DAYS_INSTALMENT"]
    inst["days_late_pos"] = inst["days_late"].clip(lower=0)
    inst["days_early_pos"] = (-inst["days_late"]).clip(lower=0)

    # Payment completeness
    inst["payment_ratio"] = inst["AMT_PAYMENT"] / inst["AMT_INSTALMENT"].clip(1)
    inst["payment_deficit"] = inst["AMT_INSTALMENT"] - inst["AMT_PAYMENT"]

    # Binary flags
    inst["is_late"] = (inst["days_late"] > 0).astype(int)
    inst["is_very_late"] = (inst["days_late"] > 30).astype(int)
    inst["is_underpaid"] = (inst["payment_deficit"] > 0).astype(int)
    inst["is_early"] = (inst["days_late"] < 0).astype(int)
    inst["is_missing"] = inst["DAYS_ENTRY_PAYMENT"].isna().astype(int)

    return inst


def build_installments_features(
    installments: pd.DataFrame,
    prev_app: pd.DataFrame,
) -> pd.DataFrame:
    """Build all installment-derived features, one row per SK_ID_CURR."""
    inst = add_row_level_features(installments)

    # Use latest installment version for restructured loans
    # (some loans have multiple NUM_INSTALMENT_VERSION values)
    inst_latest = (
        inst.sort_values("NUM_INSTALMENT_VERSION")
        .groupby(["SK_ID_PREV", "NUM_INSTALMENT_NUMBER"])
        .last()
        .reset_index()
    )

    # ── Step 1: loan level (SK_ID_PREV) ───────────────────────
    inst_loan = inst_latest.groupby("SK_ID_PREV").agg(
        inst_count=("NUM_INSTALMENT_NUMBER", "count"),
        inst_days_late_max=("days_late", "max"),
        inst_days_late_mean=("days_late", "mean"),
        inst_days_late_sum=("days_late_pos", "sum"),
        inst_late_count=("is_late", "sum"),
        inst_very_late_count=("is_very_late", "sum"),
        inst_early_count=("is_early", "sum"),
        inst_days_early_mean=("days_early_pos", "mean"),
        inst_payment_ratio_mean=("payment_ratio", "mean"),
        inst_payment_ratio_min=("payment_ratio", "min"),
        inst_payment_deficit_sum=("payment_deficit", "sum"),
        inst_payment_deficit_max=("payment_deficit", "max"),
        inst_underpaid_count=("is_underpaid", "sum"),
        inst_missing_count=("is_missing", "sum"),
    )

    # Recency window
    inst_recent = inst_latest[inst_latest["DAYS_INSTALMENT"] >= RECENT_DAYS]
    inst_recent_loan = inst_recent.groupby("SK_ID_PREV").agg(
        inst_recent_days_late_max=("days_late", "max"),
        inst_recent_late_count=("is_late", "sum"),
        inst_recent_payment_ratio=("payment_ratio", "mean"),
        inst_recent_deficit_sum=("payment_deficit", "sum"),
    )
    inst_loan = inst_loan.join(inst_recent_loan, how="left")

    # Ratio features at loan level
    inst_loan["inst_late_rate"] = inst_loan["inst_late_count"] / inst_loan["inst_count"].clip(1)
    inst_loan["inst_very_late_rate"] = inst_loan["inst_very_late_count"] / inst_loan[
        "inst_count"
    ].clip(1)
    inst_loan["inst_early_rate"] = inst_loan["inst_early_count"] / inst_loan["inst_count"].clip(1)

    # ── Step 2: two-hop to applicant level ────────────────────
    prev_ids = prev_app[["SK_ID_PREV", "SK_ID_CURR"]]
    inst_with_curr = prev_ids.merge(inst_loan.reset_index(), on="SK_ID_PREV", how="left")

    inst_agg = (
        inst_with_curr.groupby("SK_ID_CURR")
        .agg(
            inst_loan_count=("SK_ID_PREV", "count"),
            inst_days_late_max=("inst_days_late_max", "max"),
            inst_days_late_mean=("inst_days_late_mean", "mean"),
            inst_late_rate_mean=("inst_late_rate", "mean"),
            inst_very_late_rate_mean=("inst_very_late_rate", "mean"),
            inst_payment_ratio_mean=("inst_payment_ratio_mean", "mean"),
            inst_payment_ratio_min=("inst_payment_ratio_min", "min"),
            inst_deficit_sum=("inst_payment_deficit_sum", "sum"),
            inst_early_rate_mean=("inst_early_rate", "mean"),
            inst_days_early_mean=("inst_days_early_mean", "mean"),
            inst_missing_sum=("inst_missing_count", "sum"),
            inst_recent_days_late_max=("inst_recent_days_late_max", "max"),
            inst_recent_payment_ratio=("inst_recent_payment_ratio", "mean"),
            inst_recent_deficit_sum=("inst_recent_deficit_sum", "sum"),
        )
        .add_prefix("inst_")
        .rename_axis("SK_ID_CURR")
    )

    inst_agg.columns = [c.replace("inst_inst_", "inst_") for c in inst_agg.columns]

    logger.info(f"Installments features: {inst_agg.shape[1]} columns")

    del inst, inst_latest, inst_loan, inst_recent, inst_recent_loan
    del prev_ids, inst_with_curr
    gc.collect()

    return inst_agg.reset_index()
