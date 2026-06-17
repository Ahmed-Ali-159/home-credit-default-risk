"""
src/features/cross_table.py

Cross-table interaction features.

Why this is its own module:
    These features can ONLY be built AFTER all per-table feature groups
    are computed and merged. They multiply or subtract signals from
    different tables — e.g. "low EXT_SOURCE_2 AND bad payment ratio".

    Think like a loan officer asking questions that combine information:
      "This person has a low external score AND high credit card
      utilization AND was refused before AND has many active loans."
    Each clause individually is meh; the conjunction is a strong signal.

Public API:
    build_cross_table_features(df) -> pd.DataFrame
"""

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def build_cross_table_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add interaction features that combine signals across tables.

    df must already contain all per-table features merged in.
    Uses .get() with NaN defaults so missing features don't crash the function.
    """
    df = df.copy()

    # ── External score x installment payment behavior ─────────
    # Low external score + bad payment ratio = double red flag
    df["ext2_x_inst_ratio"] = df.get("EXT_SOURCE_2", np.nan) * df.get(
        "inst_payment_ratio_mean", np.nan
    )

    # ── Debt burden cross-check ───────────────────────────────
    # High application credit-to-income + high credit card utilization
    # = maxed out from every angle
    df["credit_income_x_cc_util"] = df.get("CREDIT_INCOME_RATIO", np.nan) * df.get(
        "cc_utilization_mean", np.nan
    )

    # ── Refusal x active loans ────────────────────────────────
    # Previously refused AND currently has many active loans = high risk
    df["prev_refusal_x_active"] = df.get("prev_refusal_rate", np.nan) * df.get(
        "buro_active_count", np.nan
    )

    # ── Payment behavior consistency ──────────────────────────
    # Consistent across loan types: POS completion vs CC full payment
    df["pos_cc_behavior"] = df.get("pos_completion_mean", np.nan) * df.get(
        "cc_paid_full_rate_mean", np.nan
    )

    # ── Installment deterioration ─────────────────────────────
    # Is recent payment worse than historical?
    # Negative = getting worse recently
    df["inst_payment_deterioration"] = df.get("inst_recent_payment_ratio", np.nan) - df.get(
        "inst_payment_ratio_mean", np.nan
    )

    # ── Age x clean bureau history ────────────────────────────
    # Young applicant with no bureau history is different risk
    # from young applicant with clean bureau history
    buro_ever_overdue = df.get("buro_ever_overdue", pd.Series(0, index=df.index)).fillna(0)
    df["age_x_buro_clean"] = df.get("AGE_YEARS", np.nan) * (1 - buro_ever_overdue)

    logger.info(f"Cross-table features added. Shape now: {df.shape}")
    return df
