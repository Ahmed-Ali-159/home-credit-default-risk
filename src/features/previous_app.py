"""
src/features/previous_app.py

Feature engineering for previous_application.csv.
Returns one row per SK_ID_CURR with the prefix 'prev_'.

This is single-level aggregation (groupby SK_ID_CURR directly) since
previous_application has SK_ID_CURR as a foreign key.

Public API:
    build_prev_app_features(prev_app) -> pd.DataFrame
"""

import gc
import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Days columns that have the 365243 sentinel for "no event"
DAYS_SENTINEL_COLS = [
    "DAYS_FIRST_DRAWING",
    "DAYS_FIRST_DUE",
    "DAYS_LAST_DUE_1ST_VERSION",
    "DAYS_LAST_DUE",
    "DAYS_TERMINATION",
]


def build_prev_app_features(prev_app: pd.DataFrame) -> pd.DataFrame:
    """
    Build all previous_application features, one row per SK_ID_CURR.

    Features capture: how often this client has applied, how often they
    were approved/refused, how much credit was historically granted,
    and how their behavior has changed over time (credit ask growth,
    most-recent-application status).
    """
    prev = prev_app.copy()

    # ── Fix 365243 sentinels ──────────────────────────────────
    prev[DAYS_SENTINEL_COLS] = prev[DAYS_SENTINEL_COLS].replace(365243, np.nan)

    approved = prev[prev["NAME_CONTRACT_STATUS"] == "Approved"]

    # ── Count and decision features ───────────────────────────
    agg = prev.groupby("SK_ID_CURR").agg(
        app_count=("SK_ID_PREV", "count"),
        approved_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Approved").sum()),
        refused_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Refused").sum()),
        canceled_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Canceled").sum()),
        unused_count=("NAME_CONTRACT_STATUS", lambda x: (x == "Unused offer").sum()),
        ever_refused_HC=("CODE_REJECT_REASON", lambda x: int((x == "HC").any())),
        ever_refused_SCOFR=("CODE_REJECT_REASON", lambda x: int((x == "SCOFR").any())),
    )

    # ── Approved-only amount features ─────────────────────────
    agg_approved = approved.groupby("SK_ID_CURR").agg(
        credit_mean=("AMT_CREDIT", "mean"),
        credit_max=("AMT_CREDIT", "max"),
        annuity_mean=("AMT_ANNUITY", "mean"),
        application_mean=("AMT_APPLICATION", "mean"),
        term_mean=("CNT_PAYMENT", "mean"),
        insured_rate=("NFLAG_INSURED_ON_APPROVAL", "mean"),
    )

    # Approval ratio: how much was granted vs requested?
    agg_approved["approval_ratio"] = approved.groupby("SK_ID_CURR").apply(
        lambda g: (g["AMT_CREDIT"] / g["AMT_APPLICATION"].clip(1)).mean()
    )
    agg = agg.join(agg_approved, how="left")

    # ── Time / recency features ───────────────────────────────
    agg_time = prev.groupby("SK_ID_CURR").agg(
        days_decision_max=("DAYS_DECISION", "max"),
        days_decision_mean=("DAYS_DECISION", "mean"),
    )
    agg = agg.join(agg_time, how="left")

    # ── Engineered ratios ─────────────────────────────────────
    agg["refusal_rate"] = agg["refused_count"] / agg["app_count"].clip(1)
    agg["days_since_last_app"] = -agg["days_decision_max"]

    # Most recent application status
    most_recent = prev.sort_values("DAYS_DECISION").groupby("SK_ID_CURR").last()
    agg["last_was_refused"] = (most_recent["NAME_CONTRACT_STATUS"] == "Refused").astype(int)

    # ── Credit ask growth: did the applicant escalate their ask? ──
    # last application amount / first application amount
    prev_sorted = prev.sort_values(["SK_ID_CURR", "DAYS_DECISION"])
    first_last = prev_sorted.groupby("SK_ID_CURR")["AMT_APPLICATION"].agg(
        first_application="first",
        last_application="last",
    )
    first_last["credit_ask_growth"] = first_last["last_application"] / first_last[
        "first_application"
    ].clip(1)
    agg = agg.join(first_last[["credit_ask_growth"]], how="left")

    logger.info(f"Previous application features: {agg.shape[1]} columns")

    # Free intermediate tables
    del prev, approved, agg_approved, agg_time, prev_sorted, first_last, most_recent
    gc.collect()

    return agg.add_prefix("prev_").rename_axis("SK_ID_CURR").reset_index()
