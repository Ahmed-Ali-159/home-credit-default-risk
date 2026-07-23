# api/schemas.py

"""Pydantic request/response models for the prediction API."""

from pydantic import BaseModel, Field


class LoanApplicationRequest(BaseModel):
    """All application + bureau + history features for a complete demo prediction."""

    # ── Core application fields ───────────────────────────────
    NAME_CONTRACT_TYPE: str
    AMT_CREDIT: float = Field(gt=0)
    AMT_INCOME_TOTAL: float = Field(gt=0)
    AMT_ANNUITY: float | None = Field(default=None, gt=0)
    AMT_GOODS_PRICE: float | None = Field(default=None, gt=0)
    CODE_GENDER: str
    FLAG_OWN_CAR: str
    FLAG_OWN_REALTY: str
    CNT_CHILDREN: int = Field(ge=0, default=0)
    CNT_FAM_MEMBERS: float | None = Field(default=None, ge=1)
    DAYS_BIRTH: int = Field(lt=0)
    DAYS_EMPLOYED: int | None = Field(default=None)
    DAYS_REGISTRATION: float | None = Field(default=None)
    DAYS_ID_PUBLISH: int | None = Field(default=None)
    DAYS_LAST_PHONE_CHANGE: float | None = Field(default=None)
    NAME_INCOME_TYPE: str
    OCCUPATION_TYPE: str | None = Field(default=None)
    ORGANIZATION_TYPE: str | None = Field(default=None)
    NAME_EDUCATION_TYPE: str
    NAME_FAMILY_STATUS: str
    NAME_HOUSING_TYPE: str
    EXT_SOURCE_1: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_2: float | None = Field(default=None, ge=0, le=1)
    EXT_SOURCE_3: float | None = Field(default=None, ge=0, le=1)
    REGION_RATING_CLIENT: int = Field(ge=1, le=3, default=2)
    REGION_POPULATION_RELATIVE: float | None = Field(default=None)
    OWN_CAR_AGE: float | None = Field(default=None)
    HOUR_APPR_PROCESS_START: int | None = Field(default=None)

    # ── Contact / flag fields ─────────────────────────────────
    FLAG_MOBIL: int | None = Field(default=None)
    FLAG_EMP_PHONE: int | None = Field(default=None)
    FLAG_WORK_PHONE: int | None = Field(default=None)
    FLAG_CONT_MOBILE: int | None = Field(default=None)
    FLAG_PHONE: int | None = Field(default=None)
    FLAG_EMAIL: int | None = Field(default=None)

    # ── Region mismatch flags ─────────────────────────────────
    REG_REGION_NOT_LIVE_REGION: int | None = Field(default=None)
    REG_REGION_NOT_WORK_REGION: int | None = Field(default=None)
    LIVE_REGION_NOT_WORK_REGION: int | None = Field(default=None)
    REG_CITY_NOT_LIVE_CITY: int | None = Field(default=None)
    REG_CITY_NOT_WORK_CITY: int | None = Field(default=None)
    LIVE_CITY_NOT_WORK_CITY: int | None = Field(default=None)

    # ── Building / apartment features ────────────────────────
    APARTMENTS_AVG: float | None = Field(default=None)
    BASEMENTAREA_AVG: float | None = Field(default=None)
    YEARS_BEGINEXPLUATATION_AVG: float | None = Field(default=None)
    YEARS_BUILD_AVG: float | None = Field(default=None)
    COMMONAREA_AVG: float | None = Field(default=None)
    ELEVATORS_AVG: float | None = Field(default=None)
    ENTRANCES_AVG: float | None = Field(default=None)
    FLOORSMAX_AVG: float | None = Field(default=None)
    FLOORSMIN_AVG: float | None = Field(default=None)
    LANDAREA_AVG: float | None = Field(default=None)
    LIVINGAPARTMENTS_AVG: float | None = Field(default=None)
    LIVINGAREA_AVG: float | None = Field(default=None)
    NONLIVINGAPARTMENTS_AVG: float | None = Field(default=None)
    NONLIVINGAREA_AVG: float | None = Field(default=None)
    APARTMENTS_MODE: float | None = Field(default=None)
    BASEMENTAREA_MODE: float | None = Field(default=None)
    YEARS_BEGINEXPLUATATION_MODE: float | None = Field(default=None)
    YEARS_BUILD_MODE: float | None = Field(default=None)
    COMMONAREA_MODE: float | None = Field(default=None)
    ELEVATORS_MODE: float | None = Field(default=None)
    ENTRANCES_MODE: float | None = Field(default=None)
    FLOORSMAX_MODE: float | None = Field(default=None)
    FLOORSMIN_MODE: float | None = Field(default=None)
    LANDAREA_MODE: float | None = Field(default=None)
    LIVINGAPARTMENTS_MODE: float | None = Field(default=None)
    LIVINGAREA_MODE: float | None = Field(default=None)
    NONLIVINGAPARTMENTS_MODE: float | None = Field(default=None)
    NONLIVINGAREA_MODE: float | None = Field(default=None)
    APARTMENTS_MEDI: float | None = Field(default=None)
    BASEMENTAREA_MEDI: float | None = Field(default=None)
    YEARS_BEGINEXPLUATATION_MEDI: float | None = Field(default=None)
    YEARS_BUILD_MEDI: float | None = Field(default=None)
    COMMONAREA_MEDI: float | None = Field(default=None)
    ELEVATORS_MEDI: float | None = Field(default=None)
    ENTRANCES_MEDI: float | None = Field(default=None)
    FLOORSMAX_MEDI: float | None = Field(default=None)
    FLOORSMIN_MEDI: float | None = Field(default=None)
    LANDAREA_MEDI: float | None = Field(default=None)
    LIVINGAPARTMENTS_MEDI: float | None = Field(default=None)
    LIVINGAREA_MEDI: float | None = Field(default=None)
    NONLIVINGAPARTMENTS_MEDI: float | None = Field(default=None)
    NONLIVINGAREA_MEDI: float | None = Field(default=None)
    TOTALAREA_MODE: float | None = Field(default=None)
    EMERGENCYSTATE_MODE: str | None = Field(default=None)

    # ── Social circle ─────────────────────────────────────────
    OBS_30_CNT_SOCIAL_CIRCLE: float | None = Field(default=None)
    OBS_60_CNT_SOCIAL_CIRCLE: float | None = Field(default=None)
    DEF_30_CNT_SOCIAL_CIRCLE: float | None = Field(default=None)
    DEF_60_CNT_SOCIAL_CIRCLE: float | None = Field(default=None)

    # ── Document flags ────────────────────────────────────────
    FLAG_DOCUMENT_2: int | None = Field(default=None)
    FLAG_DOCUMENT_3: int | None = Field(default=None)
    FLAG_DOCUMENT_4: int | None = Field(default=None)
    FLAG_DOCUMENT_5: int | None = Field(default=None)
    FLAG_DOCUMENT_6: int | None = Field(default=None)
    FLAG_DOCUMENT_7: int | None = Field(default=None)
    FLAG_DOCUMENT_8: int | None = Field(default=None)
    FLAG_DOCUMENT_9: int | None = Field(default=None)
    FLAG_DOCUMENT_10: int | None = Field(default=None)
    FLAG_DOCUMENT_11: int | None = Field(default=None)
    FLAG_DOCUMENT_12: int | None = Field(default=None)
    FLAG_DOCUMENT_13: int | None = Field(default=None)
    FLAG_DOCUMENT_14: int | None = Field(default=None)
    FLAG_DOCUMENT_15: int | None = Field(default=None)
    FLAG_DOCUMENT_16: int | None = Field(default=None)
    FLAG_DOCUMENT_17: int | None = Field(default=None)
    FLAG_DOCUMENT_18: int | None = Field(default=None)
    FLAG_DOCUMENT_19: int | None = Field(default=None)
    FLAG_DOCUMENT_20: int | None = Field(default=None)
    FLAG_DOCUMENT_21: int | None = Field(default=None)

    # ── Weekday one-hot (pass the raw string, predictor handles OHE) ──
    WEEKDAY_APPR_PROCESS_START: str | None = Field(default=None)

    # ── Categorical building fields (OHE at serving time) ────
    FONDKAPREMONT_MODE: str | None = Field(default=None)
    HOUSETYPE_MODE: str | None = Field(default=None)
    WALLSMATERIAL_MODE: str | None = Field(default=None)

    # ── Bureau aggregates ─────────────────────────────────────
    buro_loan_count: float | None = Field(default=None)
    buro_active_count: float | None = Field(default=None)
    buro_closed_count: float | None = Field(default=None)
    buro_prolonged_sum: float | None = Field(default=None)
    buro_credit_sum_mean: float | None = Field(default=None)
    buro_credit_sum_max: float | None = Field(default=None)
    buro_debt_sum: float | None = Field(default=None)
    buro_overdue_sum: float | None = Field(default=None)
    buro_overdue_max: float | None = Field(default=None)
    buro_dpd_max: float | None = Field(default=None)
    buro_ever_overdue: float | None = Field(default=None)
    buro_days_credit_max: float | None = Field(default=None)
    buro_days_credit_mean: float | None = Field(default=None)
    buro_worst_status: float | None = Field(default=None)
    buro_dpd_total: float | None = Field(default=None)
    buro_dpd_rate_mean: float | None = Field(default=None)
    buro_months_total: float | None = Field(default=None)
    buro_recent_dpd_sum: float | None = Field(default=None)
    buro_recent_worst: float | None = Field(default=None)
    buro_active_credit_mean: float | None = Field(default=None)
    buro_active_debt_sum: float | None = Field(default=None)
    buro_active_overdue_max: float | None = Field(default=None)
    buro_active_ratio: float | None = Field(default=None)
    buro_overdue_ratio: float | None = Field(default=None)
    buro_debt_ratio: float | None = Field(default=None)
    buro_recent_vs_overall: float | None = Field(default=None)

    # ── Previous applications ─────────────────────────────────
    prev_app_count: float | None = Field(default=None)
    prev_approved_count: float | None = Field(default=None)
    prev_refused_count: float | None = Field(default=None)
    prev_canceled_count: float | None = Field(default=None)
    prev_unused_count: float | None = Field(default=None)
    prev_ever_refused_HC: float | None = Field(default=None)
    prev_ever_refused_SCOFR: float | None = Field(default=None)
    prev_credit_mean: float | None = Field(default=None)
    prev_credit_max: float | None = Field(default=None)
    prev_annuity_mean: float | None = Field(default=None)
    prev_application_mean: float | None = Field(default=None)
    prev_term_mean: float | None = Field(default=None)
    prev_insured_rate: float | None = Field(default=None)
    prev_approval_ratio: float | None = Field(default=None)
    prev_days_decision_max: float | None = Field(default=None)
    prev_days_decision_mean: float | None = Field(default=None)
    prev_refusal_rate: float | None = Field(default=None)
    prev_days_since_last_app: float | None = Field(default=None)
    prev_last_was_refused: float | None = Field(default=None)
    prev_credit_ask_growth: float | None = Field(default=None)

    # ── POS cash ──────────────────────────────────────────────
    pos_loan_count: float | None = Field(default=None)
    pos_dpd_max: float | None = Field(default=None)
    pos_dpd_mean: float | None = Field(default=None)
    pos_late_rate_mean: float | None = Field(default=None)
    pos_ever_demand: float | None = Field(default=None)
    pos_completed_count: float | None = Field(default=None)
    pos_completion_mean: float | None = Field(default=None)
    pos_recent_dpd_max: float | None = Field(default=None)
    pos_recent_demand: float | None = Field(default=None)

    # ── Installments ──────────────────────────────────────────
    inst_loan_count: float | None = Field(default=None)
    inst_days_late_max: float | None = Field(default=None)
    inst_days_late_mean: float | None = Field(default=None)
    inst_late_rate_mean: float | None = Field(default=None)
    inst_very_late_rate_mean: float | None = Field(default=None)
    inst_payment_ratio_mean: float | None = Field(default=None)
    inst_payment_ratio_min: float | None = Field(default=None)
    inst_deficit_sum: float | None = Field(default=None)
    inst_early_rate_mean: float | None = Field(default=None)
    inst_days_early_mean: float | None = Field(default=None)
    inst_missing_sum: float | None = Field(default=None)
    inst_recent_days_late_max: float | None = Field(default=None)
    inst_recent_payment_ratio: float | None = Field(default=None)
    inst_recent_deficit_sum: float | None = Field(default=None)

    # ── Credit card ───────────────────────────────────────────
    cc_card_count: float | None = Field(default=None)
    cc_utilization_mean: float | None = Field(default=None)
    cc_utilization_max: float | None = Field(default=None)
    cc_maxed_rate_mean: float | None = Field(default=None)
    cc_overlimit_ever: float | None = Field(default=None)
    cc_paid_full_rate_mean: float | None = Field(default=None)
    cc_paid_min_rate_mean: float | None = Field(default=None)
    cc_atm_ratio_mean: float | None = Field(default=None)
    cc_dpd_max: float | None = Field(default=None)
    cc_ever_demand: float | None = Field(default=None)
    cc_util_trend_mean: float | None = Field(default=None)
    cc_balance_change_mean: float | None = Field(default=None)
    cc_recent_utilization: float | None = Field(default=None)
    cc_recent_dpd_max: float | None = Field(default=None)

    # ── Cross-table interaction features ─────────────────────
    ext2_x_inst_ratio: float | None = Field(default=None)
    credit_income_x_cc_util: float | None = Field(default=None)
    prev_refusal_x_active: float | None = Field(default=None)
    pos_cc_behavior: float | None = Field(default=None)
    inst_payment_deterioration: float | None = Field(default=None)
    age_x_buro_clean: float | None = Field(default=None)

    model_config = {"extra": "ignore"}


class PredictionResponse(BaseModel):
    default_probability: float
    risk_tier: str
    model_version: str

    @classmethod
    def from_probability(cls, probability: float, model_version: str) -> "PredictionResponse":
        if probability < 0.05:
            tier = "LOW"
        elif probability < 0.15:
            tier = "MEDIUM"
        elif probability < 0.30:
            tier = "HIGH"
        else:
            tier = "VERY_HIGH"
        return cls(
            default_probability=round(probability, 6), risk_tier=tier, model_version=model_version
        )


class ShapFeature(BaseModel):
    feature: str
    shap_value: float
    direction: str
    feature_value: float | None = None


class ExplainResponse(BaseModel):
    default_probability: float
    risk_tier: str
    model_version: str
    baseline_probability: float
    top_features: list[ShapFeature]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_version: str


class ModelInfoResponse(BaseModel):
    model_name: str
    model_version: str
    cv_auc: float | None
    n_features: int
