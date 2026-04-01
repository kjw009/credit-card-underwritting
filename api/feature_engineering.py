"""Feature engineering pipeline for credit card underwriting.

Converts simplified user inputs into the 78 features expected by xgb_default.pkl.
"""

import json
import numpy as np
from pathlib import Path

_MODELS_DIR = Path(__file__).parent.parent / "models"
_WOE_MAPS_PATH = _MODELS_DIR / "woe_maps.json"

# ── Ordinal mappings ──────────────────────────────────────────────────────────

EDUCATION_MAP = {
    "Less than High School": 1,
    "High School Diploma": 2,
    "Some College": 3,
    "Associate Degree": 4,
    "Bachelor Degree": 5,
    "Master Degree": 6,
    "Doctoral Degree": 7,
    "Professional Degree": 7,
}

def _fico_tier(score: float) -> int:
    if score < 580:
        return 1
    elif score < 670:
        return 2
    elif score < 740:
        return 3
    elif score < 800:
        return 4
    return 5

# ── WoE encoding ──────────────────────────────────────────────────────────────

_woe_maps: dict | None = None

def _load_woe() -> dict:
    global _woe_maps
    if _woe_maps is None:
        with open(_WOE_MAPS_PATH) as f:
            _woe_maps = json.load(f)
    return _woe_maps


def _woe(feature_name: str, value: float) -> float:
    maps = _load_woe()
    spec = maps.get(feature_name)
    if spec is None:
        return 0.0

    if "cat_map" in spec:
        key = str(int(value)) if float(value).is_integer() else str(value)
        return spec["cat_map"].get(key, 0.0)

    bins = spec["bins"]
    woe_vals = spec["woe"]
    for i, (lo, hi) in enumerate(bins):
        if lo <= value <= hi:
            return woe_vals[i]
    return woe_vals[0] if value < bins[0][0] else woe_vals[-1]


# ── Main pipeline ─────────────────────────────────────────────────────────────

def engineer_features(inp: dict) -> dict:
    """
    Convert simplified application inputs into the 78-feature dict the model expects.

    Required inp keys
    -----------------
    education_level           str  e.g. "Bachelor Degree"
    employment_status         str  e.g. "Full-Time"
    housing_status            str  e.g. "Rent"
    years_employed            float
    recent_employment_change  bool / 0|1
    annual_income             float  (raw $, will be log1p'd)
    total_household_income    float  (raw $)
    monthly_rent_mortgage     float
    self_reported_monthly_rent float
    total_monthly_expenses    float
    fico_score                float
    credit_utilization_ratio  float  0–1
    oldest_account_age_months float
    num_open_accounts         float
    debt_to_income_ratio      float
    num_student_loans         float
    student_loan_outstanding_balance float  (raw $, log1p'd)
    mortgage_outstanding_balance     float  (raw $, log1p'd)
    has_existing_mortgage     bool / 0|1
    savings_account_balance   float  (raw $)
    avg_monthly_deposits      float  (raw $, log1p'd)
    avg_monthly_withdrawals   float  (raw $, log1p'd)
    payroll_direct_deposit_amount float (raw $, log1p'd)
    retirement_account_balance float  (raw $, log1p'd)
    requested_credit_limit    float
    predicted_default_probability float  0–1
    employment_stability_score float  0–1
    income_stability_score    float  0–1
    financial_health_score    float  0–1
    combined_risk_score       float  ~200–600
    """

    # ── Pull raw values ───────────────────────────────────────────────────────
    annual_income_raw           = float(inp["annual_income"])
    total_household_income      = float(inp["total_household_income"])
    monthly_rent_mortgage       = float(inp["monthly_rent_mortgage"])
    total_monthly_expenses      = float(inp["total_monthly_expenses"])
    retirement_account_balance_raw = float(inp["retirement_account_balance"])
    avg_monthly_deposits_raw    = float(inp["avg_monthly_deposits"])
    avg_monthly_withdrawals_raw = float(inp["avg_monthly_withdrawals"])
    payroll_dd_raw              = float(inp["payroll_direct_deposit_amount"])
    mortgage_bal_raw            = float(inp["mortgage_outstanding_balance"])
    student_loan_bal_raw        = float(inp["student_loan_outstanding_balance"])
    self_reported_monthly_rent  = float(inp["self_reported_monthly_rent"])
    fico_score                  = float(inp["fico_score"])
    dti                         = float(inp["debt_to_income_ratio"])
    years_employed              = float(inp["years_employed"])
    num_student_loans           = float(inp["num_student_loans"])
    oldest_account_age_months   = float(inp["oldest_account_age_months"])
    num_open_accounts           = float(inp["num_open_accounts"])
    credit_utilization_ratio    = float(inp["credit_utilization_ratio"])
    savings_account_balance_raw = float(inp["savings_account_balance"])
    requested_credit_limit      = float(inp["requested_credit_limit"])
    predicted_default_prob      = float(inp["predicted_default_probability"])
    emp_stability               = float(inp["employment_stability_score"])
    inc_stability               = float(inp["income_stability_score"])
    financial_health            = float(inp["financial_health_score"])
    combined_risk               = float(inp["combined_risk_score"])
    has_mortgage                = 1.0 if inp.get("has_existing_mortgage") else 0.0
    recent_emp_change           = 1.0 if inp.get("recent_employment_change") else 0.0

    # ── Log1p transforms ──────────────────────────────────────────────────────
    annual_income                   = np.log1p(annual_income_raw)
    retirement_account_balance      = np.log1p(retirement_account_balance_raw)
    avg_monthly_deposits            = np.log1p(avg_monthly_deposits_raw)
    avg_monthly_withdrawals         = np.log1p(avg_monthly_withdrawals_raw)
    payroll_direct_deposit_amount   = np.log1p(payroll_dd_raw)
    mortgage_outstanding_balance    = np.log1p(mortgage_bal_raw)
    student_loan_outstanding_balance = np.log1p(student_loan_bal_raw)
    income_to_requested_limit_ratio = np.log1p(annual_income_raw / (requested_credit_limit + 1))

    # ── Ordinal / categorical ─────────────────────────────────────────────────
    education_level = float(EDUCATION_MAP.get(inp.get("education_level", "Some College"), 3))
    fico_score_tier = float(_fico_tier(fico_score))

    # ── Derived: income / expenses ────────────────────────────────────────────
    monthly_income = annual_income_raw / 12.0
    monthly_disposable_income = monthly_income - total_monthly_expenses
    disposable_income_ratio = (monthly_disposable_income / monthly_income) if monthly_income > 0 else 0.0

    # ── One-hot: employment status ────────────────────────────────────────────
    emp = inp.get("employment_status", "Full-Time")
    es_homemaker     = 1.0 if emp == "Homemaker" else 0.0
    es_part_time     = 1.0 if emp == "Part-Time" else 0.0
    es_retired       = 1.0 if emp == "Retired" else 0.0
    es_self_employed = 1.0 if emp == "Self-Employed" else 0.0
    es_student       = 1.0 if emp == "Student" else 0.0
    es_unemployed    = 1.0 if emp == "Unemployed" else 0.0

    # ── One-hot: housing status ───────────────────────────────────────────────
    hous = inp.get("housing_status", "Rent")
    hs_mortgage    = 1.0 if hous == "Mortgage" else 0.0
    hs_own_outright = 1.0 if hous == "Own Outright" else 0.0
    hs_rent        = 1.0 if hous == "Rent" else 0.0

    # ── Interaction / engineered features ────────────────────────────────────
    fico_utilization_score = fico_score * (1.0 - min(credit_utilization_ratio, 1.0))
    income_dti_capacity    = annual_income_raw * (1.0 - min(dti, 1.0))
    fico_default_alignment = fico_score * (1.0 - predicted_default_prob)
    health_risk_composite  = financial_health * combined_risk
    savings_coverage_months = np.log1p(savings_account_balance_raw) / (total_monthly_expenses + 1)
    debt_burden_ratio      = dti
    credit_depth           = num_open_accounts / (oldest_account_age_months / 12.0 + 1)
    income_cushion         = monthly_disposable_income / (np.log1p(requested_credit_limit) + 1)

    # ── Assemble base feature dict ────────────────────────────────────────────
    base = {
        "education_level":                    education_level,
        "years_employed":                     years_employed,
        "annual_income":                      annual_income,
        "total_household_income":             total_household_income,
        "monthly_rent_mortgage":              monthly_rent_mortgage,
        "total_monthly_expenses":             total_monthly_expenses,
        "retirement_account_balance":         retirement_account_balance,
        "debt_to_income_ratio":               dti,
        "fico_score":                         fico_score,
        "num_student_loans":                  num_student_loans,
        "oldest_account_age_months":          oldest_account_age_months,
        "avg_monthly_deposits":               avg_monthly_deposits,
        "avg_monthly_withdrawals":            avg_monthly_withdrawals,
        "payroll_direct_deposit_amount":      payroll_direct_deposit_amount,
        "has_existing_mortgage":              has_mortgage,
        "recent_employment_change_flag":      recent_emp_change,
        "predicted_default_probability":      predicted_default_prob,
        "mortgage_outstanding_balance":       mortgage_outstanding_balance,
        "student_loan_outstanding_balance":   student_loan_outstanding_balance,
        "income_to_requested_limit_ratio":    income_to_requested_limit_ratio,
        "fico_score_tier":                    fico_score_tier,
        "employment_stability_score":         emp_stability,
        "income_stability_score":             inc_stability,
        "monthly_disposable_income":          monthly_disposable_income,
        "disposable_income_ratio":            disposable_income_ratio,
        "financial_health_score":             financial_health,
        "combined_risk_score":                combined_risk,
        "self_reported_monthly_rent":         self_reported_monthly_rent,
        "employment_status_Homemaker":        es_homemaker,
        "employment_status_Part-Time":        es_part_time,
        "employment_status_Retired":          es_retired,
        "employment_status_Self-Employed":    es_self_employed,
        "employment_status_Student":          es_student,
        "employment_status_Unemployed":       es_unemployed,
        "housing_status_Mortgage":            hs_mortgage,
        "housing_status_Own Outright":        hs_own_outright,
        "housing_status_Rent":                hs_rent,
        "fico_utilization_score":             fico_utilization_score,
        "income_dti_capacity":                income_dti_capacity,
        "fico_default_alignment":             fico_default_alignment,
        "health_risk_composite":              health_risk_composite,
        "savings_coverage_months":            savings_coverage_months,
        "debt_burden_ratio":                  debt_burden_ratio,
        "credit_depth":                       credit_depth,
        "income_cushion":                     income_cushion,
    }

    # ── WoE features ──────────────────────────────────────────────────────────
    woe_pairs = [
        ("health_risk_composite_woe",           "health_risk_composite"),
        ("combined_risk_score_woe",             "combined_risk_score"),
        ("fico_default_alignment_woe",          "fico_default_alignment"),
        ("financial_health_score_woe",          "financial_health_score"),
        ("fico_score_woe",                      "fico_score"),
        ("predicted_default_probability_woe",   "predicted_default_probability"),
        ("disposable_income_ratio_woe",         "disposable_income_ratio"),
        ("debt_to_income_ratio_woe",            "debt_to_income_ratio"),
        ("debt_burden_ratio_woe",               "debt_burden_ratio"),
        ("monthly_disposable_income_woe",       "monthly_disposable_income"),
        ("income_cushion_woe",                  "income_cushion"),
        ("income_dti_capacity_woe",             "income_dti_capacity"),
        ("annual_income_woe",                   "annual_income"),
        ("avg_monthly_deposits_woe",            "avg_monthly_deposits"),
        ("avg_monthly_withdrawals_woe",         "avg_monthly_withdrawals"),
        ("fico_score_tier_woe",                 "fico_score_tier"),
        ("payroll_direct_deposit_amount_woe",   "payroll_direct_deposit_amount"),
        ("total_household_income_woe",          "total_household_income"),
        ("income_to_requested_limit_ratio_woe", "income_to_requested_limit_ratio"),
        ("fico_utilization_score_woe",          "fico_utilization_score"),
        ("oldest_account_age_months_woe",       "oldest_account_age_months"),
        ("credit_depth_woe",                    "credit_depth"),
        ("income_stability_score_woe",          "income_stability_score"),
        ("recent_employment_change_flag_woe",   "recent_employment_change_flag"),
        ("employment_status_Unemployed_woe",    "employment_status_Unemployed"),
        ("total_monthly_expenses_woe",          "total_monthly_expenses"),
        ("years_employed_woe",                  "years_employed"),
        ("employment_stability_score_woe",      "employment_stability_score"),
        ("self_reported_monthly_rent_woe",      "self_reported_monthly_rent"),
        ("monthly_rent_mortgage_woe",           "monthly_rent_mortgage"),
        ("savings_coverage_months_woe",         "savings_coverage_months"),
        ("employment_status_Student_woe",       "employment_status_Student"),
        ("employment_status_Homemaker_woe",     "employment_status_Homemaker"),
    ]

    features = dict(base)
    for woe_col, base_col in woe_pairs:
        features[woe_col] = _woe(base_col, base[base_col])

    return features
