"""Tests for the API feature engineering pipeline."""

import pytest
from api.feature_engineering import engineer_features, EDUCATION_MAP, _fico_tier


SAMPLE_INPUT = {
    "education_level": "Bachelor Degree",
    "employment_status": "Full-Time",
    "housing_status": "Rent",
    "years_employed": 5.0,
    "recent_employment_change": 0,
    "annual_income": 75000.0,
    "total_household_income": 75000.0,
    "monthly_rent_mortgage": 1500.0,
    "self_reported_monthly_rent": 1500.0,
    "total_monthly_expenses": 3000.0,
    "fico_score": 720.0,
    "credit_utilization_ratio": 0.25,
    "oldest_account_age_months": 84.0,
    "num_open_accounts": 4.0,
    "debt_to_income_ratio": 0.30,
    "num_student_loans": 1.0,
    "student_loan_outstanding_balance": 15000.0,
    "mortgage_outstanding_balance": 0.0,
    "has_existing_mortgage": 0,
    "savings_account_balance": 10000.0,
    "avg_monthly_deposits": 6500.0,
    "avg_monthly_withdrawals": 5800.0,
    "payroll_direct_deposit_amount": 6000.0,
    "retirement_account_balance": 20000.0,
    "requested_credit_limit": 5000.0,
    "predicted_default_probability": 0.12,
    "employment_stability_score": 0.80,
    "income_stability_score": 0.75,
    "financial_health_score": 0.70,
    "combined_risk_score": 420.0,
}


def test_engineer_features_returns_dict():
    result = engineer_features(SAMPLE_INPUT)
    assert isinstance(result, dict)


def test_engineer_features_key_count():
    result = engineer_features(SAMPLE_INPUT)
    assert len(result) == 78


def test_fico_tier_boundaries():
    assert _fico_tier(579) == 1
    assert _fico_tier(580) == 2
    assert _fico_tier(670) == 3
    assert _fico_tier(740) == 4
    assert _fico_tier(800) == 5


def test_education_map_coverage():
    assert EDUCATION_MAP["Bachelor Degree"] == 5
    assert EDUCATION_MAP["Doctoral Degree"] == 7


def test_one_hot_employment_status():
    result = engineer_features({**SAMPLE_INPUT, "employment_status": "Student"})
    assert result["employment_status_Student"] == 1.0
    assert result["employment_status_Unemployed"] == 0.0


def test_one_hot_housing_status():
    result = engineer_features({**SAMPLE_INPUT, "housing_status": "Mortgage"})
    assert result["housing_status_Mortgage"] == 1.0
    assert result["housing_status_Rent"] == 0.0
