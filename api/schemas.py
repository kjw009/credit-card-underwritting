from pydantic import BaseModel, field_validator, model_validator
from typing import Any, Literal
from datetime import datetime

# ── Error contract ──────────────────────────────────────────────────────────

ERROR_CODES = {
    "MISSING_FEATURES": {"status": 400, "description": "One or more required features are absent from the request body."},
    "INVALID_FEATURE_VALUE": {"status": 400, "description": "A feature value is non-numeric or out of acceptable range."},
    "PREDICTION_FAILED": {"status": 500, "description": "The model raised an exception during inference."},
    "RATE_LIMIT_EXCEEDED": {"status": 429, "description": "Too many requests. Default limit: 10 predictions/minute per IP."},
    "VALIDATION_ERROR": {"status": 422, "description": "Pydantic schema validation failed."},
    "NOT_FOUND": {"status": 404, "description": "The requested resource does not exist."},
    "INTERNAL_ERROR": {"status": 500, "description": "An unexpected server-side error occurred."},
}


class APIError(BaseModel):
    code: str
    message: str
    detail: Any = None


# ── Prediction I/O ───────────────────────────────────────────────────────────

EmploymentStatus = Literal[
    "Full-Time", "Part-Time", "Self-Employed",
    "Unemployed", "Retired", "Homemaker", "Student"
]
HousingStatus = Literal["Rent", "Mortgage", "Own Outright"]
EducationLevel = Literal[
    "Less than High School", "High School Diploma", "Some College",
    "Associate Degree", "Bachelor Degree", "Master Degree",
    "Doctoral Degree", "Professional Degree"
]


class ApplicationRequest(BaseModel):
    """Simplified application — the API derives all engineered & WoE features."""

    # Personal / Employment
    education_level: EducationLevel
    employment_status: EmploymentStatus
    housing_status: HousingStatus
    years_employed: float
    recent_employment_change: bool = False

    # Income & Expenses (raw $ values)
    annual_income: float
    total_household_income: float
    monthly_rent_mortgage: float
    self_reported_monthly_rent: float
    total_monthly_expenses: float

    # Credit History
    fico_score: float
    credit_utilization_ratio: float        # 0–1
    oldest_account_age_months: float
    num_open_accounts: float
    debt_to_income_ratio: float

    # Loans & Debt
    num_student_loans: float = 0
    student_loan_outstanding_balance: float = 0
    mortgage_outstanding_balance: float = 0
    has_existing_mortgage: bool = False

    # Banking / Assets
    savings_account_balance: float
    avg_monthly_deposits: float
    avg_monthly_withdrawals: float
    payroll_direct_deposit_amount: float = 0
    retirement_account_balance: float = 0

    # Application
    requested_credit_limit: float

    # Pre-computed scores (from credit bureau / scoring systems)
    predicted_default_probability: float   # 0–1
    employment_stability_score: float      # 0–1
    income_stability_score: float          # 0–1
    financial_health_score: float          # 0–1
    combined_risk_score: float             # ~200–600


class PredictionResponse(BaseModel):
    id: int
    decision: str          # "APPROVED" | "DECLINED"
    probability: float     # P(approved)
    fico_score: float | None
    annual_income: float | None
    debt_to_income_ratio: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class PredictionSummary(BaseModel):
    id: int
    decision: str
    probability: float
    fico_score: float | None
    annual_income: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class HealthResponse(BaseModel):
    status: str
    model: str
    feature_count: int
    total_predictions: int

# ── Authentication ────────────────────────────────────────────────────────

class UserCreate(BaseModel):
    """Payload for POST /auth/register."""
    email: str
    password: str
    # Model-level validation to enforce email format and password strength
    @field_validator("email")
    @classmethod
    def email_not_empty(cls, v: str) -> str:
        v = v.strip().lower()
        if not v:
            raise ValueError("Email cannot be empty.")
        return v

    @field_validator("password")
    @classmethod
    def password_min_length(cls, v: str) -> str:
        if len(v) < 8:
            raise ValueError("Password must be at least 8 characters.")
        return v

class UserRead(BaseModel):
    """Returned to the client after registration. Never exposes password_hash."""
    id: int
    email: str
    is_active: bool

    # from_attributes=True lets Pydantic read directly from a SQLAlchemy ORM object
    # instead of requiring a plain dict. Without this, returning a User ORM instance
    # from a route would raise a validation error.
    model_config = {"from_attributes": True}

class Token(BaseModel):
    """Returned to the client after a successful login."""
    access_token: str
    token_type: str = "bearer"

