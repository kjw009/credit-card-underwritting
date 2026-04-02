"""Credit Card Underwriting – FastAPI serving layer.

Endpoints
---------
GET  /                  HTMX front-end (HTML)
GET  /health            Liveness + model info
POST /predict           Score a single application (JSON → JSON or HTML fragment)
GET  /predictions       Paginated history from SQLite
GET  /predictions/{id}  Single record

Error contract  →  api/schemas.py::ERROR_CODES
Rate limits     →  10 predictions / minute / IP  (all routes share 60 req/min)
"""

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import joblib
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from sqlalchemy.orm import Session

from .auth import router as auth_router
from .database import Base, engine, get_db
from .feature_engineering import engineer_features
from .limiter import limiter
from .models import PredictionLog, User
from .security import SECRET_KEY, get_current_active_user
from .schemas import (
    APIError,
    ApplicationRequest,
    ERROR_CODES,
    HealthResponse,
    PredictionResponse,
    PredictionSummary,
)

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

# ── Bootstrap ────────────────────────────────────────────────────────────────

Base.metadata.create_all(bind=engine)

ROOT = Path(__file__).parent
MODEL_PATH = ROOT.parent / "models" / "xgb_default.pkl"

try:
    _model = joblib.load(MODEL_PATH)
    FEATURE_NAMES: list[str] = _model.get_booster().feature_names
    log.info("Model loaded. Features: %d", len(FEATURE_NAMES))
except Exception as exc:
    log.error("Failed to load model: %s", exc)
    raise

templates = Jinja2Templates(directory=ROOT / "templates")

app = FastAPI(
    title="Credit Card Underwriting API",
    version="1.0.0",
    description="XGBoost-powered credit decision engine with SQLite audit log.",
    responses={
        400: {"model": APIError, "description": ERROR_CODES["MISSING_FEATURES"]["description"]},
        422: {"model": APIError, "description": ERROR_CODES["VALIDATION_ERROR"]["description"]},
        429: {"model": APIError, "description": ERROR_CODES["RATE_LIMIT_EXCEEDED"]["description"]},
        500: {"model": APIError, "description": ERROR_CODES["INTERNAL_ERROR"]["description"]},
    },
)

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# if not SECRET_KEY or len(SECRET_KEY) < 32:
#     raise RuntimeError(
#         "SECRET_KEY is not set or is too short (min 32 chars). "
#         "Run: export SECRET_KEY=$(python -c 'import secrets; print(secrets.token_hex(32))')"
#     )

# Mount the auth router — adds /auth/register and /auth/token
app.include_router(auth_router)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _score(features: dict[str, float]) -> tuple[str, float]:
    """Run model inference. Returns (decision, probability)."""
    missing = [f for f in FEATURE_NAMES if f not in features]
    if missing:
        raise HTTPException(
            status_code=400,
            detail=APIError(
                code="MISSING_FEATURES",
                message=f"{len(missing)} required feature(s) missing.",
                detail=missing,
            ).model_dump(),
        )

    try:
        X = pd.DataFrame([features])[FEATURE_NAMES]
        proba = float(_model.predict_proba(X)[0][1])
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=APIError(
                code="PREDICTION_FAILED",
                message="Model inference failed.",
                detail=str(exc),
            ).model_dump(),
        )

    decision = "APPROVED" if proba >= 0.5 else "DECLINED"
    return decision, proba


def _log_prediction(
    db: Session,
    features: dict,
    decision: str,
    proba: float,
    client_ip: str,
) -> PredictionLog:
    entry = PredictionLog(
        decision=decision,
        probability=proba,
        fico_score=features.get("fico_score"),
        annual_income=features.get("annual_income"),
        debt_to_income_ratio=features.get("debt_to_income_ratio"),
        client_ip=client_ip,
        features_json=json.dumps(features),
        created_at=datetime.now(timezone.utc),
    )
    db.add(entry)
    db.commit()
    db.refresh(entry)
    return entry


# ── Routes ───────────────────────────────────────────────────────────────────

@app.get("/login", response_class=HTMLResponse, include_in_schema=False)
async def login_page(request: Request):
    return templates.TemplateResponse(request, "login.html", {})


@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def home(request: Request):
    return templates.TemplateResponse(
        request,
        "index.html",
        {"feature_names": FEATURE_NAMES},
    )


@app.get("/health", response_model=HealthResponse, tags=["ops"])
async def health(
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),  # Require authentication
):
    count = db.query(PredictionLog).count()
    return HealthResponse(
        status="ok",
        model="xgb_default",
        feature_count=len(FEATURE_NAMES),
        total_predictions=count,
    )


@app.post(
    "/predict",
    tags=["scoring"],
    responses={
        200: {"model": PredictionResponse},
        400: {"model": APIError},
        429: {"model": APIError},
        500: {"model": APIError},
    },
)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    body: ApplicationRequest,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),  # Require authentication
):
    """Score a single credit application.

    Accepts simplified human-readable inputs and automatically derives all
    engineered features (log transforms, ratios, WoE encoding) before scoring.

    Returns JSON by default.  If the request carries `HX-Request: true`
    (sent automatically by HTMX), returns an HTML fragment instead.
    """
    raw = body.model_dump()
    features = engineer_features(raw)
    decision, proba = _score(features)
    entry = _log_prediction(
        db,
        raw,
        decision,
        proba,
        request.client.host if request.client else "unknown",
    )

    # HTMX requests get an HTML fragment; REST clients get JSON.
    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            request,
            "result_fragment.html",
            {"entry": entry},
            media_type="text/html",
        )

    return PredictionResponse(
        id=entry.id,
        decision=decision,
        probability=proba,
        fico_score=entry.fico_score,
        annual_income=entry.annual_income,
        debt_to_income_ratio=entry.debt_to_income_ratio,
        created_at=entry.created_at,
    )


@app.get(
    "/predictions",
    response_model=list[PredictionSummary],
    tags=["audit"],
)
@limiter.limit("30/minute")
async def list_predictions(
    request: Request,
    limit: int = 20,
    offset: int = 0,
    db: Session = Depends(get_db),
    _: User = Depends(get_current_active_user),  # Require authentication
):
    """Return paginated prediction history (most recent first).

    Supports HTMX polling: returns an HTML table fragment when
    `HX-Request: true` is present.
    """
    rows = (
        db.query(PredictionLog)
        .order_by(PredictionLog.created_at.desc())
        .offset(offset)
        .limit(min(limit, 100))
        .all()
    )

    if request.headers.get("HX-Request") == "true":
        return templates.TemplateResponse(
            request,
            "history_fragment.html",
            {"rows": rows},
            media_type="text/html",
        )

    return rows


@app.get(
    "/predictions/{prediction_id}",
    response_model=PredictionResponse,
    tags=["audit"],
    responses={404: {"model": APIError}},
)
async def get_prediction(prediction_id: int,
                         db: Session = Depends(get_db),
                         _: User = Depends(get_current_active_user) # Require authentication
):  
    row = db.query(PredictionLog).filter(PredictionLog.id == prediction_id).first()
    if not row:
        raise HTTPException(
            status_code=404,
            detail=APIError(
                code="NOT_FOUND",
                message=f"Prediction {prediction_id} not found.",
            ).model_dump(),
        )
    return row


@app.get("/error-codes", tags=["ops"])
async def error_codes(_: User = Depends(get_current_active_user)):
    """Return the full error-code contract for this API."""
    return ERROR_CODES
