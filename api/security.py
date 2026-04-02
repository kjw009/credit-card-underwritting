# api/security.py — new file

import os
from datetime import datetime, timedelta, timezone

from passlib.context import CryptContext
from jose import JWTError, jwt
from fastapi import Depends, HTTPException
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from .models import User
from .database import get_db


SECRET_KEY: str = os.environ.get("SECRET_KEY", "")
ALGORITHM: str = "HS256"              # HMAC-SHA256 — standard for single-service JWTs
ACCESS_TOKEN_EXPIRE_MINUTES: int = 30 # Tokens expire after 30 minutes

# ── Password hashing ──────────────────────────────────────────────────────────
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

def get_pwd_hash(password: str) -> str:
    """Hash a plain-text password for storage in the database."""
    return pwd_context.hash(password)

def verify_pwd(plain_password: str, hashed_password: str) -> bool:
    """Return True if plain_password matches the stored hash, False otherwise."""
    return pwd_context.verify(plain_password, hashed_password)

# ── JWT utilities ─────────────────────────────────────────────────────────────
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/token")

def create_access_token(email: str) -> str:
    """Create a signed JWT encoding the user's email as the subject claim.
    
    The token contains:
      sub (subject) — the user's email, used to look them up on each request
      exp (expiry)  — timestamp after which the token is rejected automatically
    """
    expire = datetime.now(timezone.utc) + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    payload = {"sub": email, "exp": expire}
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def verify_token(token: str) -> str:
    """Decode and validate a JWT. Returns the email on success, raises 401 on failure.
    
    python-jose checks the signature and the exp claim automatically.
    A JWTError is raised for any invalid token (tampered, expired, wrong key).
    """
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str | None = payload.get("sub")
        if email is None:
            raise HTTPException(status_code=401, detail="Invalid token")
        return email
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid token")

# ── FastAPI dependencies ───────────────────────────────────────────────────────

def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> User:
    """FastAPI dependency that enforces authentication on any route it is added to.
    
    Usage in a route:
        @app.get("/health")
        def health(_: User = Depends(get_current_user)):
            ...
    
    FastAPI calls this before the route body runs. If the token is missing or
    invalid, a 401 is returned and the route never executes.
    """
    email = verify_token(token)
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise HTTPException(status_code=401, detail="User not found")
    return user

def get_current_active_user(current_user: User = Depends(get_current_user)) -> User:
    """Extends get_current_user to also reject disabled accounts.
    
    Use this on routes where you want to block inactive users even if their
    token is still valid (e.g. after an account is suspended).
    """
    if not current_user.is_active:
        raise HTTPException(status_code=403, detail="Account is disabled")
    return current_user