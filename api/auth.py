from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from .database import get_db
from .models import User
from .schemas import Token, UserCreate, UserRead
from .security import create_access_token, get_pwd_hash, verify_pwd


router = APIRouter(prefix="/auth", tags=["auth"])

@router.post("/register", response_model=UserRead, status_code=201)
def register_user(user: UserCreate, db: Session = Depends(get_db)):
    """Register a new user account.
    
    - Rejects duplicate emails with a 409 Conflict.
    - Stores a bcrypt hash of the password — the plain password is never saved.
    - Returns the created user (id, email, is_active) — never the password hash.
    """
    # Reject if email is already taken
    existing_user = db.query(User).filter(User.email == user.email).first()
    if existing_user:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="An account with this email already exists."
        )

    # Hash the password before storing — never store plain text
    new_user = User(
        email=user.email,
        password_hash=get_pwd_hash(user.password),
        is_active=True,
    )
    db.add(new_user)
    db.commit()
    db.refresh(new_user)  # refresh to populate the auto-generated id
    return new_user


@router.post("/token", response_model=Token)
def login_for_access_token(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Log in and receive a JWT bearer token.
    
    Accepts: application/x-www-form-urlencoded with fields `username` and `password`.
    Note: OAuth2PasswordRequestForm always uses the field name `username` — we treat
    it as the user's email address.
    
    The error message is intentionally generic (does not say whether the email
    or password was wrong) to prevent attackers from enumerating valid emails.
    """
    # Look up the user by email (OAuth2RequestForm calls the field `username`)
    user = db.query(User).filter(User.email == form_data.username).first()

    # Reject if user not found or password does not match
    # Generic error message prevents username enumeration
    if not user or not verify_pwd(form_data.password, user.password_hash):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect email or password.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is disabled."
        )

    return Token(access_token=create_access_token(user.email))