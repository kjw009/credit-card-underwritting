from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "instance" / "predictions.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)  # create instance/ if it doesn't exist
SQLALCHEMY_DATABASE_URL = f"sqlite:///{DB_PATH}"

engine = create_engine(
    SQLALCHEMY_DATABASE_URL,
    connect_args={"check_same_thread": False},
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# open a new database session for each request and ensure it's closed after
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
