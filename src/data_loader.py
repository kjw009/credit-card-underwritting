"""
Data loading utilities for credit card underwriting.
"""
import pandas as pd
from pathlib import Path

RAW_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "processed"


def load_raw(filename: str) -> pd.DataFrame:
    """Load a raw data file."""
    return pd.read_csv(RAW_DATA_DIR / filename)


def save_processed(df: pd.DataFrame, filename: str):
    """Save a processed DataFrame."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_DATA_DIR / filename, index=False)
