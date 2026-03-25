"""
Train credit card underwriting models.
"""
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"
PROCESSED_DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"


def train(input_path: Path = PROCESSED_DATA_DIR / "train.csv"):
    """Load processed data and train model."""
    df = pd.read_csv(input_path)
    # Add training logic here
    pass


if __name__ == "__main__":
    train()
