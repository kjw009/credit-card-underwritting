"""
Generate predictions from trained models.
"""
import pandas as pd
from pathlib import Path

MODELS_DIR = Path(__file__).resolve().parents[2] / "models"


def predict(input_path: Path, model_path: Path) -> pd.Series:
    """Load a trained model and generate predictions."""
    df = pd.read_csv(input_path)
    # Add prediction logic here
    pass


if __name__ == "__main__":
    pass
