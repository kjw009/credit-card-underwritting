"""
Feature engineering for credit card underwriting.
"""
import pandas as pd
import numpy as np


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Transform raw data into features for modeling."""
    df = df.copy()
    # Add feature engineering steps here
    return df
