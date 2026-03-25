"""
Visualization utilities for credit card underwriting analysis.
"""
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parents[2] / "reports" / "figures"


def save_fig(fig: plt.Figure, name: str):
    """Save a matplotlib figure to the reports/figures directory."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGURES_DIR / name, bbox_inches="tight")
