# src/data_loader.py
"""
Responsible for loading raw data only.
"""

import pandas as pd
from src.config import DATA_PATH


def load_data(path: str = DATA_PATH) -> pd.DataFrame:
    """Load the raw CSV dataset and return a DataFrame."""
    df = pd.read_csv(path)
    print(f"✅ Data loaded  →  shape: {df.shape}")
    return df
