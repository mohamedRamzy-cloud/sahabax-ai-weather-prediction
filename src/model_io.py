# src/model_io.py
"""
Save, load, and run inference with the trained pipeline model.
"""

import os
import numpy as np
import joblib
import pandas as pd
from src.config import MODEL_PATH, WIND_MAPPING


def save_model(model_dict: dict, path: str = MODEL_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model_dict, path)
    print(f" Model saved  →  {path}")


def load_model(path: str = MODEL_PATH) -> dict:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Model not found at: {path}")
    model_dict = joblib.load(path)
    print(f" Model loaded  ←  {path}")
    return model_dict


def _build_full_row(raw: dict) -> pd.DataFrame:
    """
    Accept raw human-readable fields and build the full
    27-column DataFrame that the scaler expects.
    Handles wind-direction encoding and date extraction internally.
    """
    row = dict(raw)  # copy

    # ── Wind direction encoding ───────────────────────────────────
    for wind_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        deg = WIND_MAPPING.get(row.pop(wind_col, None), 0.0)
        row[wind_col + '_sin'] = float(np.sin(np.deg2rad(deg)))
        row[wind_col + '_cos'] = float(np.cos(np.deg2rad(deg)))

    return pd.DataFrame([row])


def predict(model_dict: dict, input_df: pd.DataFrame) -> tuple:
    """
    Run inference.

    input_df can be:
      - A raw DataFrame with human-readable columns (wind dirs as strings)
      - A pre-processed DataFrame — both are handled

    Returns: (label str, probability float)
    """
    scaler       = model_dict["scaler"]
    selector     = model_dict["selector"]
    clf          = model_dict["lgb_clf"]
    all_features = model_dict["all_features"]

    df = input_df.copy()

    # Encode wind dirs if still raw strings
    for wind_col in ['WindGustDir', 'WindDir9am', 'WindDir3pm']:
        if wind_col in df.columns:
            deg = df[wind_col].map(WIND_MAPPING).fillna(0.0)
            df[wind_col + '_sin'] = np.sin(np.deg2rad(deg))
            df[wind_col + '_cos'] = np.cos(np.deg2rad(deg))
            df.drop(columns=[wind_col], inplace=True)

    # Fill any missing columns with 0
    for col in all_features:
        if col not in df.columns:
            df[col] = 0.0

    X          = df[all_features].copy()
    X_scaled   = scaler.transform(X)
    X_selected = selector.transform(X_scaled)

    pred  = clf.predict(X_selected)[0]
    prob  = clf.predict_proba(X_selected)[0][1]
    label = "Rain" if pred == 1 else "No Rain"

    return label, round(float(prob), 4)
