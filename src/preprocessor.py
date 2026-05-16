# src/preprocessor.py
"""
All data-cleaning and feature-engineering logic.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from src.config import TARGET, WIND_COLS, WIND_MAPPING, OUTLIER_COLS


# ── Memory optimiser (sklearn-compatible) ─────────────────────────────

class MemoryOptimizer(BaseEstimator, TransformerMixin):
    """Downcast numeric dtypes to reduce RAM usage."""

    def __init__(self, verbose: bool = False):
        self.verbose = verbose

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        df = X.copy() if isinstance(X, pd.DataFrame) else pd.DataFrame(X)
        start = df.memory_usage(deep=True).sum() / 1024 ** 2

        for col in df.columns:
            dtype = df[col].dtype
            unique = set(df[col].dropna().unique())

            if unique.issubset({0, 1}):
                df[col] = df[col].astype(bool)
                continue

            if np.issubdtype(dtype, np.number):
                lo, hi = df[col].min(), df[col].max()
                if np.issubdtype(dtype, np.integer):
                    if lo >= 0:
                        if   hi < 255:        df[col] = df[col].astype(np.uint8)
                        elif hi < 65_535:     df[col] = df[col].astype(np.uint16)
                        elif hi < 4_294_967_295: df[col] = df[col].astype(np.uint32)
                        else:                 df[col] = df[col].astype(np.uint64)
                    else:
                        info = np.iinfo
                        if   lo > info(np.int8).min  and hi < info(np.int8).max:  df[col] = df[col].astype(np.int8)
                        elif lo > info(np.int16).min and hi < info(np.int16).max: df[col] = df[col].astype(np.int16)
                        elif lo > info(np.int32).min and hi < info(np.int32).max: df[col] = df[col].astype(np.int32)
                        else:                                                      df[col] = df[col].astype(np.int64)
                else:
                    df[col] = df[col].astype(np.float32)

        end = df.memory_usage(deep=True).sum() / 1024 ** 2
        if self.verbose:
            print(f"🔹 Memory  {start:.2f} MB  →  {end:.2f} MB  (−{100*(start-end)/start:.1f}%)")
        return df


# ── Individual helpers ────────────────────────────────────────────────

def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.columns:
        if df[col].dtype == "O":
            df[col] = df[col].fillna(df[col].mode()[0])
        else:
            df[col] = df[col].fillna(df[col].mean())
    return df


def _extract_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df["Date"] = pd.to_datetime(df["Date"])
    df["year"]  = df["Date"].dt.year
    df["month"] = df["Date"].dt.month
    df["day"]   = df["Date"].dt.day
    return df.drop("Date", axis=1)


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    df["RainToday"]    = df["RainToday"].map({"No": 0, "Yes": 1})
    df["RainTomorrow"] = df["RainTomorrow"].map({"No": 0, "Yes": 1})
    df["Location"]     = df["Location"].map(df["Location"].value_counts())
    return df


def _encode_wind(df: pd.DataFrame) -> pd.DataFrame:
    for col in WIND_COLS:
        deg = df[col].map(WIND_MAPPING)
        df[col + "_sin"] = np.sin(np.deg2rad(deg))
        df[col + "_cos"] = np.cos(np.deg2rad(deg))
    return df.drop(WIND_COLS, axis=1)


def _cap_outliers(df: pd.DataFrame) -> pd.DataFrame:
    for col in OUTLIER_COLS:
        Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        IQR = Q3 - Q1
        df[col] = np.clip(df[col], Q1 - 1.5 * IQR, Q3 + 1.5 * IQR)
    return df


# ── Public API ────────────────────────────────────────────────────────

def preprocess(df: pd.DataFrame, target: str = TARGET):
    """
    Full preprocessing pipeline.
    Returns X (DataFrame) and y (Series).
    """
    df = df.copy()
    df = _fill_missing(df)
    df = _extract_date_features(df)
    df = _encode_categoricals(df)
    df = _encode_wind(df)
    df = _cap_outliers(df)
    df = MemoryOptimizer(verbose=True).fit_transform(df)

    X = df.drop(columns=[target])
    y = df[target]
    print(f" Preprocessing done  →  X: {X.shape}, y: {y.shape}")
    return X, y
