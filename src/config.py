# src/config.py
"""
Central configuration for the Rain in Australia project.
Edit paths and hyperparameters here only.
"""

import os

# ── Paths ────────────────────────────────────────────────────────────
BASE_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH  = os.path.join(BASE_DIR, "data", "Rain in Australia", "weatherAUS.csv")
MODEL_PATH = os.path.join(BASE_DIR, "models", "rain_model.pkl")

# ── Data settings ─────────────────────────────────────────────────────
TARGET          = "RainTomorrow"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42
K_FEATURES      = 15

WIND_COLS       = ["WindGustDir", "WindDir9am", "WindDir3pm"]
OUTLIER_COLS    = ["Rainfall", "Evaporation", "WindSpeed9am", "WindSpeed3pm"]

WIND_MAPPING = {
    "N": 0,   "NNE": 22.5, "NE": 45,  "ENE": 67.5,
    "E": 90,  "ESE": 112.5,"SE": 135, "SSE": 157.5,
    "S": 180, "SSW": 202.5,"SW": 225, "WSW": 247.5,
    "W": 270, "WNW": 292.5,"NW": 315, "NNW": 337.5,
}

# ── Model hyperparameters ─────────────────────────────────────────────
SMOTE_STRATEGY  = 0.3

LGBM_PARAMS = {
    "n_estimators":  100,
    "max_depth":     -1,
    "learning_rate": 0.1,
    "random_state":  RANDOM_STATE,
}

RF_PARAMS = {
    "n_estimators":    300,
    "max_depth":       15,
    "min_samples_split": 5,
    "min_samples_leaf":  4,
    "max_features":    "sqrt",
    "class_weight":    "balanced",
    "random_state":    RANDOM_STATE,
}

XGB_PARAMS = {
    "n_estimators":    100,
    "max_depth":       10,
    "learning_rate":   0.1,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "eval_metric":     "logloss",
    "random_state":    RANDOM_STATE,
}

MLP_PARAMS = {
    "hidden_layer_sizes": (128, 64, 32),
    "activation":         "relu",
    "solver":             "adam",
    "alpha":              0.001,
    "learning_rate_init": 0.001,
    "max_iter":           1000,
    "early_stopping":     True,
    "random_state":       RANDOM_STATE,
}
