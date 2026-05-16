"""
Core training pipeline.

Pipeline order:
    load -> preprocess -> split -> SMOTE -> scale
    -> select -> train -> evaluate -> save
"""

import pandas as pd

from src.config import DATA_PATH, MODEL_PATH, K_FEATURES
from src.data_loader import load_data
from src.preprocessor import preprocess
from src.feature_engineering import split_data, apply_smote
from src.models import get_all_models, build_lgbm
from src.evaluator import evaluate_pipeline_model
from src.model_io import save_model

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2


def train_pipeline(
    data_path: str = DATA_PATH,
    model_path: str = MODEL_PATH,
    k_features: int = K_FEATURES,
    best_model_name: str = "LightGBM",
) -> dict:
    """
    Training pipeline steps:

    1. Load and preprocess data
    2. Split dataset
    3. Apply SMOTE
    4. Scale features
    5. Select best features
    6. Train selected model
    7. Evaluate model
    8. Save trained pipeline
    """

    # Load and preprocess data
    df = load_data(data_path)
    X, y = preprocess(df)

    # Split dataset
    X_train, X_test, y_train, y_test = split_data(X, y)

    all_features = X_train.columns.tolist()

    print(f"\nTotal Features: {len(all_features)}")

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Scale features
    scaler = MinMaxScaler()

    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(chi2, k=k_features)

    X_train_sel = selector.fit_transform(
        X_train_scaled,
        y_train_res
    )

    X_test_sel = selector.transform(X_test_scaled)

    selected_features = (
        X_train_res.columns[selector.get_support()].tolist()
    )

    print(
        f"Selected {k_features} Features: "
        f"{selected_features}"
    )

    # Select model
    all_models = get_all_models()

    if best_model_name not in all_models:

        print(
            f"Model '{best_model_name}' was not found. "
            f"Using LightGBM instead."
        )

        model = build_lgbm()

    else:
        model = all_models[best_model_name]

    print(f"\nTraining Model: {best_model_name} ...")

    model.fit(X_train_sel, y_train_res)

    model_dict = {
        "model_name": best_model_name,
        "scaler": scaler,
        "selector": selector,
        "lgb_clf": model,
        "features": selected_features,
        "all_features": all_features,
    }

    # Evaluate model
    evaluate_pipeline_model(
        model_dict,
        X_train,
        y_train,
        label="Train (Original Data)"
    )

    evaluate_pipeline_model(
        model_dict,
        X_test,
        y_test,
        label="Test"
    )

    # Save model
    save_model(model_dict, path=model_path)

    return model_dict