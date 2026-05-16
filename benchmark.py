"""
Compare all models and save benchmark results.
"""

import os
import json
import pandas as pd
import matplotlib.pyplot as plt

from src.data_loader import load_data
from src.preprocessor import preprocess
from src.feature_engineering import split_data, apply_smote
from src.models import get_all_models
from src.evaluator import (
    plot_confusion_matrix,
    plot_model_comparison
)

from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import accuracy_score, classification_report

os.makedirs("outputs", exist_ok=True)

RESULTS_PATH = "outputs/benchmark_results.json"


def run_benchmark():

    # Load and preprocess data
    df = load_data()
    X, y = preprocess(df)

    # Split data
    X_train, X_test, y_train, y_test = split_data(X, y)
    all_features = X_train.columns.tolist()

    # Apply SMOTE
    X_train_res, y_train_res = apply_smote(X_train, y_train)

    # Scale data
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train_res)
    X_test_scaled = scaler.transform(X_test)

    # Feature selection
    selector = SelectKBest(chi2, k=15)
    X_train_sel = selector.fit_transform(X_train_scaled, y_train_res)
    X_test_sel = selector.transform(X_test_scaled)

    selected_features = (
        X_train_res.columns[selector.get_support()].tolist()
    )

    print(f"\nSelected Features: {selected_features}\n")

    # Train and evaluate models
    models = get_all_models()
    results = []

    print("=" * 55)
    print("MODEL BENCHMARK")
    print("=" * 55)

    fitted_models = {}

    for name, model in models.items():

        print(f"\nTraining Model: {name} ...")

        model.fit(X_train_sel, y_train_res)

        y_pred = model.predict(X_test_sel)

        rep = classification_report(
            y_pred=y_pred,
            y_true=y_test,
            output_dict=True
        )

        acc = accuracy_score(y_test, y_pred)

        row = {
            "model": name,
            "accuracy": round(acc, 4),
            "precision": round(rep["weighted avg"]["precision"], 4),
            "recall": round(rep["weighted avg"]["recall"], 4),
            "f1": round(rep["weighted avg"]["f1-score"], 4),
        }

        results.append(row)
        fitted_models[name] = model

        print(f"Accuracy: {acc:.4f}  |  F1 Score: {row['f1']:.4f}")

    results_df = (
        pd.DataFrame(results)
        .set_index("model")
        .sort_values("f1", ascending=False)
    )

    print("\n\n" + "=" * 55)
    print("MODEL RANKING")
    print("=" * 55)

    print(results_df.to_string())

    best_name = results_df.index[0]
    best_f1 = results_df.loc[best_name, "f1"]
    best_acc = results_df.loc[best_name, "accuracy"]

    print(f"\nBest Model : {best_name}")
    print(f"F1 Score   : {best_f1}")
    print(f"Accuracy   : {best_acc}")

    save_data = {
        "best_model": best_name,
        "best_f1": best_f1,
        "best_accuracy": best_acc,
        "all_features": all_features,
        "selected_features": selected_features,
        "ranking": (
            results_df
            .reset_index()
            .to_dict(orient="records")
        ),
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump(save_data, f, ensure_ascii=False, indent=2)

    print(f"\nResults saved to: {RESULTS_PATH}")

    plot_confusion_matrix(
        fitted_models[best_name],
        X_test_sel,
        y_test,
        title=f"Confusion Matrix - {best_name}",
        save_path="outputs/best_model_confusion_matrix.png",
    )

    plot_model_comparison(
        results_df,
        save_path="outputs/model_comparison.png"
    )

    print("\nBenchmark completed successfully!")
    print("Next step: python train.py")


if __name__ == "__main__":
    run_benchmark()