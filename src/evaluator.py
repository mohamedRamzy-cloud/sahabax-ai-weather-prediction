# src/evaluator.py
"""
Model evaluation: metrics, reports, confusion matrix plots.
"""

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
)


def evaluate(model, X, y, label: str = "Test") -> dict:
    """Works with any sklearn-compatible fitted model."""
    y_pred = model.predict(X)
    acc    = accuracy_score(y, y_pred)
    rep    = classification_report(y, y_pred, output_dict=True)

    print(f"\n{'='*45}")
    print(f"  {label} Results")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y, y_pred, target_names=["No Rain", "Rain"]))

    return {
        "label":     label,
        "accuracy":  acc,
        "precision": rep["weighted avg"]["precision"],
        "recall":    rep["weighted avg"]["recall"],
        "f1":        rep["weighted avg"]["f1-score"],
    }


def evaluate_pipeline_model(model_dict: dict, X: pd.DataFrame, y, label: str = "Test") -> dict:
    """
    Evaluate the dict-style pipeline model.
    X must contain ALL features the scaler was fit on (all_features).
    The function handles scale → select → predict internally.
    """
    all_features = model_dict.get("all_features", X.columns.tolist())

    X_full     = X[all_features]                               # ensure correct column order
    X_scaled   = model_dict["scaler"].transform(X_full)        # scale all features
    X_selected = model_dict["selector"].transform(X_scaled)    # select top-k
    y_pred     = model_dict["lgb_clf"].predict(X_selected)     # predict

    acc = accuracy_score(y, y_pred)
    rep = classification_report(y, y_pred, output_dict=True)

    print(f"\n{'='*45}")
    print(f"  {label} Results")
    print(f"{'='*45}")
    print(f"  Accuracy : {acc:.4f}")
    print(classification_report(y, y_pred, target_names=["No Rain", "Rain"]))

    return {
        "label":     label,
        "accuracy":  acc,
        "precision": rep["weighted avg"]["precision"],
        "recall":    rep["weighted avg"]["recall"],
        "f1":        rep["weighted avg"]["f1-score"],
    }


def benchmark_all_models(models: dict, X_train, y_train, X_test, y_test) -> pd.DataFrame:
    results = []
    for name, model in models.items():
        print(f"\n Training: {name}")
        model.fit(X_train, y_train)
        row = evaluate(model, X_test, y_test, label=name)
        results.append(row)

    df = pd.DataFrame(results).set_index("label").sort_values("f1", ascending=False)
    print("\n\n Model Comparison:")
    print(df.to_string())
    return df


def plot_confusion_matrix(model, X_test, y_test, title: str = "Confusion Matrix", save_path: str = None):
    y_pred = model.predict(X_test)
    cm     = confusion_matrix(y_test, y_pred)
    disp   = ConfusionMatrixDisplay(cm, display_labels=["No Rain", "Rain"])

    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    ax.set_title(title)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_model_comparison(results_df: pd.DataFrame, save_path: str = None):
    ax = results_df[["accuracy", "precision", "recall", "f1"]].plot(
        kind="bar", figsize=(14, 6), colormap="tab10", edgecolor="white"
    )
    ax.set_title("Model Comparison", fontsize=14)
    ax.set_ylabel("Score")
    ax.set_ylim(0.5, 1.0)
    ax.legend(loc="lower right")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()
