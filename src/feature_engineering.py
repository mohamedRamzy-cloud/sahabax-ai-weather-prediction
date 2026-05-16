# src/feature_engineering.py
"""
Scaling, feature selection, and SMOTE oversampling.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing     import MinMaxScaler
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.model_selection   import train_test_split
from imblearn.over_sampling    import SMOTE

from src.config import K_FEATURES, RANDOM_STATE, SMOTE_STRATEGY, TEST_SIZE


def split_data(X: pd.DataFrame, y: pd.Series, test_size: float = TEST_SIZE):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=RANDOM_STATE, stratify=y
    )
    print(f"✅ Split  →  train: {X_train.shape}, test: {X_test.shape}")
    return X_train, X_test, y_train, y_test


def apply_smote(X_train: pd.DataFrame, y_train: pd.Series, strategy: float = SMOTE_STRATEGY):
    """Apply SMOTE on RAW (unscaled) train data."""
    smote = SMOTE(sampling_strategy=strategy, random_state=RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    # Keep column names after resample
    X_res = pd.DataFrame(X_res, columns=X_train.columns)
    print(f" SMOTE  →  {X_res.shape}, class counts: {dict(y_res.value_counts())}")
    return X_res, y_res


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    k: int = K_FEATURES,
):
    """
    Scale with MinMaxScaler then select top-k features using chi2.
    Used in benchmark.py (where SMOTE is optional).
    NOTE: In the main pipeline (trainer.py), SMOTE runs BEFORE this step.
    """
    scaler   = MinMaxScaler()
    selector = SelectKBest(chi2, k=k)

    X_train_scaled   = scaler.fit_transform(X_train)
    X_train_selected = selector.fit_transform(X_train_scaled, y_train)

    X_test_scaled    = scaler.transform(X_test)
    X_test_selected  = selector.transform(X_test_scaled)

    feature_names = X_train.columns[selector.get_support()].tolist()
    print(f" Selected {k} features: {feature_names}")

    return (
        pd.DataFrame(X_train_selected, columns=feature_names),
        pd.DataFrame(X_test_selected,  columns=feature_names),
        scaler,
        selector,
        feature_names,
    )


def plot_correlation(X: pd.DataFrame, y: pd.Series):
    corr    = X.apply(lambda col: col.corr(pd.Series(y.values, index=X.index)))
    corr_df = corr.to_frame("Correlation_with_RainTomorrow")

    plt.figure(figsize=(6, 10))
    sns.heatmap(corr_df, annot=True, cmap="coolwarm", center=0)
    plt.title("Feature Correlation with RainTomorrow")
    plt.tight_layout()
    plt.savefig("outputs/correlation_heatmap.png", dpi=150)
    plt.show()


def plot_class_distribution(y: pd.Series):
    count = y.value_counts()
    plt.figure(figsize=(5, 5))
    plt.pie(count.values, labels=["No Rain", "Rain"],
            autopct="%1.1f%%", startangle=90, colors=["skyblue", "lightcoral"])
    plt.title("Class Distribution — RainTomorrow")
    plt.tight_layout()
    plt.savefig("outputs/class_distribution.png", dpi=150)
    plt.show()
