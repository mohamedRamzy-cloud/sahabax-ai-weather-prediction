# src/models.py
"""
Model definitions, training, and ensemble building.
"""

import lightgbm as lgb
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    BaggingClassifier,
    StackingClassifier,
)
from sklearn.neural_network import MLPClassifier

from src.config import (
    LGBM_PARAMS, RF_PARAMS, XGB_PARAMS, MLP_PARAMS, RANDOM_STATE
)


def get_all_models() -> dict:
    """Return a dict of {name: unfitted_model} for benchmark runs."""
    return {
        "Logistic Regression":    LogisticRegression(C=0.1, penalty="l2", solver="lbfgs", max_iter=1000),
        "SVM":                    SVC(C=1.0, kernel="linear", gamma="auto"),
        "Naive Bayes":            GaussianNB(),
        "Decision Tree":          DecisionTreeClassifier(
                                      criterion="entropy", max_depth=20,
                                      min_samples_split=7, min_samples_leaf=5,
                                      max_features="log2", class_weight="balanced",
                                      random_state=RANDOM_STATE),
        "Random Forest":          RandomForestClassifier(**RF_PARAMS),
        "Gradient Boosting":      GradientBoostingClassifier(
                                      n_estimators=100, learning_rate=0.1,
                                      max_depth=12, random_state=RANDOM_STATE),
        "XGBoost":                XGBClassifier(**XGB_PARAMS),
        "LightGBM":               lgb.LGBMClassifier(**LGBM_PARAMS),
        "MLP":                    MLPClassifier(**MLP_PARAMS),
    }


def build_lgbm() -> lgb.LGBMClassifier:
    return lgb.LGBMClassifier(**LGBM_PARAMS)


def build_stacking(xgb_model, bagging_model, lgb_model) -> StackingClassifier:
    """Stacking: XGB + Bagging + LightGBM  →  MLP meta-learner."""
    return StackingClassifier(
        estimators=[
            ("xgb",     xgb_model),
            ("bagging", bagging_model),
            ("lgb",     lgb_model),
        ],
        final_estimator=MLPClassifier(**MLP_PARAMS),
        cv=4,
        n_jobs=-1,
        stack_method="predict_proba",
    )


def build_bagging(base_model) -> BaggingClassifier:
    return BaggingClassifier(base_model, n_estimators=10,
                             random_state=RANDOM_STATE, n_jobs=-1)
