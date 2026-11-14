from __future__ import annotations

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from src.data import OPENING_BLOCK

RANDOM_STATE = 42


class OpeningBlockBlend(ClassifierMixin, BaseEstimator):
    def __init__(self, ignore_fraction=OPENING_BLOCK, C=1.0, max_iter=1000):
        self.ignore_fraction = ignore_fraction
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y, sample_weight=None):
        weights = np.ones(len(y), dtype=float)
        if sample_weight is not None:
            weights *= np.asarray(sample_weight, dtype=float)
        weights[: int(len(y) * self.ignore_fraction)] = 0.0
        self.model_ = LogisticRegression(C=self.C, max_iter=self.max_iter, random_state=RANDOM_STATE).fit(
            X, y, sample_weight=weights
        )
        self.classes_ = self.model_.classes_
        self.coef_ = self.model_.coef_
        self.intercept_ = self.model_.intercept_
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict(self, X):
        return self.model_.predict(X)


def make_trees(scale_pos_weight: float = 1.0) -> XGBClassifier:
    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        n_estimators=300,
        max_depth=3,
        learning_rate=0.05,
        subsample=0.85,
        colsample_bytree=0.8,
        min_child_weight=5,
        max_delta_step=1,
        reg_lambda=1.0,
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=1,
        verbosity=0,
    )


def make_geometric() -> Pipeline:
    return Pipeline(
        [
            ("scale", StandardScaler()),
            ("svc", SVC(kernel="rbf", class_weight="balanced", cache_size=1000, random_state=RANDOM_STATE)),
        ]
    )


CANDIDATES = {
    "XGBoost": (
        make_trees,
        {
            "n_estimators": [200, 400],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.03, 0.08],
            "min_child_weight": [1, 5],
        },
    ),
    "SVM (RBF)": (make_geometric, {"svc__C": [0.3, 1.0, 3.0, 10.0], "svc__gamma": ["scale", 0.03, 0.1, 0.3]}),
}


def build_stack(trees, geometric, folds, ignore_fraction=OPENING_BLOCK) -> StackingClassifier:
    return StackingClassifier(
        estimators=[("trees", trees), ("geometric", geometric)],
        final_estimator=OpeningBlockBlend(ignore_fraction=ignore_fraction),
        stack_method="auto",
        cv=folds,
        n_jobs=1,
        passthrough=False,
    )


def positive_scores(model, X) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:, 1]
    return model.decision_function(X)
