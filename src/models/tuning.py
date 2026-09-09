"""
Hyperparameter tuning con RandomizedSearchCV sobre train exclusivamente.
Se entrena y afina sobre train; val y test quedan intocados.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import make_scorer, f1_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 261
_f1_scorer = make_scorer(f1_score, zero_division=0)


def _param_grids(scale_pos_weight: float = 1.0) -> dict:
    grids = {
        "LogisticRegression": {
            "model__C": [0.001, 0.01, 0.1, 0.5, 1.0, 5.0, 10.0],
            "model__solver": ["lbfgs", "saga"],
            "model__penalty": ["l2"],
        },
        "RandomForest": {
            "n_estimators": [100, 200, 400],
            "max_depth": [6, 9, 12, None],
            "min_samples_leaf": [3, 5, 10],
            "max_features": ["sqrt", "log2"],
        },
    }
    if XGBOOST_AVAILABLE:
        grids["XGBoost"] = {
            "n_estimators": [100, 200, 400],
            "max_depth": [3, 5, 7],
            "learning_rate": [0.01, 0.05, 0.1, 0.2],
            "subsample": [0.7, 0.8, 1.0],
            "colsample_bytree": [0.7, 0.8, 1.0],
            "min_child_weight": [1, 3, 5],
        }
    return grids


def build_base_estimators(scale_pos_weight: float = 1.0) -> dict:
    estimators = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                ("model", LogisticRegression(
                    class_weight="balanced",
                    max_iter=2000,
                    random_state=RANDOM_STATE,
                )),
            ]
        ),
        "RandomForest": RandomForestClassifier(
            class_weight="balanced",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }
    if XGBOOST_AVAILABLE:
        estimators["XGBoost"] = XGBClassifier(
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    return estimators


def tune_all_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 1.0,
    n_iter: int = 30,
    cv: int = 5,
    n_jobs: int = -1,
    verbose: int = 0,
) -> dict:
    """
    RandomizedSearchCV sobre train con CV=5 estratificado.
    Devuelve dict name -> best_estimator_.

    Nota: usa solo X_train/y_train para no contaminar val ni test.
    """
    grids = _param_grids(scale_pos_weight)
    estimators = build_base_estimators(scale_pos_weight)
    cv_strategy = StratifiedKFold(n_splits=cv, shuffle=True, random_state=RANDOM_STATE)

    tuned = {}
    summary_rows = []

    for name, estimator in estimators.items():
        print(f"  [Tuning] {name} — {n_iter} combinaciones x {cv} folds ...")
        search = RandomizedSearchCV(
            estimator=estimator,
            param_distributions=grids[name],
            n_iter=n_iter,
            scoring=_f1_scorer,
            cv=cv_strategy,
            refit=True,
            n_jobs=n_jobs,
            random_state=RANDOM_STATE,
            verbose=verbose,
        )
        search.fit(X_train, y_train)
        tuned[name] = search.best_estimator_
        summary_rows.append({
            "Model": name,
            "best_cv_f1": round(search.best_score_, 4),
            "best_params": str(search.best_params_),
        })
        print(f"    best CV F1 = {search.best_score_:.4f} | params: {search.best_params_}")

    return tuned, pd.DataFrame(summary_rows)
