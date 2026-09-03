import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 261
OUTPUT_MODELS_DIR = Path("output/models")
OUTPUT_REPORTS_DIR = Path("output/reports")


def build_models(scale_pos_weight=1.0):
    # El dataset tiene ~26.5% churn (desbalanceo moderado).
    # Se usa class_weight="balanced" en sklearn y scale_pos_weight en XGBoost
    # para que el modelo penalice mas los errores en la clase minoritaria.
    models = {
        "LogisticRegression": LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=RANDOM_STATE,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=300,
            class_weight="balanced",
            max_depth=12,
            min_samples_leaf=5,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    else:
        models["GradientBoosting"] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=RANDOM_STATE,
        )

    return models


def train_models(models, X_train, y_train):
    trained = {}
    for name, model in models.items():
        print(f"  Entrenando {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def find_optimal_threshold(y_true, y_proba, thresholds=None):
    if thresholds is None:
        thresholds = np.arange(0.1, 0.91, 0.05)

    best_threshold = 0.5
    best_f1 = -1.0

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        score = f1_score(y_true, y_pred)
        if score > best_f1:
            best_f1 = score
            best_threshold = round(float(threshold), 2)

    return best_threshold, round(best_f1, 4)


def evaluate_model(model, X_test, y_test, model_name="model", threshold=0.5):
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_proba >= threshold).astype(int)

    metrics = {
        "model": model_name,
        "threshold": threshold,
        "f1_score": round(f1_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
        "pr_auc": round(average_precision_score(y_test, y_proba), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
        "y_proba": y_proba,
        "y_pred": y_pred,
    }

    return metrics


def compare_models(trained_models, X_test, y_test, save_path=None):
    results = []
    for name, model in trained_models.items():
        y_proba = model.predict_proba(X_test)[:, 1]
        optimal_threshold, optimal_f1 = find_optimal_threshold(y_test, y_proba)
        m = evaluate_model(model, X_test, y_test, model_name=name, threshold=optimal_threshold)
        results.append({
            "Model": name,
            "F1-Score": m["f1_score"],
            "AUC-ROC": m["roc_auc"],
            "PR-AUC": m["pr_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "Optimal-Threshold": optimal_threshold,
            "F1-at-default-0.5": round(
                f1_score(y_test, (y_proba >= 0.5).astype(int)), 4
            ),
        })

    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"Reporte guardado en: {save_path}")

    return df_results


def build_churn_scoring(model, X, customer_ids=None):
    proba = model.predict_proba(X)[:, 1]

    scoring = pd.DataFrame({
        "customer_id": customer_ids if customer_ids is not None else X.index,
        "churn_score": np.round(proba, 4),
    })

    scoring["risk_tier"] = pd.cut(
        scoring["churn_score"],
        bins=[0, 0.3, 0.6, 1.0],
        labels=["Low", "Medium", "High"],
    )

    action_map = {
        "High":   "Visita urgente - oferta de retencion personalizada",
        "Medium": "Contacto proactivo - revision de contrato",
        "Low":    "Mantenimiento - comunicacion periodica",
    }
    scoring["recommended_action"] = scoring["risk_tier"].map(action_map)
    scoring = scoring.sort_values("churn_score", ascending=False).reset_index(drop=True)

    return scoring


def save_model(model, name, output_dir=OUTPUT_MODELS_DIR):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / f"{name}.pkl"
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"Modelo guardado: {path}")
    return path


def load_model(name, output_dir=OUTPUT_MODELS_DIR):
    path = Path(output_dir) / f"{name}.pkl"
    with open(path, "rb") as f:
        model = pickle.load(f)
    print(f"Modelo cargado: {path}")
    return model
