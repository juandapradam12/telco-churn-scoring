import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    f1_score,
    roc_auc_score,
    precision_score,
    recall_score,
    confusion_matrix,
    classification_report,
    average_precision_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    # Mejoras futuras: threshold tuning, SMOTE, o calibracion de probabilidades.
    models = {
        "LogisticRegression": Pipeline(
            steps=[
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=2000,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
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


def expected_calibration_error(y_true, y_proba, n_bins: int = 10) -> float:
    """
    ECE (Expected Calibration Error) por bin de probabilidad.
    y_true: 0/1, y_proba: P(y=1).
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    total = len(y_true)

    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        if i == n_bins - 1:
            mask = (y_proba >= lo) & (y_proba <= hi)
        else:
            mask = (y_proba >= lo) & (y_proba < hi)

        if not np.any(mask):
            continue

        bin_prob = mask.mean()
        bin_conf = y_proba[mask].mean()
        bin_acc = y_true[mask].mean()
        ece += np.abs(bin_acc - bin_conf) * bin_prob * (total / total)

    return float(ece)


def tune_threshold(
    y_true,
    y_proba,
    threshold_grid=None,
    recall_target: float | None = None,
    objective: str = "f1",
    fallback_objective: str = "f1",
):
    """
    Selecciona un umbral usando un grid en [0,1].
    - objective: "f1" o "precision"
    - recall_target: si se define, solo se consideran umbrales con recall >= recall_target.
    """
    y_true = np.asarray(y_true)
    y_proba = np.asarray(y_proba)

    if threshold_grid is None:
        threshold_grid = np.linspace(0.02, 0.98, 97)

    best = None
    best_score = -np.inf

    for t in threshold_grid:
        y_pred = (y_proba >= t).astype(int)
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        if recall_target is not None and rec < recall_target:
            continue

        if objective == "precision":
            score = prec
        elif objective == "threshold":
            score = t
        else:
            score = f1

        if score > best_score:
            best_score = score
            best = {"threshold": float(t), "precision": float(prec), "recall": float(rec), "f1": float(f1)}

    # Si no hay umbral que cumpla recall_target, caemos al mejor por F1 sin restricciones.
    if best is None:
        # Si no se puede cumplir el recall_target, usamos el fallback para evitar umbrales extremos.
        return tune_threshold(
            y_true,
            y_proba,
            threshold_grid=threshold_grid,
            recall_target=None,
            objective=fallback_objective,
            fallback_objective=fallback_objective,
        )

    return best


def train_evaluate_with_calibration(
    models,
    X_train,
    y_train,
    X_val,
    y_val,
    X_test,
    y_test,
    recall_target_f1: float = 0.8,
    risk_recall_medium: float = 0.75,
    risk_recall_high: float = 0.6,
    calibration_method: str = "sigmoid",
    threshold_grid=None,
    n_bins_ece: int = 10,
):
    """
    Entrena modelos base, calibra probabilidades en Val y selecciona umbrales:
    - threshold_opt: umbral para maximizar F1 (con constraint de recall_target_f1)
    - threshold_medium/threshold_high: cortes de risk tier alineados a recall deseado.
    """
    results = []
    trained_detail = {}

    for name, base_model in models.items():
        print(f"  Entrenando {name} (base) ...")
        base_model.fit(X_train, y_train)

        print(f"  Calibrando {name} con Val ...")
        # En sklearn moderno, `cv="prefit"` puede no estar disponible.
        # Calibramos usando un split explícito train->val:
        # el estimador base se entrena en train y la calibración se aprende con val.
        X_combined = pd.concat([X_train, X_val], axis=0)
        y_combined = pd.concat([y_train, y_val], axis=0)
        n_train = len(X_train)
        val_indices = np.arange(n_train, n_train + len(X_val))
        train_indices = np.arange(0, n_train)
        cv_split = [(train_indices, val_indices)]

        calibrated = CalibratedClassifierCV(base_model, method=calibration_method, cv=cv_split)
        calibrated.fit(X_combined, y_combined)

        val_proba = calibrated.predict_proba(X_val)[:, 1]
        test_proba = calibrated.predict_proba(X_test)[:, 1]

        tuned = tune_threshold(
            y_val,
            val_proba,
            threshold_grid=threshold_grid,
            recall_target=recall_target_f1,
            objective="f1",
        )

        tuned_medium = tune_threshold(
            y_val,
            val_proba,
            threshold_grid=threshold_grid,
            recall_target=risk_recall_medium,
            objective="threshold",
            fallback_objective="f1",
        )
        tuned_high = tune_threshold(
            y_val,
            val_proba,
            threshold_grid=threshold_grid,
            recall_target=risk_recall_high,
            objective="threshold",
            fallback_objective="f1",
        )

        # Garantizamos orden (high >= medium)
        threshold_medium = tuned_medium["threshold"]
        threshold_high = max(tuned_high["threshold"], threshold_medium)

        # Evaluacion en test con umbral optimizado
        y_pred_test = (test_proba >= tuned["threshold"]).astype(int)

        metrics = {
            "Model": name,
            "F1-Score": round(f1_score(y_test, y_pred_test, zero_division=0), 4),
            "AUC-ROC": round(roc_auc_score(y_test, test_proba), 4),
            "PR-AUC": round(average_precision_score(y_test, test_proba), 4),
            "Precision": round(precision_score(y_test, y_pred_test, zero_division=0), 4),
            "Recall": round(recall_score(y_test, y_pred_test, zero_division=0), 4),
            "BrierScore": round(brier_score_loss(y_test, test_proba), 4),
            "ECE": round(expected_calibration_error(y_test, test_proba, n_bins=n_bins_ece), 4),
            "threshold_opt": round(tuned["threshold"], 4),
            "risk_threshold_medium": round(threshold_medium, 4),
            "risk_threshold_high": round(threshold_high, 4),
            "calibration_method": calibration_method,
        }

        results.append(metrics)
        trained_detail[name] = {
            "base": base_model,
            "calibrated": calibrated,
            "threshold_opt": tuned["threshold"],
            "risk_threshold_medium": threshold_medium,
            "risk_threshold_high": threshold_high,
        }

    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False).reset_index(drop=True)
    return df_results, trained_detail


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
        "brier_score": round(brier_score_loss(y_test, y_proba), 4),
        "ece": round(expected_calibration_error(y_test, y_proba), 4),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, output_dict=True),
    }

    return metrics


def compare_models(trained_models, X_test, y_test, save_path=None):
    results = []
    for name, model in trained_models.items():
        m = evaluate_model(model, X_test, y_test, model_name=name)
        results.append({
            "Model": name,
            "F1-Score": m["f1_score"],
            "AUC-ROC": m["roc_auc"],
            "Precision": m["precision"],
            "Recall": m["recall"],
        })

    df_results = pd.DataFrame(results).sort_values("F1-Score", ascending=False)

    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        df_results.to_csv(save_path, index=False)
        print(f"Reporte guardado en: {save_path}")

    return df_results


def build_churn_scoring(
    model,
    X,
    customer_ids=None,
    threshold_medium: float = 0.3,
    threshold_high: float = 0.6,
):
    proba = model.predict_proba(X)[:, 1]

    scoring = pd.DataFrame({
        "customer_id": customer_ids if customer_ids is not None else X.index,
        "churn_score": np.round(proba, 4),
    })

    threshold_medium = float(threshold_medium)
    threshold_high = float(max(threshold_high, threshold_medium))

    scoring["risk_tier"] = np.select(
        [
            scoring["churn_score"] >= threshold_high,
            (scoring["churn_score"] >= threshold_medium) & (scoring["churn_score"] < threshold_high),
        ],
        ["High", "Medium"],
        default="Low",
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
