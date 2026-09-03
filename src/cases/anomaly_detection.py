"""
Caso 3: Detección de Anomalías en Facturación.

Objetivo: detectar clientes con patrones de facturación o uso inusuales
que podrían indicar:
- Errores de facturación (se pro-actúa antes de que el cliente reclame).
- Condiciones especiales no registradas en el CRM.
- Señales tempranas de churn que el modelo de clasificación todavía no capta.

Modelo principal: Isolation Forest.
Validación: Precision@K (de los K flaggeados, cuántos son churners reales) +
            distribución de anomalías por segmento de negocio.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 261
OUTPUT_DIR = Path("output/reports")
FIGURES_DIR = Path("output/figures")


# ---------------------------------------------------------------------------
# Feature engineering específico para anomalías
# ---------------------------------------------------------------------------

def build_anomaly_features(df_raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """
    Construye features orientadas a detectar comportamiento atípico en facturación.
    No usa la variable Churn como input (unsupervised), pero la devuelve aparte
    para validar Precision@K.
    """
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)
    y_churn = (df["Churn"] == "Yes").astype(int)
    customer_ids = df["customerID"]

    # Features de facturación
    expected = df["tenure"] * df["MonthlyCharges"]
    df["charge_ratio"] = df["TotalCharges"] / (expected + 1.0)
    df["deviation_from_expected"] = df["TotalCharges"] - expected
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan))
    df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(df["MonthlyCharges"])

    SERVICE_COLS = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    for col in SERVICE_COLS:
        df[f"has_{col.lower()}"] = (df[col] == "Yes").astype(int)
    df["num_additional_services"] = df[[f"has_{c.lower()}" for c in SERVICE_COLS]].sum(axis=1)

    # Relación entre servicios contratados y precio: si pagas mucho pero tienes pocos servicios, anómalo.
    df["price_per_service"] = df["MonthlyCharges"] / (df["num_additional_services"] + 1.0)

    # Tenure muy corto con facturación alta es un patrón sospechoso
    df["early_high_charge"] = (df["tenure"] <= 3).astype(int) * df["MonthlyCharges"]

    numeric_features = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "charge_ratio",
        "deviation_from_expected",
        "avg_monthly_charge",
        "num_additional_services",
        "price_per_service",
        "early_high_charge",
    ]

    X = df[numeric_features].copy()
    return X, y_churn, customer_ids


# ---------------------------------------------------------------------------
# Isolation Forest
# ---------------------------------------------------------------------------

def train_isolation_forest(
    X: pd.DataFrame,
    contamination: float = 0.05,
    n_estimators: int = 200,
) -> tuple[IsolationForest, StandardScaler]:
    scaler = StandardScaler()
    X_sc = scaler.fit_transform(X)

    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    model.fit(X_sc)
    return model, scaler


def precision_at_k(y_true: np.ndarray, scores: np.ndarray, k: int) -> float:
    """
    Precision@K: de los K clientes con mayor anomaly score (más negativos en Isolation Forest),
    qué fracción son churners reales.
    """
    top_k_idx = np.argsort(scores)[:k]
    return float(y_true[top_k_idx].mean())


def build_anomaly_scoring(
    model: IsolationForest,
    scaler: StandardScaler,
    X: pd.DataFrame,
    customer_ids,
    y_churn: pd.Series,
    contamination_percentile: float = 5.0,
) -> pd.DataFrame:
    X_sc = scaler.transform(X)
    raw_scores = model.score_samples(X_sc)        # negativo = más anómalo
    labels = model.predict(X_sc)                  # -1 = anomalía, 1 = normal

    # Normalizar a [0, 1]: 1 = más anómalo
    anomaly_score = 1.0 - (raw_scores - raw_scores.min()) / (raw_scores.max() - raw_scores.min() + 1e-9)

    df_out = pd.DataFrame({
        "customer_id": customer_ids.values if hasattr(customer_ids, "values") else customer_ids,
        "anomaly_score": np.round(anomaly_score, 4),
        "is_anomaly": (labels == -1).astype(int),
        "churn_real": y_churn.values,
    })
    df_out = df_out.sort_values("anomaly_score", ascending=False).reset_index(drop=True)
    return df_out, raw_scores


def run(df_raw: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> dict:
    """Entry point. Devuelve dict con artefactos y métricas."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CASE 3: Detección de Anomalías en Facturación (Isolation Forest) ===")

    X, y_churn, customer_ids = build_anomaly_features(df_raw)
    print(f"  Features: {list(X.columns)}")
    print(f"  Clientes: {len(X)}")

    contamination = 0.05
    model, scaler = train_isolation_forest(X, contamination=contamination)
    print(f"  Modelo entrenado (contamination={contamination})")

    scoring, raw_scores = build_anomaly_scoring(
        model, scaler, X, customer_ids, y_churn
    )

    n_anomalies = scoring["is_anomaly"].sum()
    print(f"  Anomalías detectadas: {n_anomalies} ({n_anomalies/len(scoring)*100:.1f}%)")

    # Validación: Precision@K
    for k in [50, 100, 200]:
        p_at_k = precision_at_k(scoring["churn_real"].values, raw_scores, k=k)
        print(f"  Precision@{k}: {p_at_k:.3f} (vs base rate {y_churn.mean():.3f})")

    # Comparar anomaly_score entre churners y no churners (lift)
    mean_score_churn = scoring.loc[scoring["churn_real"] == 1, "anomaly_score"].mean()
    mean_score_no_churn = scoring.loc[scoring["churn_real"] == 0, "anomaly_score"].mean()
    print(f"  Anomaly score medio churners: {mean_score_churn:.4f}")
    print(f"  Anomaly score medio no-churners: {mean_score_no_churn:.4f}")

    scoring.to_csv(output_dir / "case3_anomaly_scoring.csv", index=False)
    print(f"  Scoring guardado en: {output_dir}/case3_anomaly_scoring.csv")

    return {
        "model": model,
        "scaler": scaler,
        "scoring": scoring,
        "X": X,
        "y_churn": y_churn,
        "raw_scores": raw_scores,
    }
