"""
Caso 2: Predicción de Potencial Comercial (Regresión).

Objetivo: estimar cuánto podría incrementar su facturación mensual un cliente
si se le ofrecen los servicios que aún no tiene contratados.

Variable objetivo (construida internamente):
    monthly_potential = percentil 75 de MonthlyCharges de clientes
    con mismo tipo de contrato e InternetService — MonthlyCharges actual.
    Valores <= 0 se truncan a 0 (clientes ya en el top del segmento).

Modelos: XGBoost Regressor (principal) + Ridge (baseline).
Métricas: MAE, RMSE, R².
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

RANDOM_STATE = 261
OUTPUT_DIR = Path("output/reports")
FIGURES_DIR = Path("output/figures")


# ---------------------------------------------------------------------------
# Target engineering
# ---------------------------------------------------------------------------

def build_commercial_target(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye la variable objetivo `monthly_potential` directamente desde el
    CSV original (columnas raw: MonthlyCharges, Contract, InternetService,
    TotalCharges, tenure, customerID).

    Retorna df con columnas de features + `monthly_potential`.
    """
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    # Percentil 75 de MonthlyCharges por segmento (Contract x InternetService)
    p75 = (
        df.groupby(["Contract", "InternetService"])["MonthlyCharges"]
        .transform(lambda x: x.quantile(0.75))
    )
    df["monthly_potential"] = (p75 - df["MonthlyCharges"]).clip(lower=0.0)

    return df


def preprocess_for_regression(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Feature engineering para el caso de regresión.
    Reutiliza la lógica base del caso de churn pero adapta el target.
    """
    df = build_commercial_target(df_raw)

    SERVICE_COLS = [
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
    ]
    YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]
    MULTI_CAT = [
        "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
        "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaymentMethod",
    ]

    for col in SERVICE_COLS:
        df[f"has_{col.lower()}"] = (df[col] == "Yes").astype(int)

    df["num_additional_services"] = df[[f"has_{c.lower()}" for c in SERVICE_COLS]].sum(axis=1)
    df["num_protection_services"] = df[
        ["has_onlinesecurity", "has_onlinebackup", "has_deviceprotection", "has_techsupport"]
    ].sum(axis=1)
    df["num_streaming_services"] = df[["has_streamingtv", "has_streamingmovies"]].sum(axis=1)

    for col in YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)
    df["gender"] = (df["gender"] == "Male").astype(int)

    # Features de facturación
    expected = df["tenure"] * df["MonthlyCharges"]
    df["charge_ratio"] = df["TotalCharges"] / (expected + 1.0)
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan))
    df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(df["MonthlyCharges"])
    df["log_total_charges"] = np.log1p(df["TotalCharges"])
    df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    df = pd.get_dummies(df, columns=MULTI_CAT, drop_first=True)

    drop_cols = ["customerID", "Churn", "SeniorCitizen"]
    customer_ids = df["customerID"] if "customerID" in df.columns else df.index
    df = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")

    return df, customer_ids


# ---------------------------------------------------------------------------
# Train / Evaluate
# ---------------------------------------------------------------------------

def train_regression_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float = 1.0,
) -> dict:
    models = {
        "Ridge_baseline": Ridge(alpha=1.0),
    }
    if XGBOOST_AVAILABLE:
        models["XGBoost_Regressor"] = XGBRegressor(
            n_estimators=300,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    trained = {}
    for name, model in models.items():
        print(f"  Entrenando {name} ...")
        model.fit(X_train, y_train)
        trained[name] = model
    return trained


def evaluate_regression(model, X_test, y_test, model_name: str = "model") -> dict:
    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    return {
        "Model": model_name,
        "MAE": round(mae, 4),
        "RMSE": round(rmse, 4),
        "R2": round(r2, 4),
    }


def build_potential_scoring(
    model,
    X: pd.DataFrame,
    customer_ids,
) -> pd.DataFrame:
    preds = model.predict(X).clip(min=0)
    df_out = pd.DataFrame({
        "customer_id": customer_ids,
        "monthly_potential_eur": np.round(preds, 2),
    })
    df_out["upsell_priority"] = pd.cut(
        df_out["monthly_potential_eur"],
        bins=[-0.01, 5.0, 20.0, np.inf],
        labels=["Low", "Medium", "High"],
    )
    df_out = df_out.sort_values("monthly_potential_eur", ascending=False).reset_index(drop=True)
    return df_out


def run(df_raw: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> dict:
    """Entry point. Devuelve dict con resultados y artefactos."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CASE 2: Potencial Comercial (Regresión) ===")

    df_feat, customer_ids = preprocess_for_regression(df_raw)
    target = "monthly_potential"
    y = df_feat[target]
    X = df_feat.drop(columns=[target])

    print(f"  Target: monthly_potential | media={y.mean():.2f} | mediana={y.median():.2f} | max={y.max():.2f}")
    print(f"  Clientes con potencial > 0: {(y > 0).sum()} ({(y > 0).mean()*100:.1f}%)")

    X_train, X_test, y_train, y_test, cid_train, cid_test = train_test_split(
        X, y, customer_ids, test_size=0.2, random_state=RANDOM_STATE
    )

    trained = train_regression_models(X_train, y_train)

    results = []
    for name, model in trained.items():
        m = evaluate_regression(model, X_test, y_test, model_name=name)
        results.append(m)
        print(f"  {name}: MAE={m['MAE']:.4f} | RMSE={m['RMSE']:.4f} | R²={m['R2']:.4f}")

    df_results = pd.DataFrame(results).sort_values("MAE")
    df_results.to_csv(output_dir / "case2_regression_metrics.csv", index=False)

    best_name = df_results.iloc[0]["Model"]
    best_model = trained[best_name]
    print(f"  Mejor modelo (MAE): {best_name}")

    scoring = build_potential_scoring(best_model, X, customer_ids)
    scoring.to_csv(output_dir / "case2_commercial_scoring.csv", index=False)
    print(f"  Scoring guardado en: {output_dir}/case2_commercial_scoring.csv")
    print(f"  Distribución prioridad upsell:\n{scoring['upsell_priority'].value_counts().to_string()}")

    return {
        "results": df_results,
        "trained": trained,
        "scoring": scoring,
        "best_name": best_name,
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
    }
