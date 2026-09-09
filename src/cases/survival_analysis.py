"""
Case 4: Survival analysis (tiempo hasta churn).

Usa:
- duration = tenure (meses)
- event    = Churn (1=evento, 0=censurado)

Modelos:
- Kaplan-Meier (curvas globales y por Contract)
- Cox Proportional Hazards (hazard ratios + riesgo a horizonte)

Caveat: dataset snapshot (no panel longitudinal). Los no-churners estan
right-censored en su tenure actual.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from lifelines import CoxPHFitter, KaplanMeierFitter
from lifelines.statistics import multivariate_logrank_test

RANDOM_STATE = 261
OUTPUT_DIR = Path("output/reports")
FIGURES_DIR = Path("output/figures")


def prepare_survival_frame(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Construye frame minimo para survival + covariables de Cox.
    tenure==0 se trata como 0.5 meses (clientes recien dados de alta).
    """
    df = df_raw.copy()
    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0.0)

    df["duration"] = df["tenure"].astype(float).replace(0, 0.5)
    df["event"] = (df["Churn"] == "Yes").astype(int)

    # Covariables simples e interpretables para Cox
    df["is_month_to_month"] = (df["Contract"] == "Month-to-month").astype(int)
    df["is_fiber"] = (df["InternetService"] == "Fiber optic").astype(int)
    df["is_senior"] = df["SeniorCitizen"].astype(int)
    df["has_partner"] = (df["Partner"] == "Yes").astype(int)
    df["has_dependents"] = (df["Dependents"] == "Yes").astype(int)
    df["paperless"] = (df["PaperlessBilling"] == "Yes").astype(int)
    df["electronic_check"] = (df["PaymentMethod"] == "Electronic check").astype(int)
    df["has_tech_support"] = (df["TechSupport"] == "Yes").astype(int)
    df["has_online_security"] = (df["OnlineSecurity"] == "Yes").astype(int)
    df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    return df


def fit_kaplan_meier(df: pd.DataFrame) -> dict:
    """KM global + por Contract."""
    km_global = KaplanMeierFitter(label="All customers")
    km_global.fit(df["duration"], event_observed=df["event"])

    km_by_contract = {}
    for contract, part in df.groupby("Contract"):
        km = KaplanMeierFitter(label=str(contract))
        km.fit(part["duration"], event_observed=part["event"])
        km_by_contract[str(contract)] = km

    # Log-rank entre contratos
    groups = df["Contract"].astype(str)
    logrank = multivariate_logrank_test(df["duration"], groups, df["event"])

    return {
        "km_global": km_global,
        "km_by_contract": km_by_contract,
        "logrank_pvalue": float(logrank.p_value),
        "logrank_test_statistic": float(logrank.test_statistic),
    }


def fit_cox(df: pd.DataFrame) -> tuple[CoxPHFitter, pd.DataFrame]:
    """
    Cox PH con penalizacion leve para estabilidad.
    Devuelve el fitter y tabla de hazard ratios.
    """
    cox_cols = [
        "duration",
        "event",
        "is_month_to_month",
        "is_fiber",
        "is_senior",
        "has_partner",
        "has_dependents",
        "paperless",
        "electronic_check",
        "has_tech_support",
        "has_online_security",
        "log_monthly_charges",
    ]
    data = df[cox_cols].copy()

    cph = CoxPHFitter(penalizer=0.01)
    cph.fit(data, duration_col="duration", event_col="event")

    summary = cph.summary.reset_index().rename(columns={"covariate": "feature"})
    # Compat: algunas versiones usan index name 'covariate'
    if "feature" not in summary.columns:
        summary = summary.rename(columns={summary.columns[0]: "feature"})

    hr_table = summary[[
        c for c in ["feature", "exp(coef)", "exp(coef) lower 95%", "exp(coef) upper 95%", "p"]
        if c in summary.columns
    ]].copy()
    hr_table = hr_table.rename(columns={
        "exp(coef)": "hazard_ratio",
        "exp(coef) lower 95%": "hr_ci_low",
        "exp(coef) upper 95%": "hr_ci_high",
        "p": "pvalue",
    })
    hr_table = hr_table.sort_values("hazard_ratio", ascending=False).reset_index(drop=True)
    return cph, hr_table


def build_survival_scoring(
    cph: CoxPHFitter,
    df: pd.DataFrame,
    horizons: tuple[int, ...] = (6, 12, 24),
) -> pd.DataFrame:
    """
    Para cada cliente, estima P(churn antes de t meses) = 1 - S(t|X).
    """
    cox_features = [
        "is_month_to_month",
        "is_fiber",
        "is_senior",
        "has_partner",
        "has_dependents",
        "paperless",
        "electronic_check",
        "has_tech_support",
        "has_online_security",
        "log_monthly_charges",
    ]
    X = df[cox_features]
    surv = cph.predict_survival_function(X, times=list(horizons))

    out = pd.DataFrame({"customer_id": df["customerID"].values})
    out["tenure"] = df["tenure"].values
    out["event_observed"] = df["event"].values
    for t in horizons:
        # surv index may be float
        col = surv.loc[t] if t in surv.index else surv.loc[float(t)]
        out[f"churn_prob_within_{t}m"] = np.round(1.0 - col.values, 4)

    # Prioridad operativa: riesgo a 12 meses
    out["survival_risk_tier"] = pd.cut(
        out["churn_prob_within_12m"],
        bins=[-0.01, 0.25, 0.50, 1.0],
        labels=["Low", "Medium", "High"],
    )
    out = out.sort_values("churn_prob_within_12m", ascending=False).reset_index(drop=True)
    return out


def median_survival_table(km_by_contract: dict) -> pd.DataFrame:
    rows = []
    for name, km in km_by_contract.items():
        med = km.median_survival_time_
        rows.append({
            "Contract": name,
            "median_survival_months": None if pd.isna(med) else float(med),
            "n": int(km.event_table["at_risk"].iloc[0]) if len(km.event_table) else None,
        })
    return pd.DataFrame(rows).sort_values("median_survival_months", na_position="last")


def run(df_raw: pd.DataFrame, output_dir: Path = OUTPUT_DIR) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== CASE 4: Survival Analysis (Kaplan-Meier + Cox PH) ===")
    print("  duration=tenure | event=Churn | right-censoring en no-churners")

    df = prepare_survival_frame(df_raw)
    print(f"  Clientes: {len(df)} | eventos: {df['event'].sum()} ({df['event'].mean()*100:.1f}%)")

    km = fit_kaplan_meier(df)
    print(f"  Log-rank (Contract): p={km['logrank_pvalue']:.2e}")

    medians = median_survival_table(km["km_by_contract"])
    print("  Mediana de supervivencia por Contract:")
    print(medians.to_string(index=False))
    medians.to_csv(output_dir / "survival_km_medians.csv", index=False)

    cph, hr_table = fit_cox(df)
    print("\n  Hazard ratios (Cox PH) — top factores de riesgo:")
    print(hr_table.head(8).to_string(index=False))
    hr_table.to_csv(output_dir / "survival_cox_hazard_ratios.csv", index=False)

    # Concordance / partial AIC
    print(f"  Concordance index: {cph.concordance_index_:.4f}")

    scoring = build_survival_scoring(cph, df)
    scoring.to_csv(output_dir / "survival_risk_scoring.csv", index=False)
    print(f"  Scoring guardado en: {output_dir}/survival_risk_scoring.csv")
    print("  Distribucion survival_risk_tier (riesgo a 12m):")
    print(scoring["survival_risk_tier"].value_counts().to_string())

    return {
        "df": df,
        "km": km,
        "medians": medians,
        "cph": cph,
        "hr_table": hr_table,
        "scoring": scoring,
    }
