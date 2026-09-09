"""
Score comercial unificado.

Combina:
- churn_score (riesgo de abandono, calibrado)
- monthly_potential_eur (oportunidad de upsell)
- anomaly_score (senal de facturacion atipica)

En un ranking operativo con action playbooks por segmento.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output/reports")


def _minmax(series: pd.Series) -> pd.Series:
    lo, hi = series.min(), series.max()
    if hi - lo < 1e-12:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - lo) / (hi - lo)


def build_unified_scoring(
    churn_scoring: pd.DataFrame,
    potential_scoring: pd.DataFrame,
    anomaly_scoring: pd.DataFrame,
    w_churn: float = 0.50,
    w_potential: float = 0.30,
    w_anomaly: float = 0.20,
) -> pd.DataFrame:
    """
    Une los tres scorings por customer_id y construye:

    - commercial_priority_score in [0,1]
    - commercial_segment: playbook de accion
    - priority_tier: High / Medium / Low sobre el score unificado

    Pesos por defecto reflejan prioridad de retencion (churn) sobre upsell/anomalia.
    """
    churn = churn_scoring[["customer_id", "churn_score", "risk_tier"]].copy()
    pot = potential_scoring[["customer_id", "monthly_potential_eur", "upsell_priority"]].copy()
    anom = anomaly_scoring[["customer_id", "anomaly_score", "is_anomaly"]].copy()

    df = (
        churn.merge(pot, on="customer_id", how="inner")
        .merge(anom, on="customer_id", how="inner")
    )

    df["churn_norm"] = _minmax(df["churn_score"])
    df["potential_norm"] = _minmax(df["monthly_potential_eur"])
    df["anomaly_norm"] = _minmax(df["anomaly_score"])

    w_sum = w_churn + w_potential + w_anomaly
    df["commercial_priority_score"] = (
        w_churn * df["churn_norm"]
        + w_potential * df["potential_norm"]
        + w_anomaly * df["anomaly_norm"]
    ) / w_sum

    df["commercial_priority_score"] = df["commercial_priority_score"].round(4)

    # Segmentos de playbook (matriz churn x potencial x anomalia)
    high_churn = df["risk_tier"] == "High"
    med_churn = df["risk_tier"] == "Medium"
    high_pot = df["upsell_priority"] == "High"
    is_anom = df["is_anomaly"] == 1

    conditions = [
        high_churn & high_pot,
        high_churn & is_anom,
        high_churn,
        med_churn & high_pot,
        (~high_churn) & (~med_churn) & high_pot,
        is_anom,
    ]
    labels = [
        "Retain_HighValue",      # alto churn + alto potencial
        "Retain_InvestigateBill", # alto churn + anomalia
        "Retain_Urgent",         # alto churn
        "Nurture_Upsell",        # medio churn + alto potencial
        "Grow_Upsell",           # bajo churn + alto potencial
        "Investigate_Billing",   # anomalia sin churn alto
    ]
    df["commercial_segment"] = np.select(conditions, labels, default="Maintain")

    action_map = {
        "Retain_HighValue": "Retencion urgente + oferta personalizada (cliente de alto valor)",
        "Retain_InvestigateBill": "Retencion urgente + revision de facturacion",
        "Retain_Urgent": "Visita urgente - oferta de retencion",
        "Nurture_Upsell": "Contacto proactivo + propuesta de ampliacion de servicios",
        "Grow_Upsell": "Campana de upsell / cross-sell",
        "Investigate_Billing": "Revision de facturacion / condiciones especiales",
        "Maintain": "Mantenimiento - comunicacion periodica",
    }
    df["recommended_action"] = df["commercial_segment"].map(action_map)

    # Tiers sobre score unificado (percentiles)
    p70 = df["commercial_priority_score"].quantile(0.70)
    p90 = df["commercial_priority_score"].quantile(0.90)
    df["priority_tier"] = np.select(
        [
            df["commercial_priority_score"] >= p90,
            df["commercial_priority_score"] >= p70,
        ],
        ["High", "Medium"],
        default="Low",
    )

    df = df.sort_values("commercial_priority_score", ascending=False).reset_index(drop=True)
    return df


def run(
    churn_scoring: pd.DataFrame,
    potential_scoring: pd.DataFrame,
    anomaly_scoring: pd.DataFrame,
    output_dir: Path = OUTPUT_DIR,
    w_churn: float = 0.50,
    w_potential: float = 0.30,
    w_anomaly: float = 0.20,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n=== SCORE COMERCIAL UNIFICADO ===")
    print(f"  Pesos: churn={w_churn:.2f} | potential={w_potential:.2f} | anomaly={w_anomaly:.2f}")

    unified = build_unified_scoring(
        churn_scoring,
        potential_scoring,
        anomaly_scoring,
        w_churn=w_churn,
        w_potential=w_potential,
        w_anomaly=w_anomaly,
    )

    out_path = output_dir / "unified_commercial_scoring.csv"
    unified.to_csv(out_path, index=False)

    print(f"  Scoring guardado en: {out_path}")
    print("  Distribucion priority_tier:")
    print(unified["priority_tier"].value_counts().to_string())
    print("  Distribucion commercial_segment:")
    print(unified["commercial_segment"].value_counts().to_string())
    print("\n  Top 10 por prioridad comercial:")
    cols = [
        "customer_id",
        "commercial_priority_score",
        "priority_tier",
        "commercial_segment",
        "churn_score",
        "monthly_potential_eur",
        "anomaly_score",
    ]
    print(unified[cols].head(10).to_string(index=False))

    return {"scoring": unified, "path": out_path}
