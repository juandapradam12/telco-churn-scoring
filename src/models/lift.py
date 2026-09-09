"""
Analisis de lift / gains / deciles para scoring de churn.

Responde a la pregunta de negocio:
"Si contactamos el top X% del score, cuantos churners capturamos vs aleatorio?"
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUTPUT_DIR = Path("output/reports")


def build_lift_table(
    y_true,
    y_proba,
    n_bins: int = 10,
) -> pd.DataFrame:
    """
    Construye tabla de deciles ordenados por score descendente.

    Columnas:
    - decile: 1 = top 10% mas riesgoso
    - n_customers
    - n_churners
    - churn_rate
    - lift: churn_rate / base_rate
    - cumulative_recall: % de churners capturados hasta ese decile
    - cumulative_customers: % de clientes contactados
    - cumulative_lift: cumulative_recall / cumulative_customers
    """
    y_true = np.asarray(y_true).astype(int)
    y_proba = np.asarray(y_proba).astype(float)

    df = pd.DataFrame({"y": y_true, "score": y_proba}).sort_values(
        "score", ascending=False
    ).reset_index(drop=True)

    n = len(df)
    base_rate = df["y"].mean()
    # qcut puede fallar con ties; usamos ranking + cortes equiespaciados
    df["rank"] = np.arange(n)
    edges = np.linspace(0, n, n_bins + 1).astype(int)
    labels = list(range(1, n_bins + 1))
    df["decile"] = pd.cut(df["rank"], bins=edges, labels=labels, include_lowest=True, right=False)

    rows = []
    cum_churners = 0
    cum_customers = 0
    total_churners = int(df["y"].sum())

    for d in labels:
        part = df[df["decile"] == d]
        n_cust = len(part)
        n_churn = int(part["y"].sum())
        churn_rate = n_churn / max(n_cust, 1)
        cum_churners += n_churn
        cum_customers += n_cust

        cum_recall = cum_churners / max(total_churners, 1)
        cum_pct = cum_customers / max(n, 1)

        rows.append({
            "decile": d,
            "n_customers": n_cust,
            "n_churners": n_churn,
            "churn_rate": round(churn_rate, 4),
            "base_rate": round(float(base_rate), 4),
            "lift": round(churn_rate / max(base_rate, 1e-9), 4),
            "cumulative_customers": round(cum_pct, 4),
            "cumulative_recall": round(cum_recall, 4),
            "cumulative_lift": round(cum_recall / max(cum_pct, 1e-9), 4),
        })

    return pd.DataFrame(rows)


def summarize_lift(lift_df: pd.DataFrame) -> dict:
    """Resumen ejecutivo de lift en top 10% / 20% / 30%."""
    summary = {}
    for pct, label in [(0.10, "top_10pct"), (0.20, "top_20pct"), (0.30, "top_30pct")]:
        # primer decile cuyo cumulative_customers >= pct
        row = lift_df.loc[lift_df["cumulative_customers"] >= pct - 1e-9].iloc[0]
        summary[label] = {
            "customers_contacted_pct": float(row["cumulative_customers"]),
            "churners_captured_pct": float(row["cumulative_recall"]),
            "cumulative_lift": float(row["cumulative_lift"]),
            "decile_lift": float(row["lift"]),
        }
    summary["top_decile_lift"] = float(lift_df.iloc[0]["lift"])
    summary["top_decile_churn_rate"] = float(lift_df.iloc[0]["churn_rate"])
    return summary


def run_lift_analysis(
    y_true,
    y_proba,
    model_name: str = "model",
    n_bins: int = 10,
    output_dir: Path = OUTPUT_DIR,
) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    lift_df = build_lift_table(y_true, y_proba, n_bins=n_bins)
    summary = summarize_lift(lift_df)

    out_path = output_dir / f"lift_table_{model_name}.csv"
    lift_df.to_csv(out_path, index=False)

    print(f"  Lift table guardada en: {out_path}")
    print(f"  Top decile lift: {summary['top_decile_lift']:.2f}x "
          f"(churn rate={summary['top_decile_churn_rate']:.1%})")
    print(f"  Top 10%: captura {summary['top_10pct']['churners_captured_pct']:.1%} "
          f"de churners (lift acumulado={summary['top_10pct']['cumulative_lift']:.2f}x)")
    print(f"  Top 20%: captura {summary['top_20pct']['churners_captured_pct']:.1%} "
          f"de churners (lift acumulado={summary['top_20pct']['cumulative_lift']:.2f}x)")
    print(f"  Top 30%: captura {summary['top_30pct']['churners_captured_pct']:.1%} "
          f"de churners (lift acumulado={summary['top_30pct']['cumulative_lift']:.2f}x)")

    return {"lift_table": lift_df, "summary": summary, "path": out_path}
