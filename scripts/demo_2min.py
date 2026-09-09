#!/usr/bin/env python3
"""Mini demo de 2 minutos: gancho + lift + Cliente A vs B."""
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
REPORTS = ROOT / "output" / "reports"


def main():
    print("=" * 60)
    print("TELCO CHURN SCORING — 2-MIN DEMO")
    print("=" * 60)
    print(
        "\nHOOK\n"
        "Most sales teams call the wrong customers.\n"
        "This project ranks who to retain, upsell, or investigate.\n"
        "Top 20% of the list captures ~51% of churners (2.5× random).\n"
    )

    lift_path = REPORTS / "lift_table_RandomForest.csv"
    if not lift_path.exists():
        # fallback: any lift table
        cands = list(REPORTS.glob("lift_table_*.csv"))
        lift_path = cands[0] if cands else None

    if lift_path and lift_path.exists():
        lift = pd.read_csv(lift_path)
        row10 = lift.iloc[0]
        # fila mas cercana a 20% de clientes contactados
        row20 = lift.iloc[(lift["cumulative_customers"] - 0.20).abs().argmin()]
        print("LIFT (holdout)")
        print(
            f"  Top ~10%: capture {row10['cumulative_recall']*100:.1f}% churners "
            f"| lift {row10['lift']:.2f}x"
        )
        print(
            f"  Top ~{row20['cumulative_customers']*100:.0f}%: capture "
            f"{row20['cumulative_recall']*100:.1f}% churners "
            f"| cumulative lift {row20['cumulative_lift']:.2f}x"
        )

    print("\nCLIENT A vs B")
    print("  A 3750-CKVKH  | 2 months, fiber, month-to-month")
    print("    P(churn in 12m) ≈ 52%  →  Retain_HighValue  (urgent retain)")
    print("  B 9560-BBZXK  | 36 months, two-year contract")
    print("    P(churn in 12m) ≈  2%  →  Grow_Upsell       (don't waste retain budget)")
    print("\n  A = put out the fire.  B = sell more.\n")

    uni = REPORTS / "unified_commercial_scoring.csv"
    if uni.exists():
        df = pd.read_csv(uni)
        print("UNIFIED QUEUE (top 5)")
        cols = [
            c
            for c in [
                "customer_id",
                "commercial_priority_score",
                "commercial_segment",
                "churn_score",
                "monthly_potential_eur",
            ]
            if c in df.columns
        ]
        print(df[cols].head(5).to_string(index=False))

    print("\n" + "=" * 60)
    print("Run full story: notebooks/churn_analysis.ipynb | python3 main.py")
    print("Walkthrough:    docs/demo_2min.md")
    print("=" * 60)


if __name__ == "__main__":
    main()
