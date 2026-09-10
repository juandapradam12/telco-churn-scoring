#!/usr/bin/env python3
"""
CI smoke test: valida el camino critico del proyecto en < ~2-3 min.

No reproduce el pipeline completo (sin tuning exhaustivo ni SHAP),
pero comprueba que datos → features → modelo → lift → survival → scoring unificado funciona.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_data, validate_data
from src.features.engineering import preprocess, build_features, split_data
from src.models.train import (
    build_models,
    train_evaluate_with_calibration,
    build_churn_scoring,
)
from src.models.lift import run_lift_analysis
from src.cases import commercial_potential, anomaly_detection, unified_scoring, survival_analysis


def main() -> int:
    print("=== CI SMOKE TEST ===")
    data_path = ROOT / "data" / "telco_churn.csv"
    reports = ROOT / "output" / "reports"
    reports.mkdir(parents=True, exist_ok=True)

    df_raw = load_data(data_path)
    validate_data(df_raw)

    df_features = build_features(preprocess(df_raw))
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data(
        df_features, test_size=0.2, val_size=0.2
    )

    scale_pos = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    models = build_models(scale_pos_weight=scale_pos)

    # Acelerar modelos para CI
    if "RandomForest" in models:
        models["RandomForest"].set_params(n_estimators=50, max_depth=8, n_jobs=2)
    if "XGBoost" in models:
        models["XGBoost"].set_params(n_estimators=50, max_depth=3, n_jobs=2)

    # Solo RF (+ LR) para mantener el job corto
    slim = {k: models[k] for k in models if k in {"LogisticRegression", "RandomForest"}}
    results, detail = train_evaluate_with_calibration(
        slim, X_train, y_train, X_val, y_val, X_test, y_test
    )
    assert not results.empty, "model_comparison vacio"
    assert results.iloc[0]["F1-Score"] > 0.5, "F1 demasiado bajo en smoke"
    print(results[["Model", "F1-Score", "PR-AUC"]].to_string(index=False))

    best = results.iloc[0]["Model"]
    cal = detail[best]["calibrated"]
    proba = cal.predict_proba(X_test)[:, 1]
    lift = run_lift_analysis(y_test, proba, model_name=f"CI_{best}", n_bins=10, output_dir=reports)
    assert lift["summary"]["top_decile_lift"] > 1.5, "lift top decile inesperado"

    X_all = df_features.drop(columns=["Churn", "customerID"])
    churn_scoring = build_churn_scoring(
        cal,
        X_all,
        customer_ids=df_features["customerID"],
        threshold_medium=detail[best]["risk_threshold_medium"],
        threshold_high=detail[best]["risk_threshold_high"],
    )
    assert len(churn_scoring) == len(df_raw)

    case2 = commercial_potential.run(df_raw, output_dir=reports)
    case3 = anomaly_detection.run(df_raw, output_dir=reports)
    case4 = survival_analysis.run(df_raw, output_dir=reports)
    assert case4["cph"].concordance_index_ > 0.7

    unified = unified_scoring.run(
        churn_scoring, case2["scoring"], case3["scoring"], output_dir=reports
    )
    assert len(unified["scoring"]) == len(df_raw)
    assert {"Retain_HighValue", "Maintain"} & set(unified["scoring"]["commercial_segment"])

    print("=== CI SMOKE PASSED ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
