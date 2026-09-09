import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_data, validate_data
from src.features.engineering import preprocess, build_features, split_data
from src.models.train import (
    train_evaluate_with_calibration,
    tune_threshold_cost,
    build_cost_curve,
    build_churn_scoring,
    save_model,
)
from src.models.tuning import tune_all_models
from src.models.lift import run_lift_analysis
from src.cases import commercial_potential, anomaly_detection, unified_scoring, survival_analysis
from src.visualization.plots import (
    plot_churn_distribution,
    plot_numeric_by_churn,
    plot_categorical_churn_rate,
    plot_categorical_distribution,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curve,
    plot_cost_curve,
    plot_feature_importance,
    plot_shap_summary,
    plot_churn_score_distribution,
    plot_potential_scoring,
    plot_anomaly_scoring,
    plot_lift_gains,
    plot_unified_scoring,
    plot_kaplan_meier,
    plot_cox_hazard_ratios,
    plot_survival_risk_distribution,
)

DATA_PATH = Path("data/telco_churn.csv")
REPORTS_PATH = Path("output/reports/model_comparison.csv")
SCORING_PATH = Path("output/reports/churn_scoring.csv")


def main():
    print("=" * 60)
    print("  Pipeline de Churn — Deep Dive")
    print("=" * 60)

    # 1. Carga y validacion
    print("\n[1/9] Cargando datos...")
    df_raw = load_data(DATA_PATH)
    validate_data(df_raw)

    # 2. EDA
    print("\n[2/9] Generando visualizaciones EDA...")
    cat_cols = [
        "gender", "SeniorCitizen", "Partner", "Dependents",
        "PhoneService", "MultipleLines", "InternetService",
        "OnlineSecurity", "OnlineBackup", "DeviceProtection",
        "TechSupport", "StreamingTV", "StreamingMovies",
        "Contract", "PaperlessBilling", "PaymentMethod",
    ]
    plot_churn_distribution(df_raw)
    plot_numeric_by_churn(df_raw, num_cols=["tenure", "MonthlyCharges", "TotalCharges"])
    plot_categorical_distribution(df_raw, cat_cols=cat_cols)

    # 3. Preprocesamiento y features
    print("\n[3/9] Preprocesando y construyendo features...")
    df_processed = preprocess(df_raw)
    df_features = build_features(df_processed)
    X_train, X_val, X_test, y_train, y_val, y_test, cid_train, cid_val, cid_test = split_data(
        df_features, val_size=0.2, test_size=0.2
    )
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    print(f"  Desbalanceo: {y_train.mean()*100:.1f}% churn | scale_pos_weight = {scale_pos:.2f}")
    print(f"  (Un modelo naive con threshold 0.5 ignora este desbalanceo)")

    # 4. Tuning con CV anidado sobre train
    print("\n[4/9] Hyperparameter tuning (RandomizedSearchCV sobre train)...")
    tuned_models, tuning_summary = tune_all_models(
        X_train, y_train, scale_pos_weight=scale_pos,
        n_iter=20, cv=5,
    )
    print("\n  Resumen tuning:")
    print(tuning_summary.to_string(index=False))

    # 5. Evaluacion: calibracion + threshold
    print("\n[5/9] Evaluando modelos (calibracion + threshold por costes)...")
    results, trained_detail = train_evaluate_with_calibration(
        tuned_models,
        X_train, y_train,
        X_val, y_val,
        X_test, y_test,
    )

    results.to_csv(REPORTS_PATH, index=False)
    print(f"Reporte guardado en: {REPORTS_PATH}")
    print("\nResultados:")
    print(results.to_string(index=False))

    calibrated_models = {name: d["calibrated"] for name, d in trained_detail.items()}
    plot_roc_curves(calibrated_models, X_test, y_test)
    plot_pr_curves(calibrated_models, X_test, y_test)

    best_name = results.iloc[0]["Model"]
    best_calibrated_model = trained_detail[best_name]["calibrated"]
    best_base_model = trained_detail[best_name]["base"]
    print(f"\nMejor modelo: {best_name} (F1={results.iloc[0]['F1-Score']})")

    # Reliability / calibration curve
    plot_calibration_curve(best_calibrated_model, X_test, y_test, model_name=best_name)

    # Curva de costes: ilustra por que el desbalanceo importa y como el threshold
    # optimo se desplaza segun la relacion coste(FN) / coste(FP)
    best_proba_val = best_calibrated_model.predict_proba(X_val)[:, 1]
    cost_result = tune_threshold_cost(
        y_val, best_proba_val,
        cost_fn_fp=1.0,
        cost_fn_false_neg=5.0,
    )
    cost_df = build_cost_curve(
        y_val, best_proba_val,
        cost_fn_fp=1.0,
        cost_fn_false_neg=5.0,
    )
    print(f"\n  [Analisis de costes] coste FP=1 | coste FN=5")
    print(f"  Umbral optimo por coste: {cost_result['threshold']:.2f}")
    print(f"  Recall en umbral optimo: {cost_result['recall']:.3f}")
    print(f"  Precision en umbral optimo: {cost_result['precision']:.3f}")
    print(f"  Coste total esperado: {cost_result['expected_cost']:.0f}")
    plot_cost_curve(cost_df, optimal_threshold=cost_result["threshold"])

    # Lift / gains sobre test (holdout limpio)
    print("\n  [Lift / Gains] evaluacion de negocio en test...")
    best_proba_test = best_calibrated_model.predict_proba(X_test)[:, 1]
    lift_result = run_lift_analysis(
        y_test, best_proba_test, model_name=best_name, n_bins=10
    )
    plot_lift_gains(lift_result["lift_table"], model_name=best_name)

    if hasattr(best_base_model, "feature_importances_"):
        plot_feature_importance(best_base_model, X_train.columns.tolist(), model_name=best_name)
        plot_shap_summary(best_base_model, X_test, model_name=best_name)

    # 6. Scoring comercial churn (toda la base)
    print("\n[6/9] Generando scoring de churn...")
    X_all = df_features.drop(columns=["Churn", "customerID"])
    customer_ids_all = df_features["customerID"]
    scoring = build_churn_scoring(
        best_calibrated_model,
        X_all,
        customer_ids=customer_ids_all,
        threshold_medium=trained_detail[best_name]["risk_threshold_medium"],
        threshold_high=trained_detail[best_name]["risk_threshold_high"],
    )
    scoring.to_csv(SCORING_PATH, index=False)
    print(f"Scoring guardado en: {SCORING_PATH}")
    print(f"  Distribucion tiers: {scoring['risk_tier'].value_counts().to_dict()}")

    plot_churn_score_distribution(
        scoring,
        threshold_medium=trained_detail[best_name]["risk_threshold_medium"],
        threshold_high=trained_detail[best_name]["risk_threshold_high"],
    )

    for name, d in trained_detail.items():
        save_model(d["base"], name=f"{name}_base")
        save_model(d["calibrated"], name=f"{name}_calibrated")

    # 7. Casos adicionales
    print("\n[7/9] Ejecutando casos adicionales...")

    case2 = commercial_potential.run(df_raw)
    plot_potential_scoring(case2["scoring"])

    case3 = anomaly_detection.run(df_raw)
    plot_anomaly_scoring(case3["scoring"])

    # 8. Survival analysis
    print("\n[8/9] Survival analysis (Kaplan-Meier + Cox PH)...")
    case4 = survival_analysis.run(df_raw)
    plot_kaplan_meier(case4["km"]["km_global"], case4["km"]["km_by_contract"])
    plot_cox_hazard_ratios(case4["hr_table"])
    plot_survival_risk_distribution(case4["scoring"])

    # 9. Score comercial unificado
    print("\n[9/9] Construyendo score comercial unificado...")
    unified = unified_scoring.run(
        churn_scoring=scoring,
        potential_scoring=case2["scoring"],
        anomaly_scoring=case3["scoring"],
    )
    plot_unified_scoring(unified["scoring"])

    print("\n" + "=" * 60)
    print("  Pipeline completado. Resultados en output/")
    print("=" * 60)


if __name__ == "__main__":
    main()
