import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_data, validate_data
from src.features.engineering import preprocess, build_features, split_data
from src.models.train import (
    build_models,
    train_evaluate_with_calibration,
    build_churn_scoring,
    save_model,
)
from src.visualization.plots import (
    plot_churn_distribution,
    plot_numeric_by_churn,
    plot_categorical_churn_rate,
    plot_categorical_distribution,
    plot_roc_curves,
    plot_pr_curves,
    plot_calibration_curve,
    plot_feature_importance,
    plot_shap_summary,
    plot_churn_score_distribution,
)

DATA_PATH = Path("data/telco_churn.csv")
REPORTS_PATH = Path("output/reports/model_comparison.csv")
SCORING_PATH = Path("output/reports/churn_scoring.csv")


def main():
    print("--- Pipeline de Clasificacion Churn ---")

    # 1. Carga
    print("\n[1/6] Cargando datos...")
    df_raw = load_data(DATA_PATH)
    validate_data(df_raw)

    # 2. EDA
    print("\n[2/6] Generando visualizaciones EDA...")
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
    print("\n[3/6] Preprocesando y construyendo features...")
    df_processed = preprocess(df_raw)
    df_features = build_features(df_processed)
    X_train, X_val, X_test, y_train, y_val, y_test, cid_train, cid_val, cid_test = split_data(
        df_features, val_size=0.2, test_size=0.2
    )

    # 4. Entrenamiento
    print("\n[4/6] Entrenando modelos...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(scale_pos_weight=scale_pos)

    # 5. Evaluacion
    print("\n[5/6] Evaluando modelos...")
    results, trained_detail = train_evaluate_with_calibration(
        models,
        X_train,
        y_train,
        X_val,
        y_val,
        X_test,
        y_test,
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

    # Reliability / calibration curve (importante porque el scoring de negocio asume probabilidad calibrada)
    plot_calibration_curve(best_calibrated_model, X_test, y_test, model_name=best_name)

    if hasattr(best_base_model, "feature_importances_"):
        plot_feature_importance(best_base_model, X_train.columns.tolist(), model_name=best_name)
        plot_shap_summary(best_base_model, X_test, model_name=best_name)

    # 6. Scoring comercial
    print("\n[6/6] Generando scoring de riesgo...")
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
    print("\nTop 10 clientes en riesgo alto:")
    print(scoring[scoring["risk_tier"] == "High"].head(10).to_string(index=False))

    plot_churn_score_distribution(
        scoring,
        threshold_medium=trained_detail[best_name]["risk_threshold_medium"],
        threshold_high=trained_detail[best_name]["risk_threshold_high"],
    )

    for name, d in trained_detail.items():
        save_model(d["base"], name=f"{name}_base")
        save_model(d["calibrated"], name=f"{name}_calibrated")

    print("\nPipeline completado. Resultados en la carpeta output/")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
