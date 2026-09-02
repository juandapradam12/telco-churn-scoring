import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data.loader import load_data, validate_data
from src.features.engineering import preprocess, build_features, split_data
from src.models.train import (
    build_models,
    train_models,
    compare_models,
    evaluate_model,
    build_churn_scoring,
    save_model,
)
from src.visualization.plots import (
    plot_churn_distribution,
    plot_numeric_by_churn,
    plot_categorical_distribution,
    plot_confusion_matrix,
    plot_roc_curves,
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
    X_train, X_test, y_train, y_test, scaler = split_data(df_features)

    # 4. Entrenamiento
    print("\n[4/6] Entrenando modelos...")
    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()
    models = build_models(scale_pos_weight=scale_pos)
    trained_models = train_models(models, X_train, y_train)

    # 5. Evaluacion
    print("\n[5/6] Evaluando modelos...")
    results = compare_models(trained_models, X_test, y_test, save_path=REPORTS_PATH)
    print("\nResultados:")
    print(results.to_string(index=False))

    plot_roc_curves(trained_models, X_test, y_test)

    best_name = results.iloc[0]["Model"]
    best_model = trained_models[best_name]
    best_threshold = float(results.iloc[0]["Optimal-Threshold"])
    print(
        f"\nMejor modelo: {best_name} "
        f"(F1={results.iloc[0]['F1-Score']}, umbral={best_threshold})"
    )

    best_metrics = evaluate_model(
        best_model, X_test, y_test, model_name=best_name, threshold=best_threshold
    )
    plot_confusion_matrix(
        y_test,
        best_metrics["y_pred"],
        model_name=best_name,
    )

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, X_train.columns.tolist(), model_name=best_name)
        plot_shap_summary(best_model, X_test, model_name=best_name)

    # 6. Scoring comercial
    print("\n[6/6] Generando scoring de riesgo...")
    scoring = build_churn_scoring(best_model, X_test, customer_ids=X_test.index)
    scoring.to_csv(SCORING_PATH, index=False)
    print(f"Scoring guardado en: {SCORING_PATH}")
    print("\nTop 10 clientes en riesgo alto:")
    print(scoring[scoring["risk_tier"] == "High"].head(10).to_string(index=False))

    plot_churn_score_distribution(scoring)

    for name, model in trained_models.items():
        save_model(model, name=name)

    print("\nPipeline completado. Resultados en la carpeta output/")
    print("---------------------------------------")


if __name__ == "__main__":
    main()
