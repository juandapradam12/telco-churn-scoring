import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


SERVICE_COLS_RAW = [
    "OnlineSecurity", "OnlineBackup", "DeviceProtection",
    "TechSupport", "StreamingTV", "StreamingMovies",
]

MULTI_CAT_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", "OnlineBackup",
    "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies",
    "Contract", "PaymentMethod",
]

YES_NO_COLS = ["Partner", "Dependents", "PhoneService", "PaperlessBilling"]


def preprocess(df):
    df = df.copy()

    df["Churn"] = (df["Churn"] == "Yes").astype(int)

    # Feature engineering "CRM-like" antes del one-hot:
    # convierte servicios en indicadores binarios (captura si el cliente tiene/usa el servicio).
    for col in SERVICE_COLS_RAW:
        df[f"has_{col.lower()}"] = (df[col] == "Yes").astype(int)

    df["num_additional_services"] = df[[f"has_{c.lower()}" for c in SERVICE_COLS_RAW]].sum(axis=1)
    df["num_protection_services"] = df[
        ["has_onlinesecurity", "has_onlinebackup", "has_deviceprotection", "has_techsupport"]
    ].sum(axis=1)
    df["num_streaming_services"] = df[["has_streamingtv", "has_streamingmovies"]].sum(axis=1)

    for col in YES_NO_COLS:
        df[col] = (df[col] == "Yes").astype(int)

    df["gender"] = (df["gender"] == "Male").astype(int)

    df = pd.get_dummies(df, columns=MULTI_CAT_COLS, drop_first=True)

    return df


def build_features(df):
    df = df.copy()

    # Ratio entre lo que ha pagado y lo que deberia haber pagado segun tenure
    expected = df["tenure"] * df["MonthlyCharges"]
    df["charge_ratio"] = df["TotalCharges"] / (expected + 1)

    # Desviacion (útil para detectar "facturacion inconsistente"/descuentos)
    df["expected_total_charges"] = expected
    df["deviation_from_expected"] = df["TotalCharges"] - expected

    # Promedio real por mes (evita spuriousidades cuando tenure es bajo)
    df["avg_monthly_charge"] = df["TotalCharges"] / (df["tenure"].replace(0, np.nan))
    df["avg_monthly_charge"] = df["avg_monthly_charge"].fillna(df["MonthlyCharges"])

    # Normalizacion adicional (log protege de colas)
    df["log_total_charges"] = np.log1p(df["TotalCharges"])
    df["log_monthly_charges"] = np.log1p(df["MonthlyCharges"])

    # Numero de servicios adicionales contratados
    # preferimos la variable "limpia" (num_additional_services) creada en preprocess.
    if "num_additional_services" in df.columns:
        df["num_services"] = df["num_additional_services"]
    else:
        service_cols = [c for c in df.columns if any(s in c for s in SERVICE_COLS_RAW)]
        df["num_services"] = df[service_cols].sum(axis=1)

    return df


def split_data(
    df,
    target: str = "Churn",
    test_size: float = 0.2,
    val_size: float = 0.2,
    random_state: int = 261,
):
    """
    Particion estratificada train/val/test, preservando customerID para scoring.
    """
    y = df[target]
    X = df.drop(columns=[target])

    customer_id = X["customerID"] if "customerID" in X.columns else X.index
    if "customerID" in X.columns:
        X = X.drop(columns=["customerID"])

    X_trainval, X_test, y_trainval, y_test, cid_trainval, cid_test = train_test_split(
        X, y, customer_id, test_size=test_size, stratify=y, random_state=random_state
    )

    # val_size se interpreta sobre el train+val (restante tras test)
    val_fraction = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, cid_train, cid_val = train_test_split(
        X_trainval,
        y_trainval,
        cid_trainval,
        test_size=val_fraction,
        stratify=y_trainval,
        random_state=random_state,
    )

    print(f"Train: {X_train.shape[0]} muestras | Churn: {y_train.mean()*100:.1f}%")
    print(f"Val:   {X_val.shape[0]} muestras | Churn: {y_val.mean()*100:.1f}%")
    print(f"Test:  {X_test.shape[0]} muestras  | Churn: {y_test.mean()*100:.1f}%")

    return X_train, X_val, X_test, y_train, y_val, y_test, cid_train, cid_val, cid_test
