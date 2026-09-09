"""Unit tests rapidos (sin entrenamiento pesado)."""
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import load_data, validate_data
from src.features.engineering import preprocess, build_features, split_data
from src.models.lift import build_lift_table, summarize_lift
from src.cases.survival_analysis import prepare_survival_frame


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "telco_churn.csv"


def test_load_and_validate():
    df = load_data(DATA)
    report = validate_data(df)
    assert report["shape"][0] == 7043
    assert 20 < report["churn_rate_pct"] < 35


def test_feature_split_shapes():
    df = build_features(preprocess(load_data(DATA)))
    X_train, X_val, X_test, y_train, y_val, y_test, *_ = split_data(df)
    assert len(X_train) + len(X_val) + len(X_test) == len(df)
    assert abs(y_train.mean() - y_test.mean()) < 0.05
    assert "customerID" not in X_train.columns
    assert "Churn" not in X_train.columns


def test_lift_table_monotonic_customers():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=1000)
    # scores correlated with y
    proba = np.clip(y * 0.6 + rng.random(1000) * 0.4, 0, 1)
    lift = build_lift_table(y, proba, n_bins=10)
    assert len(lift) == 10
    assert lift["cumulative_customers"].is_monotonic_increasing
    assert lift.iloc[0]["lift"] >= lift.iloc[-1]["lift"]
    summary = summarize_lift(lift)
    assert "top_10pct" in summary


def test_survival_frame_event_rate():
    df = prepare_survival_frame(load_data(DATA))
    assert set(df["event"].unique()) <= {0, 1}
    assert (df["duration"] > 0).all()
    assert abs(df["event"].mean() - 0.265) < 0.02
