"""
Test script for trained EV Charging Demand models:
- best_regression_model.pkl
- best_classification_model.pkl
- demand_threshold.pkl

Usage:
    python trained_models/test_models.py
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# -------------------------------------------------------------
# Path Resolution
# -------------------------------------------------------------
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(SCRIPT_DIR)
DATA_DIR = os.path.join(PROJECT_DIR, "data")
MODEL_DIR = SCRIPT_DIR

REG_MODEL_PATH = os.path.join(MODEL_DIR, "best_regression_model.pkl")
CLS_MODEL_PATH = os.path.join(MODEL_DIR, "best_classification_model.pkl")
THRESH_PATH = os.path.join(MODEL_DIR, "demand_threshold.pkl")

X_TEST_PATH = os.path.join(DATA_DIR, "X_test.csv")
Y_TEST_REG_PATH = os.path.join(DATA_DIR, "y_test_reg.csv")
Y_TEST_CLS_PATH = os.path.join(DATA_DIR, "y_test_cls.csv")


def load_artifacts():
    print("=" * 70)
    print("1. LOADING TRAINED MODELS & DATA")
    print("=" * 70)

    for path, name in [
        (REG_MODEL_PATH, "Regression Model"),
        (CLS_MODEL_PATH, "Classification Model"),
        (THRESH_PATH, "Demand Threshold"),
        (X_TEST_PATH, "Test Features (X_test)"),
        (Y_TEST_REG_PATH, "Test Targets (y_test_reg)"),
        (Y_TEST_CLS_PATH, "Test Targets (y_test_cls)"),
    ]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"Missing required file: {path} ({name})")

    reg_model = joblib.load(REG_MODEL_PATH)
    cls_model = joblib.load(CLS_MODEL_PATH)
    threshold = float(joblib.load(THRESH_PATH))

    X_test = pd.read_csv(X_TEST_PATH)
    y_test_reg = pd.read_csv(Y_TEST_REG_PATH).squeeze("columns")
    y_test_cls = pd.read_csv(Y_TEST_CLS_PATH).squeeze("columns")

    print(f" Loaded Regression Model:   {type(reg_model).__name__} (Pipeline)")
    print(f" Loaded Classification Model: {type(cls_model).__name__} (Pipeline)")
    print(f" Loaded High-Demand Cutoff:   {threshold:.3f} kWh")
    print(f" Loaded Test Set:             {X_test.shape[0]} rows, {X_test.shape[1]} features")

    return reg_model, cls_model, threshold, X_test, y_test_reg, y_test_cls


def test_regression(reg_model, X_test, y_test_reg):
    print("\n" + "=" * 70)
    print("2. REGRESSION MODEL BENCHMARK (best_regression_model.pkl)")
    print("=" * 70)

    y_pred = reg_model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test_reg, y_pred))
    mae = mean_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    print(f" Root Mean Squared Error (RMSE): {rmse:.4f} kWh")
    print(f" Mean Absolute Error (MAE):      {mae:.4f} kWh")
    print(f" R-squared Score (R2):           {r2:.4f}")

    # Check predictions sanity
    assert len(y_pred) == len(y_test_reg), "Prediction length mismatch!"
    assert not np.isnan(y_pred).any(), "Predictions contain NaN values!"
    print(" [PASS] Regression sanity assertions passed.")

    preview_df = pd.DataFrame({
        "Actual Demand (kWh)": y_test_reg.head(5).values,
        "Predicted Demand (kWh)": np.round(y_pred[:5], 2),
        "Absolute Error": np.round(np.abs(y_test_reg.head(5).values - y_pred[:5]), 2)
    })
    print("\nSample Predictions Preview (First 5 records):")
    print(preview_df.to_string(index=False))

    return {"RMSE": rmse, "MAE": mae, "R2": r2}


def test_classification(cls_model, X_test, y_test_cls):
    print("\n" + "=" * 70)
    print("3. CLASSIFICATION MODEL BENCHMARK (best_classification_model.pkl)")
    print("=" * 70)

    y_pred = cls_model.predict(X_test)
    y_prob = (
        cls_model.predict_proba(X_test)[:, 1]
        if hasattr(cls_model, "predict_proba")
        else None
    )

    acc = accuracy_score(y_test_cls, y_pred)
    prec = precision_score(y_test_cls, y_pred, zero_division=0)
    rec = recall_score(y_test_cls, y_pred, zero_division=0)
    f1 = f1_score(y_test_cls, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_test_cls, y_prob) if y_prob is not None else float("nan")

    print(f" Accuracy:  {acc:.4f} ({acc * 100:.2f}%)")
    print(f" Precision: {prec:.4f}")
    print(f" Recall:    {rec:.4f}")
    print(f" F1 Score:  {f1:.4f}")
    if y_prob is not None:
        print(f" ROC-AUC:   {roc_auc:.4f}")

    cm = confusion_matrix(y_test_cls, y_pred)
    print("\nConfusion Matrix:")
    print(f"   TN: {cm[0, 0]:<6} | FP: {cm[0, 1]:<6}")
    print(f"   FN: {cm[1, 0]:<6} | TP: {cm[1, 1]:<6}")

    # Check predictions sanity
    assert len(y_pred) == len(y_test_cls), "Prediction length mismatch!"
    assert set(np.unique(y_pred)).issubset({0, 1}), "Invalid binary class output!"
    if y_prob is not None:
        assert (y_prob >= 0.0).all() and (y_prob <= 1.0).all(), "Probabilities outside [0, 1]!"
    print(" [PASS] Classification sanity assertions passed.")

    return {"Accuracy": acc, "Precision": prec, "Recall": rec, "F1": f1, "ROC_AUC": roc_auc}


def test_custom_inference(reg_model, cls_model, threshold):
    print("\n" + "=" * 70)
    print("4. CUSTOM INFERENCE & SCENARIO TESTING")
    print("=" * 70)

    sample_scenarios = pd.DataFrame([
        {
            "Scenario": "Weekday Afternoon Fast Charger (High Activity)",
            "ChargerCompany": 1,
            "Location": "company",
            "ChargerType": 1,
            "StartHour": 14,
            "StartDayOfWeek": 2,  # Wednesday
            "StartMonth": 6,
            "StartYear": 2022,
            "IsWeekend": 0,
            "TimePeriod": "Afternoon",
        },
        {
            "Scenario": "Weekend Night Apartment Slow Charger (Low Activity)",
            "ChargerCompany": 0,
            "Location": "apartment",
            "ChargerType": 0,
            "StartHour": 2,
            "StartDayOfWeek": 6,  # Sunday
            "StartMonth": 1,
            "StartYear": 2022,
            "IsWeekend": 1,
            "TimePeriod": "Night",
        },
        {
            "Scenario": "Weekday Morning Public Station",
            "ChargerCompany": 1,
            "Location": "public area",
            "ChargerType": 1,
            "StartHour": 9,
            "StartDayOfWeek": 0,  # Monday
            "StartMonth": 10,
            "StartYear": 2022,
            "IsWeekend": 0,
            "TimePeriod": "Morning",
        },
    ])

    feature_cols = [
        "ChargerCompany", "Location", "ChargerType", "StartHour",
        "StartDayOfWeek", "StartMonth", "StartYear", "IsWeekend", "TimePeriod"
    ]

    for _, row in sample_scenarios.iterrows():
        scenario_name = row["Scenario"]
        input_df = pd.DataFrame([row[feature_cols].to_dict()])

        reg_pred = float(reg_model.predict(input_df)[0])
        cls_pred = int(cls_model.predict(input_df)[0])
        cls_prob = (
            float(cls_model.predict_proba(input_df)[0][1])
            if hasattr(cls_model, "predict_proba")
            else 0.0
        )

        category = "HIGH DEMAND" if cls_pred == 1 else "NORMAL DEMAND"
        print(f"\n Scenario: {scenario_name}")
        print(f"   - Predicted Demand:    {reg_pred:.2f} kWh (Threshold = {threshold:.2f} kWh)")
        print(f"   - Predicted Class:     {category} (Class {cls_pred})")
        print(f"   - High-Demand Prob:    {cls_prob * 100:.1f}%")

    print("\n" + "=" * 70)
    print(" ALL TESTS COMPLETED SUCCESSFULLY!")
    print("=" * 70)


def main():
    reg_model, cls_model, threshold, X_test, y_test_reg, y_test_cls = load_artifacts()
    test_regression(reg_model, X_test, y_test_reg)
    test_classification(cls_model, X_test, y_test_cls)
    test_custom_inference(reg_model, cls_model, threshold)


if __name__ == "__main__":
    main()
