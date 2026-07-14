"""
predict.py
================================================================
Inference layer for the Burnout Risk Prediction app.
Loads the trained Random Forest model + encoders and exposes a
single clean function `predict_burnout_risk()` used by app.py.
================================================================
"""

import os
import pickle

import numpy as np
import pandas as pd

from preprocessing import FEATURE_COLUMNS, TARGET_COLUMN, encode_single_input

ARTIFACTS_DIR = "artifacts"

RISK_COLORS = {
    "Low": "#22C55E",     # green
    "Medium": "#F59E0B",  # amber
    "High": "#EF4444",    # red
}

RISK_RECOMMENDATIONS = {
    "Low": [
        "Keep up your current balance between GenAI tools and independent study.",
        "Continue practicing core skills manually so retention stays strong.",
        "Check in periodically — habits can drift toward dependency over a semester.",
    ],
    "Medium": [
        "Set a weekly cap on GenAI usage hours and track it for a few weeks.",
        "Increase traditional (unassisted) study time, especially before exams.",
        "Practice a few problems fully by hand each week to protect skill retention.",
        "If exam anxiety is a factor, consider short breathing or mindfulness breaks before study sessions.",
    ],
    "High": [
        "Reduce reliance on GenAI for direct answers — shift toward using it only for ideation or review.",
        "Talk to an academic advisor or counselor about workload and exam-related anxiety.",
        "Rebuild fundamentals with dedicated unassisted practice sessions, even in small daily doses.",
        "Prioritize sleep, breaks, and pacing — burnout risk compounds when stress is left unmanaged.",
        "Consider a structured study plan that reintroduces traditional study habits gradually.",
    ],
}


class ModelLoadError(Exception):
    pass


def _artifact_path(name: str) -> str:
    return os.path.join(ARTIFACTS_DIR, name)


def load_artifacts():
    """Load model, encoders and feature column order from disk.
    Raises a friendly ModelLoadError if artifacts are missing."""
    model_path = _artifact_path("model.pkl")
    encoder_path = _artifact_path("encoder.pkl")
    columns_path = _artifact_path("feature_columns.pkl")

    for p in [model_path, encoder_path, columns_path]:
        if not os.path.exists(p):
            raise ModelLoadError(
                f"Required artifact not found: '{p}'. "
                f"Please run 'python train_model.py' first to train and save the model."
            )

    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(encoder_path, "rb") as f:
        encoders = pickle.load(f)
    with open(columns_path, "rb") as f:
        feature_columns = pickle.load(f)

    return model, encoders, feature_columns


def predict_burnout_risk(raw_input: dict, model=None, encoders=None):
    """
    Predict burnout risk for a single student.

    Parameters
    ----------
    raw_input : dict
        Raw feature values keyed by the original column names
        (e.g. {"Major_Category": "STEM", "Weekly_GenAI_Hours": 12.5, ...})
    model, encoders : optional
        Pass already-loaded artifacts to avoid re-reading from disk
        (useful with Streamlit caching).

    Returns
    -------
    dict with keys: prediction, confidence, probabilities, color,
    recommendations
    """
    if model is None or encoders is None:
        model, encoders, _ = load_artifacts()

    feature_encoders = {k: v for k, v in encoders.items() if k != TARGET_COLUMN}
    target_encoder = encoders[TARGET_COLUMN]

    try:
        X_row = encode_single_input(raw_input, feature_encoders)
    except Exception as exc:
        raise ValueError(f"Invalid input for prediction: {exc}")

    pred_encoded = model.predict(X_row)[0]
    proba = model.predict_proba(X_row)[0]

    pred_label = target_encoder.inverse_transform([pred_encoded])[0]
    class_labels = target_encoder.inverse_transform(np.arange(len(proba)))
    prob_dict = {label: float(p) for label, p in zip(class_labels, proba)}
    confidence = float(np.max(proba))

    return {
        "prediction": pred_label,
        "confidence": confidence,
        "probabilities": prob_dict,
        "color": RISK_COLORS.get(pred_label, "#6B7280"),
        "recommendations": RISK_RECOMMENDATIONS.get(pred_label, []),
    }
