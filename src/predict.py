# src/predict.py
"""
Safe prediction helpers for the MMM model.
"""

import sys
import os
# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import joblib
import yaml
import pandas as pd
import numpy as np

from src.feature_engineering import prepare_features
from src.etl import load_and_clean

PIPELINE_FILE = os.path.join(ROOT_DIR, "models", "mmm_pipeline.pkl")
META_FILE = os.path.join(ROOT_DIR, "models", "model_metadata.yaml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

def load_pipeline_and_meta():
    if not os.path.exists(PIPELINE_FILE):
        raise FileNotFoundError("Trained pipeline not found. Run src.train_model first.")
    pipeline = joblib.load(PIPELINE_FILE)
    if not os.path.exists(META_FILE):
        raise FileNotFoundError("Model metadata not found. Run training first.")
    meta = yaml.safe_load(open(META_FILE))
    return pipeline, meta

def _ensure_X_for_pipeline(df: pd.DataFrame, meta: dict) -> pd.DataFrame:
    feature_cols = meta["numeric_features"] + meta["categorical_features"]
    X = df.reindex(columns=feature_cols).copy()

    # Force numerics
    for c in meta["numeric_features"]:
        if c in X.columns:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
        else:
            X[c] = 0.0

    # Force categoricals
    for c in meta["categorical_features"]:
        if c in X.columns:
            X[c] = X[c].fillna("unknown").astype(str)
        else:
            X[c] = "unknown"

    return X[feature_cols]

def predict_latest(n_rows=1):
    pipeline, meta = load_pipeline_and_meta()
    df = load_and_clean()
    df = prepare_features(df)
    if len(df) < n_rows:
        raise ValueError("Not enough rows for prediction.")
    latest = df.tail(n_rows).copy().reset_index(drop=True)
    X = _ensure_X_for_pipeline(latest, meta)
    preds = pipeline.predict(X)
    return {"predicted_revenue": float(preds[0]), "input_row": latest.iloc[0].to_dict()}

def channel_contributions_finite_diff(delta=100.0, channels=None):
    if channels is None:
        channels = ["spend_google", "spend_facebook", "spend_influencer"]
    pipeline, meta = load_pipeline_and_meta()
    df = load_and_clean()
    df = prepare_features(df)
    latest = df.tail(1).copy().reset_index(drop=True)
    X_base = _ensure_X_for_pipeline(latest, meta)
    base_pred = float(pipeline.predict(X_base)[0])
    contributions = {}
    for ch in channels:
        if ch not in latest.columns:
            contributions[ch] = 0.0
            continue
        modified = latest.copy()
        modified[ch] = pd.to_numeric(modified[ch], errors="coerce").fillna(0.0) + delta
        X_mod = _ensure_X_for_pipeline(modified, meta)
        pred_mod = float(pipeline.predict(X_mod)[0])
        contributions[ch] = pred_mod - base_pred
    return {"base_prediction": base_pred, "delta": delta, "contributions": contributions}

if __name__ == "__main__":
    print("Latest prediction:", predict_latest())
    print("Contributions:", channel_contributions_finite_diff())
