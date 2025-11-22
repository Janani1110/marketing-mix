"""
src/predict.py

Load the trained model and run a prediction for the latest row.
Also compute a simple channel contribution breakdown using model coefficients on
channel-related features.
"""

import pickle
import yaml
import numpy as np
import pandas as pd
from src.feature_engineering import add_features
from src.etl import load_and_clean
import os

MODEL_FILE = "models/mmm_model.pkl"
META_FILE = "models/model_metadata.yaml"
CONFIG_PATH = "config/config.yaml"

def load_model_and_meta():
    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(f"Model not found. Train first: {MODEL_FILE}")
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(META_FILE, "r") as f:
        meta = yaml.safe_load(f)
    return model, meta

def predict_latest():
    model, meta = load_model_and_meta()
    df = load_and_clean()
    df_feats = add_features(df)
    latest = df_feats.iloc[-1:]
    feature_cols = meta["feature_cols"]
    X = latest[feature_cols].fillna(0).values
    pred = model.predict(X)[0]
    # derive contributions: multiply coeffs with channel-level features and sum
    coefs = np.array(model.coef_)
    coef_map = dict(zip(feature_cols, coefs))
    # simple contributions by grouping relevant features
    channels = {
        "google_ads": ["google_ads_adstock", "google_ads_sat", "google_ads_7d_mean"],
        "facebook_ads": ["facebook_ads_adstock", "facebook_ads_sat", "facebook_ads_7d_mean"],
        "influencer": ["influencer_adstock", "influencer_sat", "influencer_7d_mean"],
        "discount": ["discount"]
    }
    contributions = {}
    for ch, feats in channels.items():
        val = 0.0
        for f in feats:
            if f in latest.columns:
                val += coef_map.get(f, 0.0) * float(latest.iloc[0][f])
        contributions[ch] = float(val)
    # leftover = intercept + other features (weekend)
    intercept = float(model.intercept_) if hasattr(model, "intercept_") else 0.0
    explained = sum(contributions.values())
    residual = float(pred) - explained - intercept
    return {
        "predicted_revenue": float(pred),
        "contributions": contributions,
        "intercept": intercept,
        "residual": residual
    }

if __name__ == "__main__":
    out = predict_latest()
    print("Prediction result:")
    print(out)
