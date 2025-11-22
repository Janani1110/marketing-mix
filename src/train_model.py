"""
src/train_model.py

Train a regularized MMM (Ridge) model using engineered features, save to models/mmm_model.pkl
Also write model_metadata.yaml with feature list and training metrics.
"""

import os
import pickle
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score, mean_squared_error
from feature_engineering import add_features
from etl import load_and_clean

CONFIG_PATH = "config/config.yaml"
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "mmm_model.pkl")
META_FILE = os.path.join(MODEL_DIR, "model_metadata.yaml")

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def prepare_train_data(df):
    df_feats = add_features(df)
    # choose features
    feature_cols = [
        "google_ads_adstock", "facebook_ads_adstock", "influencer_adstock",
        "google_ads_sat", "facebook_ads_sat", "influencer_sat",
        "discount", "google_ads_7d_mean", "facebook_ads_7d_mean", "influencer_7d_mean",
        "is_weekend"
    ]
    X = df_feats[feature_cols].fillna(0).values
    y = df_feats[load_config()["model"]["target"]].values
    return X, y, feature_cols

def train_and_save():
    os.makedirs(MODEL_DIR, exist_ok=True)
    # load & clean data
    df = load_and_clean()
    X, y, feature_cols = prepare_train_data(df)

    # Train/test split simple - last 20% as test
    split = int(0.8 * len(X))
    if split < 1:
        split = 1
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test) if len(X_test) else model.predict(X_train)
    r2 = r2_score(y_test, y_pred) if len(y_test) else r2_score(y_train, model.predict(X_train))
    rmse = mean_squared_error(y_test, y_pred, squared=False) if len(y_test) else mean_squared_error(y_train, model.predict(X_train), squared=False)

    # Save model
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    # Save meta
    meta = {
        "feature_cols": feature_cols,
        "r2": float(r2),
        "rmse": float(rmse),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }
    with open(META_FILE, "w") as f:
        yaml.safe_dump(meta, f)

    print(f"Trained model saved to {MODEL_FILE}")
    print(f"Metadata saved to {META_FILE}")
    print(f"Metrics: r2={r2:.4f}, rmse={rmse:.2f}")

if __name__ == "__main__":
    train_and_save()
