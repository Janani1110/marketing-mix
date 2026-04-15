# src/train_model.py
"""
Train and save a robust sklearn pipeline for MMM.
Saves:
 - models/mmm_pipeline.pkl
 - models/model_metadata.yaml
"""

import os
import sys
# Add project root to sys.path
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import yaml
import joblib
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

from src.etl import load_and_clean
from src.feature_engineering import prepare_features

MODEL_DIR = os.path.join(ROOT_DIR, "models")
PIPELINE_FILE = os.path.join(MODEL_DIR, "mmm_pipeline.pkl")
META_FILE = os.path.join(MODEL_DIR, "model_metadata.yaml")
CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def build_pipeline(numeric_features, categorical_features):
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])

    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ], remainder="drop")

    model = Ridge(alpha=1.0)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    return pipeline

def prepare_training_data():
    df = load_and_clean()
    df = prepare_features(df)

    cfg = load_config()
    target = cfg["model"]["target"]
    if target not in df.columns:
        raise ValueError(f"Target '{target}' not found in data")

    # dynamic feature detection (exclude date + target)
    ignore_cols = {"date", target}
    numeric_cols = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
    categorical_cols = [c for c in ["promo_type", "season"] if c in df.columns]

    # coerce numeric columns to floats and fillna
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # fill categorical missing values and cast to str
    for c in categorical_cols:
        df[c] = df[c].fillna("unknown").astype(str)

    X = df[numeric_cols + categorical_cols].copy()
    y = df[target].astype(float).values

    return X, y, numeric_cols, categorical_cols

def train_and_save(test_size=0.2, random_state=42):
    os.makedirs(MODEL_DIR, exist_ok=True)

    X, y, numeric_features, categorical_features = prepare_training_data()

    if len(X) < 20:
        raise ValueError("Not enough rows to train model. Need >= 20 rows.")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)

    pipeline = build_pipeline(numeric_features, categorical_features)
    pipeline.fit(X_train, y_train)

    # metrics
    y_pred = pipeline.predict(X_test)
    r2 = float(r2_score(y_test, y_pred))
    rmse = float(mean_squared_error(y_test, y_pred, squared=False))

    # save pipeline and metadata
    joblib.dump(pipeline, PIPELINE_FILE)
    meta = {
        "target": load_config()["model"]["target"],
        "numeric_features": numeric_features,
        "categorical_features": categorical_features,
        "r2": r2,
        "rmse": rmse,
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test))
    }
    with open(META_FILE, "w") as f:
        yaml.safe_dump(meta, f)

    print(f"Pipeline saved to {PIPELINE_FILE}")
    print(f"Metadata saved to {META_FILE}")
    print(f"Metrics: r2={r2:.4f}, rmse={rmse:.2f}")
    return pipeline, meta

if __name__ == "__main__":
    train_and_save()
