import pandas as pd
import numpy as np
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, FunctionTransformer
from sklearn.linear_model import Ridge
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.etl import load_and_clean
from src.feature_engineering import prepare_features

df = load_and_clean()
df = prepare_features(df)

target = "revenue"

base_numeric = ["spend_google", "spend_facebook", "spend_influencer"]
num_cols = [c for c in base_numeric if c in df.columns]
categorical_cols = [c for c in ["promo_type", "season"] if c in df.columns]

X = df[num_cols + categorical_cols].copy()
y = df[target].astype(float).values
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
for c in categorical_cols:
    X[c] = X[c].astype(str).fillna("unknown")

# Add log transform!
numeric_transformer = Pipeline(steps=[
    ("log", FunctionTransformer(np.log1p, validate=False)),
    ("scaler", StandardScaler())
])
categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols),
    ("cat", categorical_transformer, categorical_cols)
], remainder="drop")

model = Ridge(alpha=1.0, positive=True)
pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
pipeline.fit(X, y)

import joblib
joblib.dump(pipeline, "scratch/test_pipeline.pkl")
meta = {
    "target": target,
    "numeric_features": num_cols,
    "categorical_features": categorical_cols,
}
with open("scratch/test_meta.yaml", "w") as f:
    yaml.safe_dump(meta, f)

# Sim optimizer
latest = df.tail(1).copy()
for c in num_cols:
    latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0.0)

base_pred = pipeline.predict(latest)[0]
allocation = {ch: 0.0 for ch in base_numeric}
budget_left = 10000.0
step = 500.0
current = latest.copy()

def marginal_gain(ch):
    mod = current.copy()
    mod[ch] = mod[ch] + step
    pred_mod = pipeline.predict(mod)[0]
    pred_cur = pipeline.predict(current)[0]
    return float(pred_mod - pred_cur)

iters = 0
while budget_left >= step and iters < 200:
    gains = {ch: marginal_gain(ch) for ch in base_numeric}
    best_ch = max(gains, key=gains.get)
    best_gain = gains[best_ch]
    if best_gain <= 0:
        break
    allocation[best_ch] += step
    budget_left -= step
    current[best_ch] += step
    iters += 1

print("Log Transform Allocator Output:")
print(allocation)

