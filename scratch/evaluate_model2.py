import pandas as pd
import numpy as np
import yaml
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
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

valid_numeric_cols = [
    "spend_google", "spend_facebook", "spend_influencer"
]
numeric_cols_new = [c for c in valid_numeric_cols if c in df.columns]

categorical_cols = []

def get_coeffs(num_cols):
    X = df[num_cols].copy()
    y = df[target].astype(float).values
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols)
    ], remainder="drop")
    model = Ridge(alpha=1.0)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    ridge_model = pipeline.named_steps['model']
    
    feature_names = num_cols 
    coeffs = dict(zip(feature_names, ridge_model.coef_))
    
    return coeffs, pipeline.score(X, y)

new_c, new_score = get_coeffs(numeric_cols_new)

print("NEW MODEL")
print("R2: ", new_score)
print(f"Spend Google Coef: {new_c.get('spend_google', 0)}")
print(f"Spend Facebook Coef: {new_c.get('spend_facebook', 0)}")
print(f"Spend Influencer Coef: {new_c.get('spend_influencer', 0)}")

