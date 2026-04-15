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

# Current logic
ignore_cols = {"date", target}
numeric_cols_old = [c for c in df.columns if c not in ignore_cols and pd.api.types.is_numeric_dtype(df[c])]
categorical_cols = [c for c in ["promo_type", "season"] if c in df.columns]

# New logic
valid_numeric_cols = [
    "spend_google", "spend_facebook", "spend_influencer",
    "spend_google_lag1", "spend_facebook_lag1", "spend_influencer_lag1",
    "spend_google_roll7", "spend_facebook_roll7", "spend_influencer_roll7",
    "day_of_week", "week_of_year", "month", "is_weekend"
]
numeric_cols_new = [c for c in valid_numeric_cols if c in df.columns]


def get_coeffs(num_cols):
    X = df[num_cols + categorical_cols].copy()
    y = df[target].astype(float).values
    for c in num_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)
    for c in categorical_cols:
        X[c] = X[c].astype(str).fillna("unknown")
    
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(steps=[("ohe", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    preprocessor = ColumnTransformer(transformers=[
        ("num", numeric_transformer, num_cols),
        ("cat", categorical_transformer, categorical_cols)
    ], remainder="drop")
    model = Ridge(alpha=1.0)
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("model", model)])
    pipeline.fit(X, y)
    ridge_model = pipeline.named_steps['model']
    
    cat_names = pipeline.named_steps['preprocessor'].transformers_[1][1].named_steps['ohe'].get_feature_names_out(categorical_cols)
    feature_names = num_cols + list(cat_names)
    coeffs = dict(zip(feature_names, ridge_model.coef_))
    
    return coeffs, pipeline.score(X, y)

old_c, old_score = get_coeffs(numeric_cols_old)
new_c, new_score = get_coeffs(numeric_cols_new)

print("OLD MODEL")
print("R2: ", old_score)
print(f"Spend Google Coef: {old_c.get('spend_google', 0)}")
print(f"Spend Facebook Coef: {old_c.get('spend_facebook', 0)}")
print(f"Spend Influencer Coef: {old_c.get('spend_influencer', 0)}")
print(f"Units Sold Coef: {old_c.get('units_sold', 0)}")

print("\nNEW MODEL")
print("R2: ", new_score)
print(f"Spend Google Coef: {new_c.get('spend_google', 0)}")
print(f"Spend Facebook Coef: {new_c.get('spend_facebook', 0)}")
print(f"Spend Influencer Coef: {new_c.get('spend_influencer', 0)}")

