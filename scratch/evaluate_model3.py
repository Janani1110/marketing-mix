import pandas as pd
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import Ridge, LinearRegression
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
    c for c in df.columns if (c.startswith("spend_") and ("_" not in c.replace("spend_",""))) and df[c].dtype != object
]
# Wait, "spend_google_share" has "_" after. Let's just hardcode.
valid_numeric_cols = ["spend_google", "spend_facebook", "spend_influencer"]
num_cols = valid_numeric_cols

X = df[num_cols].copy()
y = df[target].astype(float).values
for c in num_cols:
    X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0.0)

numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, num_cols)
], remainder="drop")

# Try Ridge with positive=True
try:
    model1 = Ridge(alpha=1.0, positive=True)
    pipeline1 = Pipeline(steps=[("preprocessor", preprocessor), ("model", model1)])
    pipeline1.fit(X, y)
    print("Ridge(positive=True) Coefs:", pipeline1.named_steps['model'].coef_)
except Exception as e:
    print("Ridge Error:", e)

# Try Linear Regression with positive=True
try:
    model2 = LinearRegression(positive=True)
    pipeline2 = Pipeline(steps=[("preprocessor", preprocessor), ("model", model2)])
    pipeline2.fit(X, y)
    print("LinearRegression(positive=True) Coefs:", pipeline2.named_steps['model'].coef_)
except Exception as e:
    print("Linear Error:", e)

