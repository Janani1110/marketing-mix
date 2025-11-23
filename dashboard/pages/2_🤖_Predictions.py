# dashboard/pages/2_🤖_Predictions.py
import streamlit as st
import pandas as pd
import numpy as np
import sys
import os

# Resolve project root dynamically (no assumptions)
CURRENT_FILE = os.path.abspath(__file__)
DASHBOARD_DIR = os.path.dirname(CURRENT_FILE)

# Go up until we reach the project root (contains src/)
root = DASHBOARD_DIR
while root != "/" and "src" not in os.listdir(root):
    root = os.path.dirname(root)

if "src" not in os.listdir(root):
    raise RuntimeError("❌ Could not locate project root containing 'src' folder.")

if root not in sys.path:
    sys.path.insert(0, root)

from src.etl import load_and_clean
from src.predict import load_pipeline_and_meta, _ensure_X_for_pipeline, predict_latest, channel_contributions_finite_diff
from src.etl import load_and_clean
from src.feature_engineering import prepare_features

st.set_page_config(page_title="Predictions", layout="wide")
st.title("📊 Revenue Predictions")

# Load clean data for defaults
df_clean = load_and_clean()
df_clean = prepare_features(df_clean)

# Get median/default values for non-controllable features
median_row = df_clean.median(numeric_only=True)
default_vals = median_row.to_dict()

# Sidebar inputs for user-controllable features
st.sidebar.header("Adjust Inputs")

spend_google = st.sidebar.number_input("Google Spend", min_value=0.0, value=float(default_vals.get("spend_google", 1000)))
spend_facebook = st.sidebar.number_input("Facebook Spend", min_value=0.0, value=float(default_vals.get("spend_facebook", 1000)))
spend_influencer = st.sidebar.number_input("Influencer Spend", min_value=0.0, value=float(default_vals.get("spend_influencer", 1000)))
discount = st.sidebar.slider("Discount (%)", min_value=0.0, max_value=0.5, value=float(default_vals.get("discount", 0.05)), step=0.01)
promo_type = st.sidebar.selectbox("Promo Type", options=sorted(df_clean["promo_type"].unique()), index=0)
season = st.sidebar.selectbox("Season", options=sorted(df_clean["season"].unique()), index=0)

# Build input DataFrame for prediction
input_df = pd.DataFrame({
    "spend_google": [spend_google],
    "spend_facebook": [spend_facebook],
    "spend_influencer": [spend_influencer],
    "discount": [discount],
    "promo_type": [promo_type],
    "season": [season],
})

# Fill remaining features with median/defaults
all_features = df_clean.columns.tolist()
for col in all_features:
    if col not in input_df.columns:
        input_df[col] = float(default_vals.get(col, 0.0))

# Prepare features using your feature engineering pipeline
input_df_prepared = prepare_features(input_df)

# Load pipeline and meta
pipeline, meta = load_pipeline_and_meta()
X_input = _ensure_X_for_pipeline(input_df_prepared, meta)

# Make prediction
predicted_revenue = float(pipeline.predict(X_input)[0])

st.metric("Predicted Revenue", f"${predicted_revenue:,.2f}")

# Show channel contributions using finite-difference
contributions = channel_contributions_finite_diff(delta=500.0, channels=["spend_google", "spend_facebook", "spend_influencer"])
st.subheader("Estimated Channel Contributions (per +$500 spend)")
st.write(pd.Series(contributions["contributions"], name="Δ Revenue"))

# Optional: show input data used for prediction
with st.expander("Show full input row used for prediction"):
    st.dataframe(input_df_prepared)
