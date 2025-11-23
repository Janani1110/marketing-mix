import streamlit as st
import pandas as pd
import yaml
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
from src.feature_engineering import prepare_features

st.set_page_config(layout="wide")

st.title("📊 Time Series Analysis")

df = load_and_clean()
df = prepare_features(df)

col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📈 Revenue vs Spend (Last 180 days)")
    ts_cols = ["spend_google","spend_facebook","spend_influencer","revenue"]

    ts = df.set_index("date")[ts_cols].tail(180)
    st.line_chart(ts)

with col2:
    st.subheader("📅 Summary Metrics")

    st.metric("Latest Revenue", f"${df['revenue'].iloc[-1]:,.0f}")
    st.metric("7-day Avg Revenue", f"${df['revenue'].tail(7).mean():,.0f}")
    st.metric("Total Units Sold", int(df["units_sold"].sum()))

st.markdown("### 📦 Spend Mix Breakdown")

spend_cols = [c for c in df.columns if c.startswith("spend_")]
latest_spend = df[spend_cols].iloc[-1]

st.bar_chart(latest_spend)
