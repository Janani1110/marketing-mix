# dashboard/app.py
"""
Streamlit dashboard updated for expanded MMM dataset.

Run:
    streamlit run dashboard/app.py
"""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import streamlit as st
import pandas as pd
import yaml

from src.etl import load_and_clean
from src.feature_engineering import prepare_features
from src.train_model import train_and_save
from src.predict import predict_latest, channel_contributions_finite_diff
from src.optimizer import optimize_budget

CONFIG_PATH = "config/config.yaml"
cfg = yaml.safe_load(open(CONFIG_PATH))

st.set_page_config(layout="wide", page_title="Real-Time MMM — Expanded Dataset")

st.title("📊 Real-Time Marketing Mix Modeling — Expanded Dataset")


# =============================================================
# LEFT SECTION: TIME SERIES VISUALIZATIONS
# =============================================================
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("Spends & Revenue (time series)")
    df = load_and_clean()
    df = prepare_features(df)

    ts_cols = [
        "spend_google",
        "spend_facebook",
        "spend_influencer",
        "revenue"
    ]

    ts = df.set_index("date")[ts_cols].tail(180)
    st.line_chart(ts)


# =============================================================
# RIGHT SECTION: LATEST PREDICTION
# =============================================================
with col2:
    st.subheader("🤖 Latest Prediction")

    try:
        pred = predict_latest()
        st.metric("Predicted Revenue", f"${pred['predicted_revenue']:.0f}")

        st.write("Latest Input Row:")
        st.json(pred["input_row"])

    except Exception as e:
        st.error(f"Prediction error: {e}")


# =============================================================
# CHANNEL CONTRIBUTIONS
# =============================================================
st.markdown("---")
st.subheader("📉 Channel Contributions (Finite Difference)")

try:
    contr = channel_contributions_finite_diff(delta=500.0)

    st.write(
        pd.Series(
            contr["contributions"],
            name="Estimated Revenue Lift per +$500 Spend"
        )
    )

except Exception as e:
    st.error(f"Contribution calc error: {e}")


# =============================================================
# BUDGET OPTIMIZER SECTION
# =============================================================
st.markdown("---")
st.subheader("💰 Budget Optimizer")

budget = st.number_input(
    "Total budget to allocate (USD)",
    value=int(cfg["optimizer"]["budget"]),
    step=1000
)

step = st.number_input(
    "Allocation step size (USD)",
    value=500,
    step=100
)

if st.button("Run optimizer"):
    try:
        with st.spinner("Optimizing..."):
            alloc = optimize_budget(float(budget), step=float(step))

            st.write(f"Initial prediction: **{int(alloc['initial_prediction'])}**")
            st.write(f"Final prediction after optimization: **{int(alloc['final_prediction'])}**")

            st.write("Recommended Spend Allocation:")
            st.write(pd.Series(alloc["allocated_budget"]).astype(float))

            st.write("Budget Left:", alloc["budget_left"])
            st.write("Iterations:", alloc["iterations"])

    except Exception as e:
        st.error(f"Optimizer error: {e}")


# =============================================================
# DATA INSPECTOR
# =============================================================
st.markdown("---")
st.subheader("🗂 Data Inspector — Latest Rows")

st.dataframe(df.tail(10))
