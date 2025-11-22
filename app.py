"""
dashboard/app.py

Streamlit dashboard for the real-time MMM demo.
Shows time series, latest predicted revenue, channel contributions, and a budget optimizer.
"""

import streamlit as st
import pandas as pd
import yaml
import os
from src.predict import predict_latest
from src.etl import load_and_clean
from src.optimizer import optimize_budget


st.set_page_config(layout="wide", page_title="Real-Time MMM Dashboard")

st.title("📈 Real-Time Marketing Mix Modeling (MMM) Dashboard")

cfg = yaml.safe_load(open("config/config.yaml"))

# Load cleaned data (this will read combined historical + streamed)
if os.path.exists(cfg["data"]["clean_path"]):
    df = pd.read_csv(cfg["data"]["clean_path"], parse_dates=["date"])
else:
    # If clean not generated, generate now
    from src.etl import load_and_clean
    df = load_and_clean()

# Left column: time series
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("Spends & Revenue (latest)")
    st.line_chart(df.set_index("date")[["google_ads","facebook_ads","influencer","revenue"]])

with col2:
    st.subheader("Latest Prediction")
    try:
        pred_info = predict_latest()
        st.metric("Predicted revenue", f"${pred_info['predicted_revenue']:.0f}")
        st.write("Channel contributions (approx):")
        st.write(pd.DataFrame.from_dict(pred_info["contributions"], orient="index", columns=["contribution"]).assign(contribution=lambda d: d["contribution"].round(2)))
        st.write("Residual / intercept info:")
        st.write({"intercept": round(pred_info["intercept"],2), "residual": round(pred_info["residual"],2)})
    except Exception as e:
        st.error(f"Prediction error: {e}")

st.markdown("---")
st.subheader("Budget Optimizer")
budget = st.number_input("Total budget (USD)", value=cfg["optimizer"]["budget"], step=1000)
if st.button("Optimize allocation"):
    try:
        alloc = optimize_budget(budget)
        st.write(pd.Series(alloc).rename("allocated"))
    except Exception as e:
        st.error(f"Optimization error: {e}")

st.markdown("---")
st.subheader("Data Inspector")
st.write(df.tail(10))
