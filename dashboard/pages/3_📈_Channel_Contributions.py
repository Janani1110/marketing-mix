import streamlit as st
import pandas as pd

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

from src.predict import channel_contributions_finite_diff

st.title("📈 Channel Contribution Analysis")

delta = st.number_input("Spend increment ($)", value=500, step=100)

try:
    contr = channel_contributions_finite_diff(delta=delta)

    st.subheader("Estimated Revenue Lift per +$" + str(delta))
    contrib_series = pd.Series(contr["contributions"])
    st.bar_chart(contrib_series)

    st.dataframe(contrib_series.rename("Δ Revenue"))

except Exception as e:
    st.error(f"Contribution error: {e}")
