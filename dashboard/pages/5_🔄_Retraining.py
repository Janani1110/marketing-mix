import streamlit as st
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
from src.retrain import run_full_retrain_cycle

st.title("🔄 Model Retraining")

st.markdown("""
Use this page to **manually retrain** the MMM model on the latest cleaned dataset.
""")

if st.button("Run Full Retrain Now"):
    with st.spinner("Retraining model... this may take a few seconds..."):
        pipeline, meta = run_full_retrain_cycle()
        st.success("Retrain complete!")

        st.json(meta)
