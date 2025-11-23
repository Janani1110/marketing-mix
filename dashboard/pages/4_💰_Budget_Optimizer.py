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
from src.optimizer import optimize_budget

st.title("💰 Budget Optimization")

budget = st.number_input("Total budget to allocate ($)", 1000, 100000, 10000, step=500)
step_size = st.number_input("Step size ($)", 100, 2000, 500)

if st.button("Optimize"):
    with st.spinner("Optimizing..."):
        try:
            result = optimize_budget(budget, step_size)
            st.success("Optimization complete!")

            st.metric("Initial Prediction", f"${result['initial_prediction']:,.0f}")
            st.metric("Final Prediction", f"${result['final_prediction']:,.0f}")

            st.subheader("Budget Allocation")
            st.bar_chart(pd.Series(result["allocated_budget"]))

            st.write("Iterations:", result["iterations"])

        except Exception as e:
            st.error(f"Optimizer error: {e}")
