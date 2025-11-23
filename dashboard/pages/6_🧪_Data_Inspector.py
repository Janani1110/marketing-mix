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
from src.etl import load_and_clean

st.title("🧪 Data Inspector")

df = load_and_clean()

st.subheader("Latest 50 rows")
st.dataframe(df.tail(50))

st.subheader("Column Summary")
st.write(df.describe(include="all"))
