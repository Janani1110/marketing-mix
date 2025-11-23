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

# ---------------------------------------------------------
# Streamlit Page Config
# ---------------------------------------------------------
st.set_page_config(
    page_title="Marketing Mix Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ---------------------------------------------------------
# MAIN HOME PAGE
# ---------------------------------------------------------
st.title("🚀 Marketing Mix Analytics Dashboard")

st.markdown("""
This dashboard provides:

### 🔍 Key Features
- **📊 Time-Series Visualizations**  
  Explore trends in spend, impressions, conversions, and revenue.

- **🤖 Real-Time Predictions**  
  Enter marketing inputs and instantly estimate revenue.

- **📈 Channel Contribution Modeling**  
  Understand which channels drive results using model explainability.

- **💰 Budget Optimizer**  
  Use the trained model to experiment with budget allocations.

- **🔄 Automated + Manual Retraining**  
  Quickly retrain models on new data streams.

- **🧪 Data Inspector**  
  Inspect raw, cleaned, and streamed datasets in detail.

Navigate using the **sidebar** to begin.
""")

st.info("💡 Tip: Pages automatically appear in the sidebar because they are inside the `pages/` folder.")
