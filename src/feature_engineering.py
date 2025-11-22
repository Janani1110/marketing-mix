"""
src/feature_engineering.py

Feature transformations used by the MMM pipeline.
"""

import numpy as np
import pandas as pd
import yaml

CONFIG_PATH = "config/config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def adstock(series, decay=0.6):
    """
    Geometric adstock: carry-over effect.
    series: iterable of spend values (ordered oldest->newest)
    decay: float in (0,1)
    returns: numpy array of adstocked values (same length)
    """
    carry = 0.0
    out = []
    for x in series:
        carry = x + decay * carry
        out.append(carry)
    return np.array(out)

def hill_saturation(x, alpha=1.0, beta=1.0):
    """
    Hill function: alpha * x^beta / (1 + alpha * x^beta)
    For positive x. Produces saturation-like curve in (0,1).
    We then multiply by x to keep scale if desired or just return fraction.
    """
    x = np.array(x, dtype=float)
    num = alpha * (x ** beta)
    den = 1.0 + num
    return num / den

def add_features(df):
    cfg = load_config()
    decay = cfg["features"]["adstock_decay"]
    alpha = cfg["features"]["hill_alpha"]
    beta = cfg["features"]["hill_beta"]

    df = df.copy()
    # Ensure sorts
    df = df.sort_values("date").reset_index(drop=True)

    for channel in ["google_ads", "facebook_ads", "influencer"]:
        adstocked = adstock(df[channel].values, decay=decay)
        df[f"{channel}_adstock"] = adstocked
        # saturation (scale back to spend magnitude by multiplying fraction with spend)
        sat_frac = hill_saturation(df[channel].values, alpha=alpha, beta=beta)
        df[f"{channel}_sat_frac"] = sat_frac
        df[f"{channel}_sat"] = sat_frac * df[channel].values

    # rolling features (recent 7-day spend)
    for ch in ["google_ads", "facebook_ads", "influencer"]:
        df[f"{ch}_7d_mean"] = df[ch].rolling(window=7, min_periods=1).mean()

    # time features
    df["day_of_week"] = pd.to_datetime(df["date"]).dt.dayofweek
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    return df
