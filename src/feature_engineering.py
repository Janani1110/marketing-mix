# src/feature_engineering.py
"""
Feature engineering for the expanded dataset (robust, dtype-safe).

Produces:
 - CTR, CPC, CPM
 - sessions, conversion_rate, avg_order_value (if missing)
 - day/month/week features
 - lag1 and 7-day rolling features for key numeric columns
 - interaction terms and spend shares

Use prepare_features(df) to get an engineered DataFrame.
"""

import pandas as pd
import numpy as np
from typing import List

def safe_div_num(a, b):
    a = pd.to_numeric(a, errors="coerce")
    b = pd.to_numeric(b, errors="coerce")
    with np.errstate(divide='ignore', invalid='ignore'):
        out = np.where((b == 0) | (pd.isna(b)), 0.0, a / b)
    return pd.Series(out)

def safe_numeric(df: pd.DataFrame, cols: List[str]):
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
    return df

def compute_ad_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ensure core numeric columns exist and are numeric
    core_numeric = [
        "spend_google", "spend_facebook", "spend_influencer",
        "impressions_google", "impressions_facebook", "impressions_influencer",
        "clicks_google", "clicks_facebook", "clicks_influencer",
        "units_sold", "revenue", "sessions", "avg_order_value", "conversion_rate",
        "discount"
    ]
    df = safe_numeric(df, core_numeric)

    # CTRs
    df["ctr_google"]     = safe_div_num(df.get("clicks_google", 0), df.get("impressions_google", 0))
    df["ctr_facebook"]   = safe_div_num(df.get("clicks_facebook", 0), df.get("impressions_facebook", 0))
    df["ctr_influencer"] = safe_div_num(df.get("clicks_influencer", 0), df.get("impressions_influencer", 0))

    # CPCs
    df["cpc_google"]     = safe_div_num(df.get("spend_google", 0), df.get("clicks_google", 0))
    df["cpc_facebook"]   = safe_div_num(df.get("spend_facebook", 0), df.get("clicks_facebook", 0))
    df["cpc_influencer"] = safe_div_num(df.get("spend_influencer", 0), df.get("clicks_influencer", 0))

    # CPMs (cost per 1000 impressions)
    df["cpm_google"]     = safe_div_num(df.get("spend_google", 0) * 1000.0, df.get("impressions_google", 0))
    df["cpm_facebook"]   = safe_div_num(df.get("spend_facebook", 0) * 1000.0, df.get("impressions_facebook", 0))
    df["cpm_influencer"] = safe_div_num(df.get("spend_influencer", 0) * 1000.0, df.get("impressions_influencer", 0))

    # sessions fallback (if sessions column missing or zero)
    if "sessions" not in df.columns or df["sessions"].isna().all():
        clicks_sum = df.get("clicks_google", 0).fillna(0) + df.get("clicks_facebook", 0).fillna(0) + df.get("clicks_influencer", 0).fillna(0)
        df["sessions"] = clicks_sum * 0.85

    # avg order value fallback
    if "avg_order_value" not in df.columns or df["avg_order_value"].isna().all():
        df["avg_order_value"] = safe_div_num(df.get("revenue", 0), df.get("units_sold", 0))

    # conversion rate fallback
    if "conversion_rate" not in df.columns or df["conversion_rate"].isna().all():
        df["conversion_rate"] = safe_div_num(df.get("units_sold", 0), df.get("sessions", 0))

    # ensure numeric for derived
    derived_numeric = [
        "ctr_google","ctr_facebook","ctr_influencer",
        "cpc_google","cpc_facebook","cpc_influencer",
        "cpm_google","cpm_facebook","cpm_influencer",
        "sessions","conversion_rate","avg_order_value"
    ]
    df = safe_numeric(df, derived_numeric)

    return df

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date").reset_index(drop=True)
    df["day_of_week"] = df["date"].dt.dayofweek.fillna(0).astype(int)
    # dt.isocalendar returns DataFrame with 'week' column on newer pandas
    try:
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
    except Exception:
        df["week_of_year"] = df["date"].dt.week.astype(int)
    df["month"] = df["date"].dt.month.fillna(0).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5,6]).astype(int)
    return df

def add_lags_and_rolls(df: pd.DataFrame, lag_cols: List[str]=None) -> pd.DataFrame:
    df = df.copy()
    if lag_cols is None:
        lag_cols = [
            "spend_google","spend_facebook","spend_influencer",
            "impressions_google","impressions_facebook","impressions_influencer",
            "clicks_google","clicks_facebook","clicks_influencer",
            "sessions","units_sold","avg_order_value","conversion_rate",
            "ctr_google","ctr_facebook","ctr_influencer"
        ]
    for col in lag_cols:
        if col in df.columns:
            df[f"{col}_lag1"] = df[col].shift(1).fillna(0.0)
            df[f"{col}_roll7"] = df[col].rolling(window=7, min_periods=1).mean().fillna(0.0)
    return df

def add_interactions(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # interactions: spend x ctr
    df["google_spend_x_ctr"] = df.get("spend_google", 0.0) * df.get("ctr_google", 0.0)
    df["facebook_spend_x_ctr"] = df.get("spend_facebook", 0.0) * df.get("ctr_facebook", 0.0)
    df["influencer_spend_x_ctr"] = df.get("spend_influencer", 0.0) * df.get("ctr_influencer", 0.0)

    df["spend_total"] = df.get("spend_google", 0.0) + df.get("spend_facebook", 0.0) + df.get("spend_influencer", 0.0)
    df["spend_google_share"] = safe_div_num(df.get("spend_google", 0.0), df["spend_total"])
    df["spend_facebook_share"] = safe_div_num(df.get("spend_facebook", 0.0), df["spend_total"])
    df["spend_influencer_share"] = safe_div_num(df.get("spend_influencer", 0.0), df["spend_total"])

    # coerce to numeric
    df = safe_numeric(df, ["google_spend_x_ctr","facebook_spend_x_ctr","influencer_spend_x_ctr",
                          "spend_total","spend_google_share","spend_facebook_share","spend_influencer_share"])
    return df

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # ensure categorical placeholders exist
    if "promo_type" not in df.columns:
        df["promo_type"] = "none"
    if "season" not in df.columns:
        df["season"] = "unknown"

    df["promo_type"] = df["promo_type"].astype(str)
    df["season"] = df["season"].astype(str)

    df = compute_ad_metrics(df)
    df = add_time_features(df)
    df = add_lags_and_rolls(df)
    df = add_interactions(df)

    # final safety: convert any remaining object-like numeric cells to numeric or 0
    # (do not change categorical types)
    numeric_candidate_cols = [c for c in df.columns if c not in ["date","promo_type","season"]]
    df = safe_numeric(df, numeric_candidate_cols)

    return df

# Backwards compatibility - not used for final pipeline automatic detection
def get_feature_columns():
    return []
