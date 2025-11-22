"""
src/etl.py

Batch + streaming merge into a single cleaned CSV used for modeling.
Call `load_and_clean()` to regenerate clean_data.csv.
"""

import pandas as pd
import os
import yaml

CONFIG_PATH = "config/config.yaml"

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def load_dataframes():
    cfg = load_config()
    hist_path = cfg["data"]["historical_path"]
    stream_path = cfg["data"]["streamed_path"]
    # Load historical (must exist)
    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Historical data not found: {hist_path}")
    df_hist = pd.read_csv(hist_path, parse_dates=["date"])
    # Load streamed (may be missing if streamer not run yet)
    if os.path.exists(stream_path) and os.path.getsize(stream_path) > 0:
        df_stream = pd.read_csv(stream_path, parse_dates=["date"])
    else:
        df_stream = pd.DataFrame(columns=df_hist.columns)
    return df_hist, df_stream

def simple_clean(df):
    # Basic cleaning: drop duplicates, fill missing numeric with median, convert types
    df = df.copy()
    df = df.drop_duplicates(subset=["date"], keep="last")
    numeric_cols = ["google_ads", "facebook_ads", "influencer", "discount", "revenue"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            median = df[col].median()
            df[col] = df[col].fillna(median)
    # sort
    df = df.sort_values("date").reset_index(drop=True)
    return df

def load_and_clean():
    cfg = load_config()
    out_path = cfg["data"]["clean_path"]
    df_hist, df_stream = load_dataframes()
    df = pd.concat([df_hist, df_stream], ignore_index=True)
    df = simple_clean(df)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Cleaned data written to {out_path} (rows={len(df)})")
    return df

if __name__ == "__main__":
    load_and_clean()
