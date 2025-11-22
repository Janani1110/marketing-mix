# src/etl.py
"""
Batch + streaming merge into a single cleaned CSV used for modeling.
Call `load_and_clean()` to regenerate clean_data.csv.
"""

import pandas as pd
import os
import yaml

CONFIG_PATH = "config/config.yaml"

def load_config():
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

def load_dataframes():
    cfg = load_config()
    hist_path = cfg["data"]["historical_path"]
    stream_path = cfg["data"]["streamed_path"]

    if not os.path.exists(hist_path):
        raise FileNotFoundError(f"Historical data not found: {hist_path}")

    df_hist = pd.read_csv(hist_path, parse_dates=["date"])

    if os.path.exists(stream_path) and os.path.getsize(stream_path) > 0:
        df_stream = pd.read_csv(stream_path, parse_dates=["date"])
    else:
        df_stream = pd.DataFrame(columns=df_hist.columns)

    return df_hist, df_stream

def simple_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # drop duplicate dates
    if "date" in df.columns:
        df = df.drop_duplicates(subset=["date"], keep="last")

    # auto-detect numeric columns (numbers stored as numeric types)
    # plus try to coerce common numeric-like columns
    for col in df.columns:
        if col == "date":
            continue
        # try convert floats/ints where possible
        df[col] = pd.to_numeric(df[col], errors="ignore")

    # now find numeric columns and coerce missing values
    numeric_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

    for col in numeric_cols:
        median = df[col].median()
        df[col] = pd.to_numeric(df[col], errors="coerce").fillna(median)

    # categorical cleanup
    if "season" in df.columns:
        df["season"] = df["season"].fillna("unknown").astype(str)
    if "promo_type" in df.columns:
        df["promo_type"] = df["promo_type"].fillna("none").astype(str)

    # final sort by date
    if "date" in df.columns:
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
