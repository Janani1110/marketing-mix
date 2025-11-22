"""
src/data_streamer.py

Simulated data streamer for the expanded MMM dataset.
Appends a new row to data/streamed_data.csv every N seconds.

Run:
    python src/data_streamer.py
"""

import csv
import os
import time
from datetime import datetime
import numpy as np
import yaml

CONFIG_PATH = "config/config.yaml"
STREAM_FILE = "data/streamed_data.csv"

PROMO_TYPES = ["email", "tv", "radio", "social", "holiday"]
SEASONS = ["winter", "spring", "summer", "fall"]

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def create_header_if_missing(file_path, header):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def generate_row():
    # --- Spend ---
    spend_google = float(np.random.normal(2500, 600))
    spend_facebook = float(np.random.normal(2000, 500))
    spend_influencer = float(np.random.normal(1500, 400))

    # --- Impressions & Clicks ---
    impressions_google = int(max(50000, spend_google * 120))
    clicks_google = int(impressions_google * np.random.uniform(0.02, 0.06))

    impressions_facebook = int(max(40000, spend_facebook * 110))
    clicks_facebook = int(impressions_facebook * np.random.uniform(0.02, 0.05))

    impressions_influencer = int(max(30000, spend_influencer * 90))
    clicks_influencer = int(impressions_influencer * np.random.uniform(0.03, 0.07))

    # --- Sessions ---
    sessions = int(
        clicks_google * 0.6 +
        clicks_facebook * 0.55 +
        clicks_influencer * 0.5 +
        np.random.normal(2000, 500)
    )

    # --- Conversion rate ---
    conversion_rate = float(np.random.uniform(0.01, 0.05))

    # --- Units sold ---
    units_sold = int(max(1, sessions * conversion_rate))

    # --- Average Order Value (AOV) ---
    avg_order_value = float(np.random.normal(70, 15))

    # --- Revenue ---
    revenue = round(units_sold * avg_order_value, 2)

    # --- Discount ---
    discount = float(np.random.uniform(0.0, 0.25))

    # --- Promo / season rotation ---
    promo_type = np.random.choice(PROMO_TYPES)
    season = np.random.choice(SEASONS)

    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "spend_google": round(spend_google, 2),
        "spend_facebook": round(spend_facebook, 2),
        "spend_influencer": round(spend_influencer, 2),

        "impressions_google": impressions_google,
        "clicks_google": clicks_google,
        "impressions_facebook": impressions_facebook,
        "clicks_facebook": clicks_facebook,
        "impressions_influencer": impressions_influencer,
        "clicks_influencer": clicks_influencer,

        "sessions": sessions,
        "units_sold": units_sold,
        "avg_order_value": round(avg_order_value, 2),
        "conversion_rate": round(conversion_rate, 4),
        "discount": round(discount, 3),

        "promo_type": promo_type,
        "season": season,
        "revenue": revenue
    }

def append_row(file_path, header, row):
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)

def main():
    cfg = load_config()
    interval = cfg["data"].get("stream_interval_sec", 60)

    header = [
        "date",
        "spend_google", "spend_facebook", "spend_influencer",
        "impressions_google", "clicks_google",
        "impressions_facebook", "clicks_facebook",
        "impressions_influencer", "clicks_influencer",
        "sessions", "units_sold", "avg_order_value", "conversion_rate",
        "discount", "promo_type", "season", "revenue"
    ]

    create_header_if_missing(STREAM_FILE, header)
    print(f"Streaming to {STREAM_FILE} every {interval}s. Ctrl-C to stop.")

    try:
        while True:
            row = generate_row()
            append_row(STREAM_FILE, header, row)

            print(f"[{row['date']}] appended → rev=${row['revenue']} units={row['units_sold']} promo={row['promo_type']}")
            time.sleep(interval)

    except KeyboardInterrupt:
        print("Streamer stopped by user.")

if __name__ == "__main__":
    main()
