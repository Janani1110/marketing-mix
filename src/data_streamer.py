"""
src/data_streamer.py

Simple simulated data streamer that appends a new row to data/streamed_data.csv
every N seconds (configured in config/config.yaml). This acts as the 'real-time'
data ingestion step for demos / portfolios.

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

def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)

def create_header_if_missing(file_path, header):
    if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
        with open(file_path, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(header)

def generate_row():
    # You can tune distributions here to mimic realistic patterns
    g = float(np.random.normal(loc=400, scale=80))
    f = float(np.random.normal(loc=220, scale=60))
    i = float(np.random.normal(loc=120, scale=40))
    discount = float(max(0, min(0.25, np.random.normal(loc=0.06, scale=0.03))))
    # revenue roughly correlated with total spend + noise
    revenue = max(0, 50 + 8 * g + 6 * f + 5 * i - 1000 * discount + np.random.normal(0, 500))
    return {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "google_ads": round(g, 2),
        "facebook_ads": round(f, 2),
        "influencer": round(i, 2),
        "discount": round(discount, 3),
        "revenue": round(revenue, 2)
    }

def append_row(file_path, header, row):
    with open(file_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=header)
        writer.writerow(row)

def main():
    cfg = load_config()
    interval = cfg["data"].get("stream_interval_sec", 60)
    header = ["date", "google_ads", "facebook_ads", "influencer", "discount", "revenue"]
    create_header_if_missing(STREAM_FILE, header)
    print(f"Streaming to {STREAM_FILE} every {interval}s. Ctrl-C to stop.")
    try:
        while True:
            row = generate_row()
            append_row(STREAM_FILE, header, row)
            print(f"[{row['date']}] appended: google={row['google_ads']}, fb={row['facebook_ads']}, infl={row['influencer']}, revenue={row['revenue']}")
            time.sleep(interval)
    except KeyboardInterrupt:
        print("Streamer stopped by user.")

if __name__ == "__main__":
    main()
