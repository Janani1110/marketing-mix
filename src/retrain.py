"""
src/retrain.py

Runs the ETL cleaning + full model retraining cycle.

Controlled by config/config.yaml:

retrain:
    enabled: true
    run_every_minutes: 15
"""

import time
import yaml
import sys
import os

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from src.etl import load_and_clean
from src.train_model import train_and_save

CONFIG_PATH = os.path.join(ROOT_DIR, "config", "config.yaml")


def load_config():
    with open(CONFIG_PATH, "r") as f:
        return yaml.safe_load(f)


def run_full_retrain_cycle():
    """
    Executes a full retrain cycle:
        1. Clean + merge historical + streamed data
        2. Train model and save pipeline + metadata
    """
    print("\n=================================")
    print("🔥 Starting full retrain cycle...")
    print("=================================")

    # Step 1 — ETL cleaning
    print("→ Running ETL (load + merge + clean)...")
    df = load_and_clean()
    print(f"✓ ETL complete. Clean rows: {len(df)}")

    # Step 2 — Train model using cleaned data
    print("→ Training MMM model...")
    pipeline, meta = train_and_save()
    print("✓ Model retraining complete.")

    print("=================================")
    print("✅ Retrain cycle finished")
    print("=================================\n")

    return pipeline, meta


def main():
    cfg = load_config()
    enabled = cfg.get("retrain", {}).get("enabled", False)
    interval = cfg.get("retrain", {}).get("run_every_minutes", None)

    # Auto retrain loop
    if enabled and interval:
        print(f"Auto-retrain enabled: running every {interval} minutes. Press Ctrl-C to stop.")
        try:
            while True:
                run_full_retrain_cycle()
                time.sleep(interval * 60)
        except KeyboardInterrupt:
            print("\nAuto-retrain stopped by user.")
    else:
        # Single run
        print("Auto-retrain disabled → running a single cycle now.")
        run_full_retrain_cycle()


if __name__ == "__main__":
    main()
