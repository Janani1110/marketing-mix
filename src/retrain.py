"""
src/retrain.py

Simple retrain loop that regenerates clean data and retrains the model.
Set `retrain.run_every_minutes` in config/config.yaml to control automatic retraining.
"""

import time
import yaml
from train_model import train_and_save

CONFIG_PATH = "config/config.yaml"

def load_config():
    return yaml.safe_load(open(CONFIG_PATH))

def main():
    cfg = load_config()
    run_every = cfg.get("retrain", {}).get("run_every_minutes", None)
    enabled = cfg.get("retrain", {}).get("enabled", False)
    if enabled and run_every:
        print(f"Auto-retrain enabled every {run_every} minutes. Ctrl-C to stop.")
        try:
            while True:
                train_and_save()
                time.sleep(run_every * 60)
        except KeyboardInterrupt:
            print("Auto-retrain stopped by user.")
    else:
        train_and_save()

if __name__ == "__main__":
    main()
