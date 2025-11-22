# src/optimizer.py
"""
Safe greedy budget allocator that uses the trained pipeline + metadata.
"""

import pandas as pd
import numpy as np

from src.predict import load_pipeline_and_meta
from src.feature_engineering import prepare_features
from src.etl import load_and_clean

def _prepare_latest_and_feature_cols(meta):
    df = load_and_clean()
    df = prepare_features(df)
    latest = df.tail(1).copy().reset_index(drop=True)
    feature_cols = meta["numeric_features"] + meta["categorical_features"]
    # ensure expected columns exist
    for c in feature_cols:
        if c not in latest.columns:
            latest[c] = 0 if c in meta["numeric_features"] else "unknown"
    return latest, feature_cols

def optimize_budget(total_budget: float, step: float = 500.0, channels=None, max_iters: int = 2000):
    if channels is None:
        channels = ["spend_google", "spend_facebook", "spend_influencer"]

    pipeline, meta = load_pipeline_and_meta()

    latest, feature_cols = _prepare_latest_and_feature_cols(meta)

    # coerce numeric columns in latest
    for c in meta["numeric_features"]:
        latest[c] = pd.to_numeric(latest[c], errors="coerce").fillna(0.0)

    # base pred
    X_base = latest.reindex(columns=feature_cols).copy()
    for c in meta["numeric_features"]:
        X_base[c] = pd.to_numeric(X_base[c], errors="coerce").fillna(0.0)
    for c in meta["categorical_features"]:
        X_base[c] = X_base[c].astype(str).fillna("unknown")

    base_pred = float(pipeline.predict(X_base)[0])

    allocation = {ch: 0.0 for ch in channels}
    budget_left = float(total_budget)
    current = latest.copy()
    iters = 0

    def marginal_gain(ch):
        mod = current.copy()
        mod[ch] = pd.to_numeric(mod[ch], errors="coerce").fillna(0.0) + step
        X_mod = mod.reindex(columns=feature_cols).copy()
        for c in meta["numeric_features"]:
            X_mod[c] = pd.to_numeric(X_mod[c], errors="coerce").fillna(0.0)
        for c in meta["categorical_features"]:
            X_mod[c] = X_mod[c].astype(str).fillna("unknown")
        pred_mod = float(pipeline.predict(X_mod)[0])
        X_cur = current.reindex(columns=feature_cols).copy()
        for c in meta["numeric_features"]:
            X_cur[c] = pd.to_numeric(X_cur[c], errors="coerce").fillna(0.0)
        for c in meta["categorical_features"]:
            X_cur[c] = X_cur[c].astype(str).fillna("unknown")
        pred_cur = float(pipeline.predict(X_cur)[0])
        return pred_mod - pred_cur

    while budget_left >= step and iters < max_iters:
        gains = {ch: marginal_gain(ch) for ch in channels}
        best_ch = max(gains, key=gains.get)
        best_gain = gains[best_ch]
        if best_gain <= 0:
            break
        allocation[best_ch] += step
        budget_left -= step
        current[best_ch] = pd.to_numeric(current[best_ch], errors="coerce").fillna(0.0) + step
        iters += 1

    X_final = current.reindex(columns=feature_cols).copy()
    for c in meta["numeric_features"]:
        X_final[c] = pd.to_numeric(X_final[c], errors="coerce").fillna(0.0)
    for c in meta["categorical_features"]:
        X_final[c] = X_final[c].astype(str).fillna("unknown")

    final_pred = float(pipeline.predict(X_final)[0])

    return {
        "initial_prediction": base_pred,
        "final_prediction": final_pred,
        "allocated_budget": allocation,
        "budget_left": budget_left,
        "iterations": iters
    }

if __name__ == "__main__":
    print("Demo allocation (10000, step 500):")
    print(optimize_budget(10000, step=500))
