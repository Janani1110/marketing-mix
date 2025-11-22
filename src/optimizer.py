"""
src/optimizer.py

Simple budget optimizer that uses channel ROI approximations derived from model coeffs
and allocates budget to maximize estimated revenue given a total budget.

This is a linear approximation (sufficient for a demo). For more realistic non-linear
response curves, you can use non-linear solvers or discretization.
"""

import yaml
import pickle
import os
from ortools.linear_solver import pywraplp
from src.predict import load_model_and_meta  # note: placeholder to ensure import; not used

CONFIG_PATH = "config/config.yaml"
MODEL_FILE = "models/mmm_model.pkl"
META_FILE = "models/model_metadata.yaml"

def load_model_coefficients():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(META_FILE, "r") as f:
        meta = yaml.safe_load(f)
    feature_cols = meta["feature_cols"]
    coefs = dict(zip(feature_cols, model.coef_))
    return coefs

def approximate_channel_roi():
    """
    Build a very rough ROI per unit spend for each channel by averaging the coefficients
    for that channel's related features. This is an approximation for demo purposes.
    """
    coefs = load_model_coefficients()
    channels = {
        "google_ads": ["google_ads_adstock", "google_ads_sat", "google_ads_7d_mean"],
        "facebook_ads": ["facebook_ads_adstock", "facebook_ads_sat", "facebook_ads_7d_mean"],
        "influencer": ["influencer_adstock", "influencer_sat", "influencer_7d_mean"],
    }
    rois = {}
    for ch, feats in channels.items():
        vals = [coefs.get(f, 0.0) for f in feats]
        # average positive coefficients (if negative, floor to small positive)
        avg = float(sum(vals) / (len(vals) if len(vals) else 1))
        rois[ch] = max(0.0001, avg)
    return rois

def optimize_budget(total_budget):
    rois = approximate_channel_roi()
    solver = pywraplp.Solver.CreateSolver("GLOP")
    g = solver.NumVar(0, total_budget, "google")
    f = solver.NumVar(0, total_budget, "facebook")
    i = solver.NumVar(0, total_budget, "influencer")
    solver.Add(g + f + i <= total_budget)

    objective = solver.Objective()
    objective.SetCoefficient(g, rois["google_ads"])
    objective.SetCoefficient(f, rois["facebook_ads"])
    objective.SetCoefficient(i, rois["influencer"])
    objective.SetMaximization()

    solver.Solve()
    return {
        "google_ads": g.solution_value(),
        "facebook_ads": f.solution_value(),
        "influencer": i.solution_value()
    }

if __name__ == "__main__":
    cfg = yaml.safe_load(open(CONFIG_PATH))
    budget = cfg["optimizer"].get("budget", 10000)
    allocation = optimize_budget(budget)
    print(f"Budget allocation for total budget {budget}:")
    print(allocation)
