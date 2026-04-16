import os
import json
from flask import Flask, render_template, request, jsonify

from src.etl import load_and_clean
from src.feature_engineering import prepare_features
from src.predict import load_pipeline_and_meta, _ensure_X_for_pipeline, channel_contributions_finite_diff
from src.optimizer import optimize_budget
from src.retrain import run_full_retrain_cycle

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/api/time-series", methods=["GET"])
def time_series():
    try:
        df = load_and_clean()
        df = prepare_features(df)
        ts = df.tail(180).copy()
        # Convert date to string for JSON serialization
        ts["date"] = ts["date"].astype(str)
        return jsonify(ts.to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data = request.json
        spend_google = float(data.get("spend_google", 0))
        spend_facebook = float(data.get("spend_facebook", 0))
        spend_influencer = float(data.get("spend_influencer", 0))
        discount = float(data.get("discount", 0.05))
        promo_type = data.get("promo_type", "email")
        season = data.get("season", "spring")

        # Load standard defaults to fill missing fields
        df_clean = load_and_clean()
        df_clean = prepare_features(df_clean)
        median_row = df_clean.median(numeric_only=True).to_dict()

        import pandas as pd
        input_df = pd.DataFrame({
            "spend_google": [spend_google],
            "spend_facebook": [spend_facebook],
            "spend_influencer": [spend_influencer],
            "discount": [discount],
            "promo_type": [promo_type],
            "season": [season],
        })

        all_features = df_clean.columns.tolist()
        for col in all_features:
            if col not in input_df.columns:
                input_df[col] = float(median_row.get(col, 0.0))

        input_df_prepared = prepare_features(input_df)
        pipeline, meta = load_pipeline_and_meta()
        X_input = _ensure_X_for_pipeline(input_df_prepared, meta)
        
        predicted_revenue = float(pipeline.predict(X_input)[0])

        return jsonify({
            "predicted_revenue": predicted_revenue
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/channel-contributions", methods=["GET"])
def channel_contributions():
    try:
        contributions = channel_contributions_finite_diff(delta=500.0, channels=["spend_google", "spend_facebook", "spend_influencer"])
        return jsonify(contributions)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/optimize", methods=["POST"])
def optimize():
    try:
        data = request.json
        budget = float(data.get("budget", 10000))
        step_size = float(data.get("step_size", 500))
        result = optimize_budget(budget, step_size)
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/retrain", methods=["POST"])
def retrain():
    try:
        pipeline, meta = run_full_retrain_cycle()
        return jsonify([{"status": "success", "meta": meta}])
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/api/data-inspector", methods=["GET"])
def data_inspector():
    try:
        df = load_and_clean()
        df["date"] = df["date"].astype(str)
        return jsonify(df.tail(50).to_dict(orient="records"))
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
