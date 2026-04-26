import os
import json
import logging
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, jsonify, redirect, url_for, flash
from werkzeug.utils import secure_filename

from model.train_model import train
from utils.preprocess import load_and_clean, encode_and_scale

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.secret_key = "churnprediction_secret_2024"
app.config["UPLOAD_FOLDER"] = os.path.join(os.path.dirname(__file__), "data")
app.config["MAX_CONTENT_LENGTH"] = 50 * 1024 * 1024

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model", "churn_model.pkl")
DEFAULT_DATA = os.path.join(os.path.dirname(__file__), "data", "TelcoCustomerChurn.csv")

_artifact = None


def load_artifact():
    global _artifact
    if _artifact is None and os.path.exists(MODEL_PATH):
        _artifact = joblib.load(MODEL_PATH)
    return _artifact


@app.route("/")
def index():
    artifact = load_artifact()
    stats = {}
    if artifact:
        stats = {
            "total_customers": artifact.get("total_customers", 0),
            "churned_customers": artifact.get("churned_customers", 0),
            "churn_rate": round(artifact.get("churn_rate", 0) * 100, 1),
            "best_model": artifact.get("best_model_name", "—"),
            "auc": artifact["all_metrics"][artifact["best_model_name"]]["roc_auc"],
        }
    return render_template("index.html", stats=stats, model_ready=artifact is not None)


@app.route("/train", methods=["POST"])
def retrain():
    uploaded = request.files.get("dataset")
    if uploaded and uploaded.filename.endswith(".csv"):
        filename = secure_filename(uploaded.filename)
        save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
        uploaded.save(save_path)
        data_path = save_path
    else:
        data_path = DEFAULT_DATA

    global _artifact
    try:
        _artifact = train(data_path)
        flash("Model trained successfully!", "success")
    except Exception as e:
        logger.exception("Training failed")
        flash(f"Training failed: {str(e)}", "error")

    return redirect(url_for("index"))


@app.route("/dashboard")
def dashboard():
    artifact = load_artifact()
    if not artifact:
        flash("Please train a model first.", "warning")
        return redirect(url_for("index"))

    charts = artifact.get("charts", {})
    print("CHART KEYS:", list(charts.keys()))          # add this
    print("JSON LENGTH:", len(json.dumps(charts)))     # add this

    metrics = artifact.get("all_metrics", {})
    return render_template(
        "dashboard.html",
        charts_json=json.dumps(charts),
        metrics_json=json.dumps(metrics),
        best_model=artifact.get("best_model_name"),
        stats={
            "total": artifact.get("total_customers", 0),
            "churned": artifact.get("churned_customers", 0),
            "churn_rate": round(artifact.get("churn_rate", 0) * 100, 1),
        },
    )


@app.route("/predict", methods=["GET", "POST"])
def predict():
    artifact = load_artifact()
    if not artifact:
        flash("Please train a model first.", "warning")
        return redirect(url_for("index"))

    feature_names = artifact["feature_names"]

    if request.method == "POST":
        try:
            raw = {f: request.form.get(f, "0") for f in feature_names}
            df_input = pd.DataFrame([raw])

            for col in df_input.columns:
                if col in artifact["encoders"]:
                    le = artifact["encoders"][col]
                    val = df_input[col].astype(str).iloc[0]
                    if val in le.classes_:
                        df_input[col] = le.transform([val])
                    else:
                        df_input[col] = 0
                else:
                    try:
                        df_input[col] = pd.to_numeric(df_input[col])
                    except Exception:
                        df_input[col] = 0

            df_input = df_input[feature_names].astype(float)
            df_scaled_arr = artifact["scaler"].transform(df_input)
            df_scaled = pd.DataFrame(df_scaled_arr, columns=feature_names)
            model = artifact["model"]
            prob = float(model.predict_proba(df_scaled)[0][1])
            pred = int(model.predict(df_scaled)[0])

            risk = "Low" if prob < 0.35 else ("Medium" if prob < 0.65 else "High")
            risk_color = {"Low": "#10b981", "Medium": "#f59e0b", "High": "#ef4444"}[risk]

            result = {
                "prediction": pred,
                "probability": round(prob * 100, 1),
                "risk": risk,
                "risk_color": risk_color,
            }
            return render_template("prediction.html", feature_names=feature_names,
                                   encoders=artifact["encoders"], result=result,
                                   form_data=request.form)
        except Exception as e:
            logger.exception("Prediction error")
            flash(f"Prediction error: {str(e)}", "error")

    return render_template("prediction.html", feature_names=feature_names,
                           encoders=artifact["encoders"], result=None, form_data={})


@app.route("/insights")
def insights():
    artifact = load_artifact()
    if not artifact:
        flash("Please train a model first.", "warning")
        return redirect(url_for("index"))

    metrics = artifact.get("all_metrics", {})
    charts = artifact.get("charts", {})
    return render_template(
        "insights.html",
        metrics=metrics,
        best_model=artifact.get("best_model_name"),
        charts_json=json.dumps(charts),
    )


@app.route("/api/charts")
def api_charts():
    artifact = load_artifact()
    if not artifact:
        return jsonify({"error": "no model"}), 404
    return jsonify(artifact.get("charts", {}))


if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        logger.info("No model found. Training on default dataset...")
        _artifact = train(DEFAULT_DATA)
    app.run(debug=True, port=5000)
