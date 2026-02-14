import json
import os

from flask import Flask, jsonify, render_template, request

from pipeline.pipeline import FakeNewsBiLSTM

# Absolute paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
template_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

with open(os.path.join(base_dir, "config/config.json")) as f:
    cfg = json.load(f)


@app.after_request
def add_cors_headers(response):
    origin = request.headers.get("Origin", "")

    is_local = origin in {"http://127.0.0.1:5173", "http://localhost:5173"}
    is_vercel = origin.startswith("https://") and origin.endswith(".vercel.app")
    is_file_origin = origin == "null"

    if is_local or is_vercel or is_file_origin:
        response.headers["Access-Control-Allow-Origin"] = origin or "*"
        response.headers["Vary"] = "Origin"

    response.headers["Access-Control-Allow-Methods"] = "GET,POST,OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type,Authorization"
    return response


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = (request.form.get("news_text") or "").strip()
        if not text:
            return render_template(
                "result.html",
                predictions=[{"prediction": "N/A", "probability": 0.0, "text": "Empty input"}],
            )

        df_pred = FakeNewsBiLSTM.predict_text(cfg, text)
        return render_template("result.html", predictions=df_pred.to_dict(orient="records"))
    return render_template("index.html")


@app.route("/api/predict", methods=["POST", "OPTIONS"])
def api_predict():
    if request.method == "OPTIONS":
        return ("", 204)

    payload = request.get_json(silent=True) or {}
    text = str(payload.get("text", "")).strip()

    if not text:
        return jsonify({"error": "Field 'text' is required."}), 400

    df_pred = FakeNewsBiLSTM.predict_text(cfg, text)
    row = df_pred.iloc[0].to_dict()

    return jsonify(
        {
            "prediction": row.get("prediction", ""),
            "probability": float(row.get("probability", 0.0)),
            "text": row.get("text", text),
        }
    )
