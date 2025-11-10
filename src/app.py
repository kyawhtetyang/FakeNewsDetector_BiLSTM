import os
from flask import Flask, render_template, request
import json
from pipeline.pipeline import FakeNewsBiLSTM

# Absolute paths
base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
template_dir = os.path.join(base_dir, "templates")
static_dir = os.path.join(base_dir, "static")

app = Flask(__name__, template_folder=template_dir, static_folder=static_dir)

with open(os.path.join(base_dir, "config/config.json")) as f:
    cfg = json.load(f)

@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        text = request.form.get("news_text")
        df_pred = FakeNewsBiLSTM.predict_text(cfg, text)
        return render_template("result.html", predictions=df_pred.to_dict(orient="records"))
    return render_template("index.html")

