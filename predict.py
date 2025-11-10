import json
from pipeline.pipeline import FakeNewsBiLSTM
with open("config/config.json") as f:
    cfg = json.load(f)
res = FakeNewsBiLSTM.predict_pipeline(cfg)
print(res.head())


