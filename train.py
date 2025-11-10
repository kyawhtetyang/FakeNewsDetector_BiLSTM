import json
from pipeline.pipeline import FakeNewsBiLSTM
with open("config/config.json") as f:
    cfg = json.load(f)
FakeNewsBiLSTM.train_pipeline(cfg)

