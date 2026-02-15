import pandas as pd
from src.preprocess import PreProcessor
from src.features import Features
from src.model import BiLSTMMODEL
from src.evaluation import Evaluator
from src.logger import Logger
from src.utils import Utils
import os

class FakeNewsBiLSTM:
    _inference_model = None
    _inference_feat = None
    _inference_class_names = None

    @staticmethod
    def check_or_load(cfg):
        if Utils.check_paths(cfg["model_path"], cfg["tokenizer_path"]):
            model, feat = Utils.load_existing(cfg["model_path"], cfg["tokenizer_path"])
            return model, feat
        return None, None

    @staticmethod
    def load_inference_artifacts(cfg):
        """Load and cache inference artifacts once per process."""
        if (
            FakeNewsBiLSTM._inference_model is not None
            and FakeNewsBiLSTM._inference_feat is not None
            and FakeNewsBiLSTM._inference_class_names is not None
        ):
            return (
                FakeNewsBiLSTM._inference_model,
                FakeNewsBiLSTM._inference_feat,
                FakeNewsBiLSTM._inference_class_names,
            )

        feat = Features(max_words=cfg["max_words"], max_len=cfg["max_len"])
        feat.load(cfg["tokenizer_path"])
        model = BiLSTMMODEL.load(cfg["model_path"])
        class_names = Logger.load_class_names(cfg["class_names_path"])

        FakeNewsBiLSTM._inference_model = model
        FakeNewsBiLSTM._inference_feat = feat
        FakeNewsBiLSTM._inference_class_names = class_names
        return model, feat, class_names

    @staticmethod
    def build_train_model(cfg, X_train, y_train, X_test, y_test, num_classes):
        model = BiLSTMMODEL.build(max_words=cfg["max_words"], max_len=cfg["max_len"], num_classes=num_classes)
        history, elapsed = BiLSTMMODEL.train(model, X_train, y_train, X_test, y_test, epochs=cfg["epochs"], batch_size=cfg["batch_size"])
        BiLSTMMODEL.save(model, cfg["model_path"])
        return model, history, elapsed

    @staticmethod
    def train_pipeline(cfg):
        model, feat = FakeNewsBiLSTM.check_or_load(cfg)
        if model is not None:
            print("Found existing model & tokenizer. Skipping training.")
            return model, feat
        pre = PreProcessor()
        df = pre.load_file(cfg["processed_csv"])
        df["clean_text"] = df["text"].apply(pre.clean)
        feat = Features(max_words=cfg["max_words"], max_len=cfg["max_len"])
        class_names, X_train, X_test, y_train, y_test = feat.split_features(df)
        feat.save(cfg["tokenizer_path"])
        Logger.save_class_names(class_names, cfg["class_names_path"])
        model, history, elapsed = FakeNewsBiLSTM.build_train_model(cfg, X_train, y_train, X_test, y_test, len(class_names))
        y_pred = Evaluator.evaluate(model, X_test, y_test, class_names)
        return model, feat, class_names

    @staticmethod
    def predict_pipeline(cfg):
        pre = PreProcessor()
        df = pre.load_file(cfg["to_predict_csv"])
        df["clean_text"] = df["text"].apply(pre.clean)
        feat = Features(max_words=cfg["max_words"], max_len=cfg["max_len"])
        feat.load(cfg["tokenizer_path"])
        model = BiLSTMMODEL.load(cfg["model_path"])
        seqs = feat.transform(df["clean_text"])
        class_names = Logger.load_class_names(cfg["class_names_path"])
        results = Evaluator.batch_predict(model, seqs, class_names)
        results["text"] = df["text"]
        os.makedirs(os.path.dirname(cfg["to_predict_csv"]), exist_ok=True)
        results.to_csv("output/predictions.csv", index=False)
        return results

    @staticmethod
    def predict_text(cfg, text):
        """Predict a single input text."""
        from src.preprocess import PreProcessor
        from src.evaluation import Evaluator

        pre = PreProcessor()
        clean_text = pre.clean(text)
        model, feat, class_names = FakeNewsBiLSTM.load_inference_artifacts(cfg)
        seq = feat.transform([clean_text])  # list of one

        df = Evaluator.batch_predict(model, seq, class_names)
        df["text"] = text
        return df
