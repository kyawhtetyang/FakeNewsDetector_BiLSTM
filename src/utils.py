import os, pickle
from src.model import BiLSTMMODEL
from src.features import Features

class Utils:
    @staticmethod
    def check_paths(model_path, tokenizer_path):
        return os.path.exists(model_path) and os.path.exists(tokenizer_path)

    @staticmethod
    def load_existing(model_path, tokenizer_path):
        feat = Features()
        feat.load(tokenizer_path)
        model = BiLSTMMODEL.load(model_path)
        return model, feat

