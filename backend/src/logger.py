import os, json, pandas as pd
from datetime import datetime

class Logger:
    @staticmethod
    def log_experiment(model_name, accuracy, f1_macro, f1_micro, elapsed_time, path):
        df = pd.DataFrame([[model_name, accuracy, f1_macro, f1_micro, elapsed_time]],
                          columns=["model_name","accuracy","f1_macro","f1_micro","training_time"])
        if os.path.exists(path):
            df_old = pd.read_csv(path)
            df = pd.concat([df_old, df], ignore_index=True)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, index=False)

    @staticmethod
    def save_class_names(class_names, path):
        class_names = [str(c) for c in class_names]  # convert to strings
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w") as f:
            json.dump(class_names, f)


    @staticmethod
    def load_class_names(path):
        with open(path, "r") as f:
            return json.load(f)


