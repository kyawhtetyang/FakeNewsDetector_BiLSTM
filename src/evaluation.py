import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class Evaluator:
    @staticmethod
    def evaluate(model, X_test, y_test, class_names=None):
        # Ensure class_names are strings
        if class_names is not None:
            class_names = [str(c) for c in class_names]

        y_pred_prob = model.predict(X_test)
        num_classes = model.output_shape[-1]

        if num_classes == 1:
            y_pred = (y_pred_prob >= 0.5).astype(int)
        else:
            y_pred = np.argmax(y_pred_prob, axis=1)

        print(classification_report(y_test, y_pred, target_names=class_names))

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6, 5))
        sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix")
        plt.show()

        return y_pred

    @staticmethod
    def batch_predict(model, seqs, class_names=None):
        num_classes = model.output_shape[-1]
        pred_probs = model.predict(seqs)

        if num_classes == 1:
            # Binary classification
            preds = (pred_probs >= 0.5).astype(int).flatten()
            labels = class_names if class_names else ["Real", "Fake"]
            df = pd.DataFrame({
                "prediction": [labels[p] for p in preds],
                "probability": pred_probs.flatten()
            })
        else:
            # Multi-class classification
            preds = pred_probs.argmax(axis=1)
            if not class_names:
                raise ValueError("class_names must be provided for multi-class predictions")
            df = pd.DataFrame({
                "prediction": [class_names[p] for p in preds],
                "probability": pred_probs.max(axis=1)
            })

        return df


