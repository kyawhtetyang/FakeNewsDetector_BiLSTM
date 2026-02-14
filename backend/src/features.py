from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pickle, os

class Features:
    def __init__(self, max_words=10000, max_len=150):
        self.max_words = max_words
        self.max_len = max_len
        self.tokenizer = None

    def fit_tokenizer(self, texts):
        self.tokenizer = Tokenizer(num_words=self.max_words)
        self.tokenizer.fit_on_texts(texts)

    def transform(self, texts):
        seqs = self.tokenizer.texts_to_sequences(texts)
        return pad_sequences(seqs, maxlen=self.max_len)

    def save(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self.tokenizer, f)

    def load(self, path):
        with open(path, "rb") as f:
            self.tokenizer = pickle.load(f)

    def split_features(self, df, label_col="label"):
        class_names = sorted(df[label_col].unique())
        label_map = {l:i for i,l in enumerate(class_names)}
        y = df[label_col].map(label_map).values
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(df["clean_text"], y, test_size=0.2, random_state=42)
        self.fit_tokenizer(X_train_raw)
        X_train = self.transform(X_train_raw)
        X_test = self.transform(X_test_raw)
        return class_names, X_train, X_test, y_train, y_test

