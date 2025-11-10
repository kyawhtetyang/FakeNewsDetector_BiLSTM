import os, pickle, pandas as pd, re, string
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

try:
    import nltk
    nltk.data.find("corpora/stopwords")
    nltk.data.find("corpora/wordnet")
except Exception:
    import nltk
    nltk.download("stopwords", quiet=True)
    nltk.download("wordnet", quiet=True)

class PreProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()

    def load_file(self, path):
        return pd.read_csv(path)

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))
        words = [self.lemmatizer.lemmatize(w) for w in text.split() if w not in self.stop_words]
        return " ".join(words)

