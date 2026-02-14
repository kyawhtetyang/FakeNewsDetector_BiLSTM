import os
import pandas as pd
import re
import string

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# Ensure Render/runtime NLTK data directory is checked first
nltk_data_dir = os.getenv("NLTK_DATA")
if nltk_data_dir and nltk_data_dir not in nltk.data.path:
    nltk.data.path.insert(0, nltk_data_dir)


def _resource_exists(resource: str) -> bool:
    try:
        nltk.data.find(resource)
        return True
    except LookupError:
        return False


# stopwords are required; wordnet is optional (lemmatization fallback)
HAS_STOPWORDS = _resource_exists("corpora/stopwords")
HAS_WORDNET = _resource_exists("corpora/wordnet")

if not HAS_STOPWORDS:
    raise RuntimeError(
        "Missing required NLTK resource: corpora/stopwords. "
        "Install it before running, e.g. python -m nltk.downloader stopwords"
    )


class PreProcessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer() if HAS_WORDNET else None

    def load_file(self, path):
        return pd.read_csv(path)

    def clean(self, text):
        text = str(text).lower()
        text = re.sub(r"http\S+|www\S+", "", text)
        text = re.sub(r"#\w+", "", text)
        text = re.sub(r"\d+", "", text)
        text = text.translate(str.maketrans("", "", string.punctuation))

        words = []
        for token in text.split():
            if token in self.stop_words:
                continue
            if self.lemmatizer is not None:
                token = self.lemmatizer.lemmatize(token)
            words.append(token)

        return " ".join(words)
