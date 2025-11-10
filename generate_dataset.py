import pandas as pd, os
from src.preprocess import PreProcessor

RAW_PATH = "data/raw/fake_news.csv"
CLEAN_PATH = "data/processed/fake_news_clean.csv"

def prepare_dataset():
	# Ensure the processed folder exists
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    df = pd.read_csv(RAW_PATH)
    processor = PreProcessor()
    df["clean_text"] = df["text"].astype(str).apply(processor.clean)
    df.to_csv(CLEAN_PATH, index=False)
    print(f"✅ Dataset prepared: {CLEAN_PATH}")
    return df

if __name__ == "__main__":
    prepare_dataset()


