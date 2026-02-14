import pytest
import pandas as pd
from src.preprocess import PreProcessor

import pytest
def test_text_cleaning_removes_punctuation_and_lowercases():
    processor = PreProcessor()
    text = "Breaking!!! COVID cases 2025!!!"
    cleaned = processor.clean(text)

    assert isinstance(cleaned, str)
    assert "!" not in cleaned
    assert cleaned == cleaned.lower()

def test_dataframe_cleaning_adds_clean_text_column(tmp_path):
    df = pd.DataFrame({"text": ["Hello!!! WORLD??", "Fake news???"]})
    csv_path = tmp_path / "dummy.csv"
    df.to_csv(csv_path, index=False)

    processor = PreProcessor()
    df["clean_text"] = df["text"].apply(processor.clean)

    assert "clean_text" in df.columns
    assert len(df["clean_text"]) == 2


