import pandas as pd
from src.preprocess import PreProcessor
from pipeline.pipeline import FakeNewsBiLSTM

def test_fakenews_bilstm_pipeline_train_and_predict(tmp_path):
    # ------------------------
    # Step 1: Prepare dummy training dataset
    # ------------------------
    train_path = tmp_path / "dummy_fake_news.csv"
    df_train = pd.DataFrame({
        "text": ["This is real news", "This is fake news"],
        "label": [1, 0]
    })
    df_train.to_csv(train_path, index=False)

    # ------------------------
    # Step 2: Prepare dummy prediction dataset
    # ------------------------
    predict_path = tmp_path / "dummy_predict.csv"
    df_predict = pd.DataFrame({
        "text": [
            "Scientists discover a new species of frog in the Amazon rainforest.",
            "Breaking news: Giant asteroid is heading towards Earth next week.",
            "Local bakery wins award for best sourdough bread in the country.",
            "Government plans to increase funding for renewable energy projects.",
            "Man claims he can talk to animals and demonstrates it on TV."
        ]
    })
    df_predict.to_csv(predict_path, index=False)

    # ------------------------
    # Step 3: Dummy configuration
    # ------------------------
    config = {
        "processed_csv": str(tmp_path / "processed.csv"),
        "model_path": str(tmp_path / "fake_news_lstm.h5"),
        "tokenizer_path": str(tmp_path / "tokenizer.pkl"),
        "class_names_path": str(tmp_path / "class_names.json"),
        "to_predict_csv": str(predict_path),  # now defined
        "max_words": 500,
        "max_len": 20,
        "embedding_dim": 8,
        "lstm_units": 4,
        "epochs": 1,
        "batch_size": 2
    }

    # ------------------------
    # Step 4: Preprocess training data
    # ------------------------
    processor = PreProcessor()
    df_train["clean_text"] = df_train["text"].apply(processor.clean)
    df_train.to_csv(config["processed_csv"], index=False)

    # ------------------------
    # Step 5: Train the model
    # ------------------------
    model, feat, class_names = FakeNewsBiLSTM.train_pipeline(config)

    # ------------------------
    # Step 6: Predict on dummy dataset
    # ------------------------
    predictions = FakeNewsBiLSTM.predict_pipeline(config)

    # ------------------------
    # Step 7: Assertions
    # ------------------------
    assert isinstance(predictions, pd.DataFrame)
    assert "prediction" in predictions.columns
    assert "text" in predictions.columns
    assert len(predictions) == len(df_predict)

# ------------------------
# Case Studies:
# ------------------------
"""
Preprocessing:
Input: "Breaking!!! COVID cases 2025!!!"
Output: "breaking covid cases 2025"

Model:
Checks BiLSTM forward pass outputs valid probabilities for binary classification.

Pipeline:
Verifies that FakeNewsBiLSTM pipeline can train and predict on dummy dataset successfully.
"""


