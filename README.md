# Fake News Detector v9 (BiLSTM + Flask + Docker)

**Project Overview**
An end-to-end fake news detection system that combines **machine learning, data preprocessing, web deployment, and Docker containerization**. Demonstrates the full ML lifecycle from raw data to production-ready API with a minimal web interface.

---

## Features
End-to-End ML Pipeline
- Data ingestion & preprocessing: cleans text, removes stopwords, lemmatizes.
- Feature engineering: tokenizer + padded sequences.
- BiLSTM-based classification for binary or multi-class fake news detection.
- Automatic saving/loading of model, tokenizer, and class labels.

Evaluation & Logging
- Confusion matrix and classification report visualization.
- Experiment logging: tracks accuracy, F1 scores, and training time.
- Unit tests validate preprocessing, model, and pipeline functionality.

Deployment Ready
- Flask web interface for interactive predictions.
- Accepts user input for single articles or batch prediction.
- Fully containerized with Docker & docker-compose for easy deployment.

Clean & Modular Codebase
- Follows real-world software engineering principles: modular structure (src/, pipeline/, tests/).
- Clear separation of responsibilities: preprocessing, features, model, evaluation, logging.
- Configurable via config/config.json for flexible experimentation.

Production-Oriented Extras
- Docker ensures environment reproducibility.
- Ready for scaling to larger datasets.
- Supports model versioning and experiment tracking.

---

## Folder Structure
```php
v9/
├─ config/                # Model & pipeline configuration
├─ data/raw & processed/   # Raw and cleaned datasets
├─ src/                    # Core modules (preprocess, features, model, evaluation, utils, app)
├─ templates/              # Flask HTML templates
├─ static/                 # CSS & JS for frontend
├─ models/                 # Saved models
├─ output/                 # Predictions, tokenizer, experiment logs
├─ pipeline/               # Training & prediction pipeline
├─ tests/                  # Unit tests for preprocessing, model, and pipeline
├─ generate_dataset.py     # Prepares cleaned dataset
├─ train.py                # Trains BiLSTM model
├─ predict.py              # Runs prediction pipeline
├─ main.py                 # Flask app entry point
├─ Dockerfile
├─ docker-compose.yml
└─ requirements.txt
```
---

## Setup
### Local Python Environment
```bash
cd ~/FakeNewsDetector_BiLSTM/v9
conda create -n tf python=3.11 -y
conda activate tf
pip install -r requirements.txt

# Prepare dataset
python generate_dataset.py

# Train model
python train.py

# Predict new articles
python predict.py

# Launch Flask web interface
python main.py
# Flask runs on http://127.0.0.1:5001 by default

### Docker Deployment
# Build and run container
docker-compose build
docker-compose up

# Or individual commands
docker build -t fake-news-api-web .
docker run -p 5001:5001 fake-news-api-web


```
