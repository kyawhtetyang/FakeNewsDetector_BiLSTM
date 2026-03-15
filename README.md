# Fake News Detector (BiLSTM)

One repository with split deployment structure:
- `backend/` Flask + TensorFlow inference API
- `frontend/` static UI (Vercel-ready) calling backend API

## Structure

```text
v0/
  backend/
    src/
    pipeline/
    tests/
    config/
    templates/
    static/
    main.py
    train.py
    predict.py
    requirements.txt
    Dockerfile
    docker-compose.yml
  frontend/
    index.html
```

## Live Demo

- Frontend (Vercel): `https://fake-news-detector-bi-lstm-bx1an8c5p.vercel.app`
- Backend (Render): `https://fakenewsdetector-bilstm.onrender.com`

## Setup Notes

- NLTK data is required for preprocessing. Download once:

```bash
python -m nltk.downloader stopwords wordnet
```

- Dataset CSV is not included in this repo. Place it at `backend/data/raw/fake_news.csv`.
- Required columns: `text`, `label` (0 = fake, 1 = real).

To generate the cleaned dataset:

```bash
cd backend
python generate_dataset.py
```

- Model artifacts are generated locally and not tracked in git.
  Files: `backend/output/models/*.h5`, `backend/output/tokenizer.pkl`.

## Local Run (Backend)

```bash
cd backend
conda activate tf
pip install -r requirements.txt
python main.py
```

Backend default URL: `http://127.0.0.1:5001`

### API

`POST /api/predict`

Request:
```json
{ "text": "Your news article text" }
```

Response:
```json
{ "prediction": "Fake", "probability": 0.91, "text": "..." }
```

## Local Run (Frontend)

Run a local static server for frontend:

```bash
cd frontend
python -m http.server 5173
```

Then open `http://127.0.0.1:5173` and keep backend URL as `http://127.0.0.1:5001`.

## Deployment

- Frontend: Vercel (root directory: `frontend`)
- Backend: Render (root directory: `backend`)

## Tests

```bash
cd backend
conda run -n tf pytest -q
```
