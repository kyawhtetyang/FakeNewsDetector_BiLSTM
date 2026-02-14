# Fake News Detector v9 (BiLSTM)

One repository with split deployment structure:
- `backend/` Flask + TensorFlow inference API
- `frontend/` static UI (Vercel-ready) calling backend API

## Structure

```text
v9/
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
    vercel.json
```

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

Open `frontend/index.html` in browser and set backend URL field if needed.

## Deployment

- Frontend: Vercel (root directory: `frontend`)
- Backend: Render/VPS (root directory: `backend`)

## Tests

```bash
cd backend
conda run -n tf pytest -q
```
