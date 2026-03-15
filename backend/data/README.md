# Dataset Notes

This project expects a raw dataset CSV at:

- `backend/data/raw/fake_news.csv`

The CSV should contain at least:
- `text` (news article text)
- `label` (0 = fake, 1 = real)

To generate the cleaned dataset used for training:

```bash
cd backend
python generate_dataset.py
```

This will create:

- `backend/data/processed/fake_news_clean.csv`

Notes:
- The dataset file is not included in this repo.
- Store your local CSVs under `backend/data/raw/` (ignored by git).
