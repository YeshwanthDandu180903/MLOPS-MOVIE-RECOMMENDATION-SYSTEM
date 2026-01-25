\
# 🚀 MLOps Movie Recommendation System (TF‑IDF + Cosine Similarity)

> An end-to-end MLOps-style recommender system that ingests movie metadata from MongoDB, validates + transforms it into text features, trains a TF‑IDF semantic similarity model, and serves recommendations via a Flask web UI + JSON APIs. Includes optional AWS S3 model registry + runtime artifact download.

---

## 📋 Overview

This project demonstrates a practical, production-minded recommendation pipeline:

- **Offline pipeline** to build a semantic model (TF‑IDF vectorizer + similarity matrix)
- **Artifact management** (training dataframe + vectorizer + sparse matrix + cosine similarity)
- **“Best model” packaging** and **optional S3 upload** for deployment-ready artifacts
- **Online serving** (Flask UI + APIs) with **robust movie search** (exact/substr/fuzzy)

It’s designed to be reproducible, modular, and easy to deploy locally or in containers.

---

## 🏗️ Architecture & Components

### **Training / Offline Pipeline**

**1) Data Ingestion (MongoDB → CSV artifact)**
- Fetches movie documents from MongoDB collection
- Writes the raw dataset to an artifact path under `src/artifacts/data_ingestion/`

**2) Data Validation (Schema checks)**
- Uses `config/schema.yaml` to verify required columns exist
- Logs extra columns (allowed)
- Generates a validation report under `src/artifacts/data_validation/`

**3) Data Transformation (Feature engineering for recommender)**
- Drops unused columns
- Handles null values in key text fields
- Converts list-like fields (genres/cast/keywords) into space-joined text
- Builds the training feature:
	- `combined_text = overview + genres + keywords + cast + director`
- Creates search helpers:
	- `title_norm`, `title_tokens`
- Normalizes rating:
	- `rating_norm` used for ranking/boosting at inference time

**4) Recommender Training (TF‑IDF + cosine similarity)**
- Trains `TfidfVectorizer(stop_words="english", max_features=5000)`
- Builds `tfidf_matrix` (sparse) and `cosine_similarity` matrix
- Saves artifacts into `src/artifacts/models/`

**5) Best Model Packaging + (Optional) Registry Upload**
- Copies the latest trained artifacts into `src/artifacts/models/best_model/`
- Uploads best_model artifacts to S3 (optional) via `ModelPusher`

### **Serving / Online Prediction**

**Flask UI + JSON API**
- Loads recommender artifacts at startup
- Can load artifacts from:
	- **S3** (default) into `model_cache/best_model/`
	- **Local fallback** when configured
- Provides:
	- HTML UI (`templates/index.html`)
	- API responses via JSON

**Search & Matching Engine**
User input is normalized and resolved via:
- exact match
- substring match
- prefix match (short queries)
- fuzzy fallback (SequenceMatcher / token heuristics)

---

## 🧠 Recommendation Method (Important Concepts)

### **Content-Based Filtering**
This is a **content-based recommender** (not collaborative filtering). It recommends movies similar to a given movie based on text metadata.

### **TF‑IDF Vectorization**
- Converts `combined_text` into numeric vectors
- Downweights common words and upweights rare, informative terms

### **Cosine Similarity**
- Measures similarity between TF‑IDF vectors
- Produces an $N \times N$ similarity matrix for fast lookup

### **Ranking Blend (Similarity + Rating Boost)**
In serving (`app.py`), the final ranking score blends:
- semantic similarity (cosine)
- normalized rating (`rating_norm`) for quality bias

---

## 🛠️ Tech Stack

### Core
- **Language**: Python 3.10
- **Recommender / ML**: scikit-learn (TF‑IDF, cosine similarity)
- **Data**: pandas, NumPy, SciPy (sparse matrices)

### Backend / Serving
- **Web App**: Flask + Jinja2 templates
- **CORS**: flask-cors

### Data + Cloud (Optional)
- **MongoDB**: pymongo (data ingestion)
- **AWS S3**: boto3 (artifact registry + runtime download)

### Ops / Engineering Practices
- Modular pipeline stages (ingestion/validation/transformation/training/pusher)
- Configuration via constants + YAML schema
- Structured logging across stages

---

## 🚀 Quick Start

### 1) Create Environment & Install Dependencies

```bash
python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt
```

### 2) Configure MongoDB (for Training)

Set the MongoDB connection string as an environment variable:

```bash
set MONGODB_URL=mongodb+srv://<user>:<pass>@<cluster>/<db>?retryWrites=true&w=majority
```

Your pipeline uses:
- Database: `movie_recommender`
- Collection: `movies_metadata`

### 3) Run Training Pipeline

This executes ingestion → validation → transformation → training → best_model packaging → (optional) S3 push.

```bash
python demo.py
```

### 4) Run the Web App (Serving)

```bash
python app.py
```

Open in browser:
- http://localhost:5000

---

## ⚙️ Configuration

### Schema
- `config/schema.yaml` defines required columns like `title`, `overview`, `genres`, etc.

### Key Environment Variables

**MongoDB**
- `MONGODB_URL` – required for training ingestion

**S3 Model Registry (optional)**
- `AWS_ACCESS_KEY_ID`
- `AWS_SECRET_ACCESS_KEY`
- `AWS_DEFAULT_REGION` (or the region used in your AWS config)
- `MODEL_BUCKET_NAME` (default: `mlops-movie-recommender`)
- `MODEL_PUSHER_S3_KEY` (default: `model-registry/movie-recommender`)
- `BEST_MODEL_S3_DIR` (derived from `MODEL_PUSHER_S3_KEY/best_model` unless overridden)

**Serving artifact source**
- `MODEL_SOURCE`: `s3` (default) or `auto`
	- `s3`: must download from S3
	- `auto`: try S3, else use local artifacts
- `FORCE_S3_DOWNLOAD`: `1/true/yes` to force re-download

---

## 📊 Pipeline Workflow (End-to-End)

### Data Ingestion
```
MongoDB → DataFrame → src/artifacts/data_ingestion/movies.csv
```

### Data Validation
```
movies.csv → schema.yaml checks → validation_report.yaml
```

### Data Transformation
```
movies.csv → clean + feature engineering → combined_text + search fields → transformed CSV
```

### Model Training
```
combined_text → TF‑IDF → tfidf_matrix → cosine_similarity.npy
```

### Best Model Packaging + Upload
```
latest artifacts → src/artifacts/models/best_model/ → (optional) S3
```

---

## 🐳 Docker

Build:
```bash
docker build -t movie-recommender .
```

Run:
```bash
docker run -p 5000:5000 \
	-e MODEL_SOURCE=auto \
	movie-recommender
```

If you want S3 loading inside Docker, pass AWS credentials as env vars.

---

## 📁 Project Structure

```
MLOPS-MOVIE-RECOMMENDATION-SYSTEM/
├── app.py
├── demo.py
├── Dockerfile
├── requirements.txt
├── config/
│   ├── model.yaml
│   └── schema.yaml
├── model_cache/
│   └── best_model/                 # runtime cache for S3 downloads
├── src/
│   ├── artifacts/                  # pipeline outputs (timestamped runs + stage dirs)
│   ├── cloud_storage/              # AWS S3 helper
│   ├── components/
│   │   ├── data_ingestion.py
│   │   ├── data_validation.py
│   │   ├── data_transformation.py
│   │   ├── recommender_trainer.py
│   │   ├── recommender_evaluation.py
│   │   └── model_pusher.py
│   ├── configuration/
│   │   ├── aws_connection.py
│   │   └── mongo_db_connection.py
│   ├── constants/
│   │   └── __init__.py
│   ├── data_access/
│   │   └── proj1_data.py
│   ├── entity/                     # configs + artifacts dataclasses
│   ├── exception/
│   ├── logger/
│   ├── pipline/
│   │   ├── training_pipeline.py
│   │   └── prediction_pipeline.py
│   └── utils/
├── templates/
│   └── index.html
└── static/
		└── style.css
```

---

## 🔐 Security & Best Practices

- Store credentials in environment variables (never commit keys)
- Keep MongoDB Atlas network rules tight when deploying
- Prefer IAM roles (on EC2/ECS) instead of static AWS keys
- Use S3 versioning for model registry buckets (recommended)
- Log pipeline outputs for reproducibility and debugging

---

## 🧪 Model Evaluation (Optional)

There is an evaluation component (`RecommenderEvaluation`) that can compute:
- Precision@K / Recall@K / F1@K on similarity-based pseudo-ground-truth
- Track best model metrics in `best_metrics.json`

Note: In the current pipeline, best_model artifacts are always updated to the latest trained model (evaluation gate is currently commented out in the pipeline).

---

## 🧯 Troubleshooting

**1) MongoDB ingestion fails**
- Ensure `MONGODB_URL` is set and the cluster allows your IP.

**2) Serving fails with missing artifacts**
- If `MODEL_SOURCE=s3`, ensure S3 bucket/prefix contains:
	- `training_df.csv`
	- `tfidf_vectorizer.pkl`
	- `tfidf_matrix.npz`
	- `cosine_similarity.npy`
- Or use `MODEL_SOURCE=auto` to allow local fallback.

**3) Docker port mismatch**
- App runs on port `5000` by default; expose/map `5000:5000`.

---

## 📞 Contact

If you’d like, I can also add:
- a GitHub Actions workflow for Docker build + deployment
- a `docker-compose.yml` for local dev
- a minimal `/health` endpoint + structured API docs

