import os
from pathlib import Path
from datetime import date

# ============================================================
# Project / Pipeline level constants
# ============================================================

PIPELINE_NAME: str = "movie_recommendation_pipeline"
ARTIFACT_DIR: Path = Path("src/artifacts")

CURRENT_YEAR = date.today().year

# ============================================================
# MongoDB constants
# ============================================================

DATABASE_NAME = "movie_recommender"
COLLECTION_NAME = "movies_metadata"
MONGODB_URL_KEY = "MONGODB_URL"

# ============================================================
# Artifact directories
# ============================================================

DATA_INGESTION_DIR = ARTIFACT_DIR / "data_ingestion"
DATA_VALIDATION_DIR = ARTIFACT_DIR / "data_validation"
DATA_TRANSFORMATION_DIR = ARTIFACT_DIR / "data_transformation"
MODEL_DIR = ARTIFACT_DIR / "models"

# Create dirs safely
for dir_path in [
    ARTIFACT_DIR,
    DATA_INGESTION_DIR,
    DATA_VALIDATION_DIR,
    DATA_TRANSFORMATION_DIR,
    MODEL_DIR
]:
    dir_path.mkdir(parents=True, exist_ok=True)

# ============================================================
# Data Ingestion constants
# ============================================================

DATA_INGESTION_COLLECTION_NAME = COLLECTION_NAME
INGESTED_DATA_FILE_NAME = "movies.csv"
INGESTED_DATA_PATH = DATA_INGESTION_DIR / INGESTED_DATA_FILE_NAME

# ============================================================
# Data Validation constants
# ============================================================

SCHEMA_FILE_PATH = Path("config/schema.yaml")
DATA_VALIDATION_REPORT_FILE_NAME = "validation_report.yaml"
DATA_VALIDATION_REPORT_PATH = DATA_VALIDATION_DIR / DATA_VALIDATION_REPORT_FILE_NAME

# ============================================================
# Data Transformation constants
# ============================================================

COMBINED_TEXT_COLUMN = "combined_text"

TEXT_COLUMNS = [
    "overview",
    "genres",
    "keywords",
    "cast",
    "director"
]

# ============================================================
# Model / Recommender artifacts
# ============================================================

TFIDF_VECTORIZER_FILE_NAME = "tfidf_vectorizer.pkl"
TFIDF_MATRIX_FILE_NAME = "tfidf_matrix.npz"
COSINE_SIMILARITY_FILE_NAME = "cosine_similarity.npy"

TFIDF_VECTORIZER_PATH = MODEL_DIR / TFIDF_VECTORIZER_FILE_NAME
TFIDF_MATRIX_PATH = MODEL_DIR / TFIDF_MATRIX_FILE_NAME
COSINE_SIMILARITY_PATH = MODEL_DIR / COSINE_SIMILARITY_FILE_NAME

# ============================================================
# Optional: Model Evaluation (Ranking metrics)
# ============================================================

TOP_K_RECOMMENDATIONS = 10

# ============================================================
# AWS / Cloud (optional – future extension)
# ============================================================

AWS_ACCESS_KEY_ID_ENV_KEY = "AWS_ACCESS_KEY_ID"
AWS_SECRET_ACCESS_KEY_ENV_KEY = "AWS_SECRET_ACCESS_KEY"
REGION_NAME = "us-east-1"

MODEL_BUCKET_NAME = "movie-recommender-mlops"
MODEL_PUSHER_S3_KEY = "model-registry/movie-recommender"

# ============================================================
# App / API constants
# ============================================================

APP_HOST = "0.0.0.0"
APP_PORT = 5000