print(">>> Starting backend...")

import os
import pickle
import numpy as np
import pandas as pd
import unicodedata
from pathlib import Path
from scipy import sparse
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from difflib import SequenceMatcher

from src.logger import logging
from src.cloud_storage.aws_storage import SimpleStorageService
from src.constants import (
    AWS_ACCESS_KEY_ID_ENV_KEY,
    AWS_SECRET_ACCESS_KEY_ENV_KEY,
    BEST_MODEL_DIR,
    BEST_MODEL_S3_DIR,
    MODEL_BUCKET_NAME,
    TRAINING_DF_FILE_NAME,
    TFIDF_MATRIX_FILE_NAME,
    TFIDF_VECTORIZER_FILE_NAME,
    COSINE_SIMILARITY_FILE_NAME
)

# -----------------------------
# LOAD MODELS & DATA (S3 first, fallback to local)
# -----------------------------

MODEL_SOURCE = os.getenv("MODEL_SOURCE", "s3").strip().lower()  # s3 | auto
FORCE_S3_DOWNLOAD = os.getenv("FORCE_S3_DOWNLOAD", "0").strip().lower() in {"1", "true", "yes"}

def _aws_creds_present() -> bool:
    return bool(os.getenv(AWS_ACCESS_KEY_ID_ENV_KEY) and os.getenv(AWS_SECRET_ACCESS_KEY_ENV_KEY))


def _download_s3_required_artifacts(bucket: str, prefix: str, local_dir: Path) -> bool:
    """Download the required model artifacts from S3 into local_dir.

    Uses direct key downloads (no prefix listing) so it works even when the IAM
    policy does not grant s3:ListBucket.
    """
    try:
        s3 = SimpleStorageService()

        required = [
            TRAINING_DF_FILE_NAME,
            TFIDF_VECTORIZER_FILE_NAME,
            TFIDF_MATRIX_FILE_NAME,
            COSINE_SIMILARITY_FILE_NAME,
        ]

        ok = True
        for filename in required:
            key = f"{prefix.rstrip('/')}/{filename}"
            local_path = local_dir / filename
            local_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                s3.s3_client.download_file(bucket, key, str(local_path))
            except Exception as e:
                ok = False
                logging.warning(f"Failed to download s3://{bucket}/{key}: {e}")

        return ok
    except Exception as e:
        logging.warning(f"S3 download failed: {e}")
        return False


def _has_required_artifacts(model_dir: Path) -> bool:
    required = [
        TRAINING_DF_FILE_NAME,
        TFIDF_VECTORIZER_FILE_NAME,
        TFIDF_MATRIX_FILE_NAME,
        COSINE_SIMILARITY_FILE_NAME,
    ]
    return all((model_dir / f).exists() for f in required)


def _missing_required_artifacts(model_dir: Path) -> list[str]:
    required = [
        TRAINING_DF_FILE_NAME,
        TFIDF_VECTORIZER_FILE_NAME,
        TFIDF_MATRIX_FILE_NAME,
        COSINE_SIMILARITY_FILE_NAME,
    ]
    return [f for f in required if not (model_dir / f).exists()]


def _resolve_model_dir() -> Path:
    cache_dir = Path("model_cache") / "best_model"

    # If running in 'auto' mode and cache is already complete (and not forced), use it.
    if MODEL_SOURCE == "auto" and not FORCE_S3_DOWNLOAD and _has_required_artifacts(cache_dir):
        logging.info("Loaded best model artifacts from local cache_dir")
        return cache_dir

    # Always try S3 first in both 's3' and 'auto' modes.
    if MODEL_SOURCE in {"s3", "auto"}:
        cache_dir.mkdir(parents=True, exist_ok=True)
        downloaded = _download_s3_required_artifacts(MODEL_BUCKET_NAME, BEST_MODEL_S3_DIR, cache_dir)
        if downloaded and _has_required_artifacts(cache_dir):
            logging.info("Loaded best model artifacts from S3")
            return cache_dir

        if MODEL_SOURCE == "s3":
            raise RuntimeError(
                "MODEL_SOURCE is set to 's3' but the model could not be downloaded from S3. "
                f"Bucket: {MODEL_BUCKET_NAME}, Prefix: {BEST_MODEL_S3_DIR}. "
                "Fix AWS auth (set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY, or configure AWS CLI credentials/AWS_PROFILE), "
                "and ensure the uploaded artifacts exist under that prefix. "
                "If you want local fallback, set MODEL_SOURCE=auto."
            )

    # In 'auto' mode only, allow local fallbacks.
    if MODEL_SOURCE == "auto":
        if _has_required_artifacts(BEST_MODEL_DIR):
            logging.info("Loaded best model artifacts from local best_model directory")
            return BEST_MODEL_DIR

        local_models_dir = Path("src/artifacts/models")
        if _has_required_artifacts(local_models_dir):
            logging.info("Loaded latest model artifacts from local models directory")
            return local_models_dir

        candidates = [cache_dir, BEST_MODEL_DIR, local_models_dir]
        details = "; ".join(
            f"{c.as_posix()} missing {', '.join(_missing_required_artifacts(c))}" for c in candidates
        )
        raise FileNotFoundError(
            "Could not locate a complete set of model artifacts. "
            f"Checked: {details}. "
            "If S3 prefix changed, update BEST_MODEL_S3_DIR/MODEL_PUSHER_S3_KEY in src/constants/__init__.py or set env BEST_MODEL_S3_DIR."
        )

    raise ValueError("Invalid MODEL_SOURCE. Use 's3' or 'auto'.")


model_dir = _resolve_model_dir()

df_path = model_dir / TRAINING_DF_FILE_NAME
tfidf_path = model_dir / TFIDF_VECTORIZER_FILE_NAME
tfidf_matrix_path = model_dir / TFIDF_MATRIX_FILE_NAME
cosine_path = model_dir / COSINE_SIMILARITY_FILE_NAME

df = pd.read_csv(df_path)

with open(tfidf_path, "rb") as f:
    tfidf = pickle.load(f)

tfidf_matrix = sparse.load_npz(tfidf_matrix_path)
cosine_sim = np.load(cosine_path)


# -----------------------------
# FIX: ensure rating_norm exists
# -----------------------------
df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(df["rating"].mean())
df["rating_norm"] = (df["rating"] - df["rating"].min()) / (df["rating"].max() - df["rating"].min())


# -----------------------------
# TEXT NORMALIZATION
# -----------------------------

def normalize_text(t):
    if not isinstance(t, str):
        return ""
    t = unicodedata.normalize("NFKD", t)
    return t.encode("ascii", "ignore").decode("utf-8").lower().strip()


df["title_norm"] = df["title"].astype(str).apply(normalize_text)


# -----------------------------
# MOVIE FINDER ENGINE
# -----------------------------

def seq_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()


def find_movie(query):
    q = normalize_text(query)

    # exact
    exact = df[df["title_norm"] == q]
    if len(exact):
        return exact["title"].iloc[0]

    # substring (safe)
    sub = df[df["title_norm"].str.contains(q, case=False, regex=False, na=False)]
    if len(sub):
        return sub.sort_values("vote_count", ascending=False)["title"].iloc[0]

    # short queries
    if len(q) <= 4:
        pre = df[df["title_norm"].str.startswith(q, na=False)]
        if len(pre):
            return pre.sort_values("vote_count", ascending=False)["title"].iloc[0]

    # fuzzy fallback
    best = None
    best_score = 0
    for _, row in df.iterrows():
        score = seq_ratio(q, row["title_norm"])
        if score > best_score:
            best_score = score
            best = row["title"]

    return best


# -----------------------------
# RECOMMENDER SYSTEM
# -----------------------------

def recommend(movie_input, top_n=10):
    title = find_movie(movie_input)
    if title is None:
        return {"error": f"Movie '{movie_input}' not found."}

    idx = df.index[df["title"] == title][0]

    # similarity list
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1: top_n * 3]

    results = []
    for movie_idx, sim in sim_scores:
        rating_norm = df.iloc[movie_idx]["rating_norm"]
        final_score = 0.7 * sim + 0.3 * rating_norm
        results.append((movie_idx, final_score))

    # sort by final score
    results = sorted(results, key=lambda x: x[1], reverse=True)[:top_n]

    movie_indices = [i[0] for i in results]

    return {
        "matched_title": title,
        "results": df.iloc[movie_indices][["title", "genres", "rating", "poster_url"]].to_dict(orient="records")
    }


# -----------------------------
# FLASK BACKEND
# -----------------------------

app = Flask(__name__)
CORS(app)


@app.route("/")
def home():
    return render_template("index.html")   # IMPORTANT (loads HTML)


@app.route("/health")
def health():
    return {"status": "ok", "movies": len(df)}


@app.route("/recommend")
def recommend_api():
    title = request.args.get("title", "")
    top_n = int(request.args.get("top_n", 10))
    return jsonify(recommend(title, top_n))


@app.route("/search")
def search_api():
    q = request.args.get("query", "").lower()
    sub = df[df["title_norm"].str.contains(q, case=False, regex=False, na=False)].head(10)
    return jsonify(sub[["title", "poster_url"]].to_dict(orient="records"))



@app.route("/suggest")
def suggest_api():
    q = request.args.get("query", "").lower().strip()

    if not q:
        return jsonify([])

    # safe substring match
    matches = df[df["title_norm"].str.contains(q, case=False, regex=False, na=False)]

    # return top 8 suggestions
    suggestions = matches["title"].head(8).tolist()

    return jsonify(suggestions)


if __name__ == "__main__":
    print("Server running at: http://127.0.0.1:5000")
    app.run(host="0.0.0.0", port=5000, debug=True)