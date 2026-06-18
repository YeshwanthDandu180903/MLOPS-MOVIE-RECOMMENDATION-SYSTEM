# 🚀 MLOps Movie Recommendation System

<div align="center">

[![Python](https://img.shields.io/badge/Python-3.10-blue?logo=python&logoColor=white)](https://www.python.org/)
[![Flask](https://img.shields.io/badge/Flask-v2.x-lightgrey?logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
[![MongoDB](https://img.shields.io/badge/MongoDB-Atlas-green?logo=mongodb&logoColor=white)](https://www.mongodb.com/)
[![AWS S3](https://img.shields.io/badge/AWS-S3-orange?logo=amazon-s3&logoColor=white)](https://aws.amazon.com/s3/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-v1.x-blue?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Docker](https://img.shields.io/badge/Docker-Enabled-blue?logo=docker&logoColor=white)](https://www.docker.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An end-to-end MLOps-style movie recommendation system that ingests movie metadata from MongoDB, validates schemas, cleans and transforms text features, trains a TF‑IDF semantic similarity model, and serves recommendations via a Flask web application and JSON APIs. Includes S3 integration for model storage and artifact caching.

</div>

---

## 📋 Table of Contents
- [Overview](#-overview)
- [Architecture & Workflow](#-architecture--workflow)
- [Recommendation Engine & Mathematics](#-recommendation-engine--mathematics)
- [Tech Stack](#-tech-stack)
- [Quick Start](#-quick-start)
- [Configuration](#-configuration)
- [Docker Deployment](#-docker-deployment)
- [Directory Structure](#-directory-structure)
- [Troubleshooting](#-troubleshooting)
- [Security & Best Practices](#-security--best-practices)
- [License & Contact](#-license--contact)

---

## 📋 Overview

This project showcases a production-ready, modular recommendation pipeline structured around two main flows:

| Stage | Component | Key Features |
| :--- | :--- | :--- |
| **Offline Pipeline** | Data Ingestion | Pulls raw movies metadata from MongoDB Atlas, saving snapshots locally. |
| | Data Validation | Performs schema validation (`config/schema.yaml`) to verify field constraints. |
| | Data Transformation | Cleans text, handles missing features, joins fields, and normalizes ratings. |
| | Recommender Training | Vectorizes clean text using TF‑IDF and generates similarity arrays. |
| | Model Push | Packages artifacts and optionally archives them to an AWS S3 Model Registry. |
| **Online Serving** | Prediction API / Web UI | Loads model artifacts dynamically (local or S3) to serve fast search and recommendations. |

---

## 🏗️ Architecture & Workflow

The end-to-end data lifecycle flows through standard pipeline stages:

```
[ MongoDB Atlas ] ──(Data Ingestion)──> [ raw_movies.csv ]
                                                │
                                        (Schema Validation)
                                                │
                                                ▼
[ transformed.csv ] <──(Feature Engineering)─── [ Schema Report ]
         │
         ├───(TF‑IDF Vectorizer)───────────────> [ tfidf_vectorizer.pkl ]
         └───(Cosine Similarity Computation)────> [ cosine_similarity.npy ]
                                                        │
                                                        ▼
[ Flask UI / JSON APIs ] <──(Model Loader)─── [ S3 Model Registry / Local Cache ]
```

---

## 🧠 Recommendation Engine & Mathematics

### 1. Feature Engineering
The recommender utilizes a **Content-Based Filtering** approach. We combine metadata elements into a single cohesive string for each movie:
```python
combined_text = overview + genres + keywords + cast + director
```

### 2. TF-IDF Representation
We convert the text corpus into a sparse matrix where each movie is represented as a numerical vector using **Term Frequency-Inverse Document Frequency (TF-IDF)**.

### 3. Cosine Similarity
To determine how similar two movies are, we calculate the angle between their TF-IDF vector representations:

$$
\text{Similarity}(\mathbf{A}, \mathbf{B}) = \cos(\theta) = \frac{\mathbf{A} \cdot \mathbf{B}}{\|\mathbf{A}\| \|\mathbf{B}\|} = \frac{\sum_{i=1}^{n} A_i B_i}{\sqrt{\sum_{i=1}^{n} A_i^2} \sqrt{\sum_{i=1}^{n} B_i^2}}
$$

This generates an $N \times N$ matrix enabling instant similarity score lookups.

### 4. Ranking Blend & Popularity Boost
During inference, recommendations are sorted using a blended scoring function that balances similarity and rating quality:

$$
\text{Score} = w \cdot \text{Similarity}(\mathbf{A}, \mathbf{B}) + (1 - w) \cdot \text{NormalizedRating}
$$

> [!NOTE]
> Rating normalization scales original movie ratings into a $[0, 1]$ range. The ranking blend ensures high-quality similar movies are prioritized over poor-quality matches.

---

## 🛠️ Tech Stack

- **ML & Mathematics**: `Python 3.10`, `scikit-learn` (TF‑IDF, Cosine Similarity), `pandas`, `NumPy`, `SciPy` (sparse matrices)
- **Serving Engine**: `Flask`, `Jinja2 Templates`, `Flask-CORS`
- **Data & Storage**: `MongoDB` (pymongo client), `AWS S3` (boto3 integration)
- **Containerization**: `Docker`

---

## 🚀 Quick Start

### 1. Clone & Set Up Environment
Clone the repository and set up a Python virtual environment:
```bash
git clone https://github.com/yourname/MLOPS-MOVIE-RECOMMENDATION-SYSTEM.git
cd MLOPS-MOVIE-RECOMMENDATION-SYSTEM
python -m venv .venv
```
Activate the environment:
* **Windows**:
  ```bash
  .venv\Scripts\activate
  ```
* **macOS/Linux**:
  ```bash
  source .venv/bin/activate
  ```

Install requirements:
```bash
pip install -r requirements.txt
```

### 2. Configure Credentials
Set the MongoDB connection URL in your environment variables:
* **Windows (Command Prompt)**:
  ```cmd
  set MONGODB_URL=mongodb+srv://<username>:<password>@cluster.mongodb.net/movie_recommender?retryWrites=true&w=majority
  ```
* **Windows (PowerShell)**:
  ```powershell
  $env:MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/movie_recommender?retryWrites=true&w=majority"
  ```
* **Linux/macOS**:
  ```bash
  export MONGODB_URL="mongodb+srv://<username>:<password>@cluster.mongodb.net/movie_recommender?retryWrites=true&w=majority"
  ```

### 3. Run Pipeline Training
Run the pipeline setup wrapper to ingest data, validate the schema, transform text, train the models, and package artifacts:
```bash
python demo.py
```

### 4. Launch Recommendation Web App
Start the serving Flask app locally:
```bash
python app.py
```
Open [http://localhost:5000](http://localhost:5000) in your web browser.

---

## ⚙️ Configuration

### Schema Validation
Schema structure checks are managed via [config/schema.yaml](config/schema.yaml). This guarantees that mandatory features like `title`, `overview`, `genres`, and `cast` exist in raw datasets.

### Environment Variables

| Variable | Description | Default / Example |
| :--- | :--- | :--- |
| `MONGODB_URL` | Connection URL to MongoDB Atlas clusters | *Required for Training* |
| `MODEL_SOURCE` | Source for model artifacts (`s3` or `auto`) | `s3` |
| `FORCE_S3_DOWNLOAD` | Force re-download of model files from S3 bucket | `false` |
| `MODEL_BUCKET_NAME` | S3 bucket name for model registry | `mlops-movie-recommender` |
| `AWS_ACCESS_KEY_ID` | AWS Credentials | *Required for S3 features* |
| `AWS_SECRET_ACCESS_KEY` | AWS Credentials | *Required for S3 features* |
| `AWS_DEFAULT_REGION` | AWS Region | `us-east-1` |

---

## 🐳 Docker Deployment

### Build Image
```bash
docker build -t movie-recommender .
```

### Run Container
To run locally without S3 dependencies using local cache:
```bash
docker run -p 5000:5000 -e MODEL_SOURCE=auto movie-recommender
```

> [!TIP]
> If using S3 model loading, pass credentials to Docker run via `-e AWS_ACCESS_KEY_ID=... -e AWS_SECRET_ACCESS_KEY=...`.

---

## 📁 Directory Structure

```
MLOPS-MOVIE-RECOMMENDATION-SYSTEM/
├── app.py                      # Flask Application Server
├── demo.py                     # Training Pipeline Run Wrapper
├── Dockerfile                  # Production Deployment Config
├── requirements.txt            # Python Dependencies
├── pyproject.toml              # Project Config
├── config/                     
│   ├── model.yaml              # Hyperparameter & Model Configuration
│   └── schema.yaml             # Input Data Schema definitions
├── templates/                  
│   └── index.html              # UI Web templates
├── static/                     
│   └── style.css               # Web stylesheet
├── model_cache/                # Deployment runtime local model cache
└── src/                        
    ├── artifacts/              # Local storage for pipeline stages outputs
    ├── components/             # Reusable pipeline steps (ingestion, trainer...)
    ├── configuration/          # AWS & MongoDB connectors
    ├── constants/              # Global workflow configuration values
    ├── data_access/            # MongoDB interface handlers
    ├── entity/                 # Input/Output data schemas & configs
    └── pipline/                # Core Training & Prediction pipelines
```

---

## 🧪 Model Evaluation

The evaluation stage (`RecommenderEvaluation`) calculates performance metrics:
- **Precision@K / Recall@K / F1-Score@K** based on content match rankings against predefined test distributions.
- Results are recorded under `best_metrics.json`.

---

## 🧯 Troubleshooting

1. **MongoDB Connection Fails**:
   - Ensure the IP address of your host machine is whitelisted in MongoDB Atlas Network Access rules.
   - Verify `MONGODB_URL` contains the correct password.

2. **Model Download Issues on Start**:
   - If using `MODEL_SOURCE=s3` and S3 variables are not set or bucket is missing, set `MODEL_SOURCE=auto` to enable local fallback execution if local training was already completed.

3. **Port Conflicts**:
   - The app defaults to port `5000`. If this is occupied (e.g., by AirPlay Receiver on macOS), you can override the port in `app.py`.

---

## 🔐 Security & Best Practices
* **Secret Management**: Never commit credentials like MongoDB URLs or AWS Access Keys to Git. Use environment variables.
* **Networking**: Tighten MongoDB ingress IP rules in Atlas.
* **IAM Roles**: When deploying on AWS (EC2/ECS), use IAM instance roles instead of hardcoding credentials.

---

## 📄 License & Contact

Distributed under the MIT License. See `LICENSE` for details.

For questions or feedback, please contact the maintainer or create a GitHub Issue in the repository.
