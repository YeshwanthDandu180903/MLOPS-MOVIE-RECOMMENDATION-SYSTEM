from dataclasses import dataclass
from typing import Optional


# =========================================================
# Data Ingestion Artifact
# =========================================================
@dataclass
class DataIngestionArtifact:
    ingested_data_file_path: str


# # =========================================================
# # Data Validation Artifact
# # =========================================================
@dataclass
class DataValidationArtifact:
    validation_status: bool
    validation_report_file_path: str
    message: Optional[str] = None


# # =========================================================
# # Data Transformation Artifact
# # =========================================================


@dataclass
class DataTransformationArtifact:
    transformed_data_file_path: str


# # =========================================================
# # Recommender Model Artifact (CORE)
# # =========================================================
@dataclass
class RecommenderModelArtifact:
    tfidf_vectorizer_path: str
    tfidf_matrix_path: str
    cosine_similarity_path: str
    training_df_path: str


# # =========================================================
# # Recommender Evaluation Artifact (Ranking-based)
# # =========================================================
@dataclass
class RecommenderEvaluationArtifact:
    precision_at_k: float
    recall_at_k: float
    f1_at_k: float
    genre_precision_at_k: Optional[float] = None


# # =========================================================
# # Recommender Model Pusher Artifact (Optional – S3)
# # =========================================================
@dataclass
class RecommenderModelPusherArtifact:
    bucket_name: str
    s3_model_path: str
