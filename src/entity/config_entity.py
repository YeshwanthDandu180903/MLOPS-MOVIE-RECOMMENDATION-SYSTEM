import os
from dataclasses import dataclass
from datetime import datetime

from src.constants import (
    ARTIFACT_DIR,
    COLLECTION_NAME,
    DATA_INGESTION_DIR,
    DATA_TRANSFORMATION_DIR,
    DATA_VALIDATION_DIR,
    PIPELINE_NAME,
    SCHEMA_FILE_PATH,
)
TIMESTAMP = datetime.now().strftime("%m_%d_%Y_%H_%M_%S")


# =========================================================
# Training Pipeline Config
# =========================================================
@dataclass
class TrainingPipelineConfig:
    pipeline_name: str = PIPELINE_NAME
    artifact_dir: str = os.path.join(ARTIFACT_DIR, TIMESTAMP)
    timestamp: str = TIMESTAMP


training_pipeline_config = TrainingPipelineConfig()


# =========================================================
# Data Ingestion Config
# =========================================================
@dataclass
class DataIngestionConfig:
    data_ingestion_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_INGESTION_DIR
    )
    ingested_data_path: str = os.path.join(
        data_ingestion_dir, "movies.csv"
    )
    collection_name: str = COLLECTION_NAME


@dataclass
class DataValidationConfig:
    data_validation_dir: str = DATA_VALIDATION_DIR
    validation_report_path: str = os.path.join(
        DATA_VALIDATION_DIR, "validation_report.yaml"
    )
    schema_file_path: str = SCHEMA_FILE_PATH


# =========================================================
# Data Transformation Config
# =========================================================
@dataclass
class DataTransformationConfig:
    data_transformation_dir: str = os.path.join(
        training_pipeline_config.artifact_dir, DATA_TRANSFORMATION_DIR
    )
    transformed_data_path: str = os.path.join(
        data_transformation_dir, "movies_transformed.csv"
    )


# =========================================================
# Recommender Model Config
# =========================================================
@dataclass
class RecommenderModelConfig:
    model_dir: str = "src/artifacts/models"
    tfidf_vectorizer_path: str = os.path.join(model_dir, "tfidf_vectorizer.pkl")
    tfidf_matrix_path: str = os.path.join(model_dir, "tfidf_matrix.npz")
    cosine_similarity_path: str = os.path.join(model_dir, "cosine_similarity.npy")
    
@dataclass
class ModelPusherConfig:
    bucket_name: str
    s3_model_dir: str
    local_artifact_dir: str
    