import os
import sys
import pickle
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.entity.config_entity import RecommenderModelConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    RecommenderModelArtifact
)
from src.exception import MyException
from src.logger import logging
from src.constants import COMBINED_TEXT_COLUMN


class RecommenderTrainer:
    def __init__(
        self,
        data_transformation_artifact: DataTransformationArtifact,
        recommender_model_config: RecommenderModelConfig
    ):
        try:
            self.data_transformation_artifact = data_transformation_artifact
            self.recommender_model_config = recommender_model_config
        except Exception as e:
            raise MyException(e, sys)

    def initiate_recommender_trainer(self) -> RecommenderModelArtifact:
        """
        Train TF-IDF based recommender and save artifacts
        """
        logging.info("Entered Recommender Trainer stage")

        try:
            # Load transformed data
            df = pd.read_csv(
                self.data_transformation_artifact.transformed_data_file_path
            )
            logging.info(f"Loaded transformed data: {df.shape}")

            # 🔒 SAVE TRAINING DATAFRAME (CRITICAL)
            training_df_path = os.path.join(
                self.recommender_model_config.model_dir,
                "training_df.csv"
            )

            df.to_csv(training_df_path, index=False)
            logging.info(f"Training DataFrame saved at: {training_df_path}")

            # TF-IDF Vectorization
            tfidf = TfidfVectorizer(
                stop_words="english",
                max_features=5000
            )

            tfidf_matrix = tfidf.fit_transform(df[COMBINED_TEXT_COLUMN])
            logging.info("TF-IDF vectorization completed")

            # Cosine similarity
            cosine_sim = cosine_similarity(tfidf_matrix, tfidf_matrix)
            logging.info("Cosine similarity matrix computed")

            # Create model directory
            model_dir = self.recommender_model_config.model_dir
            os.makedirs(model_dir, exist_ok=True)

            # Save TF-IDF vectorizer
            with open(self.recommender_model_config.tfidf_vectorizer_path, "wb") as f:
                pickle.dump(tfidf, f)

            # Save TF-IDF matrix
            sparse.save_npz(
                self.recommender_model_config.tfidf_matrix_path,
                tfidf_matrix
            )

            # Save cosine similarity matrix
            np.save(
                self.recommender_model_config.cosine_similarity_path,
                cosine_sim
            )

            logging.info("Recommender artifacts saved successfully")

            return RecommenderModelArtifact(
                tfidf_vectorizer_path=self.recommender_model_config.tfidf_vectorizer_path,
                tfidf_matrix_path=self.recommender_model_config.tfidf_matrix_path,
                cosine_similarity_path=self.recommender_model_config.cosine_similarity_path,
                training_df_path=training_df_path
            )

        except Exception as e:
            raise MyException(e, sys)