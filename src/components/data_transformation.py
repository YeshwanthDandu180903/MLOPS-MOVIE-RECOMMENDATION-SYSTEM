import os
import sys
import unicodedata
import ast
import pandas as pd

from src.entity.config_entity import DataTransformationConfig
from src.entity.artifact_entity import (
    DataTransformationArtifact,
    DataIngestionArtifact,
    DataValidationArtifact
)
from src.exception import MyException
from src.logger import logging


class DataTransformation:

    TEXT_FEATURES = ["overview", "genres", "keywords", "cast", "director"]

    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_transformation_config: DataTransformationConfig,
        data_validation_artifact: DataValidationArtifact
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_transformation_config = data_transformation_config
            self.data_validation_artifact = data_validation_artifact
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------
    # EXACT NOTEBOOK FUNCTIONS
    # ---------------------------

    @staticmethod
    def normalize_text(text):
        """Used ONLY for title (matches notebook)"""
        if not isinstance(text, str):
            return ""
        text = unicodedata.normalize('NFKD', text).encode(
            'ascii', 'ignore'
        ).decode('utf-8')
        return text.lower().strip()

    @staticmethod
    def join_list(x):
        """Notebook-equivalent list parsing"""
        if isinstance(x, str) and x.startswith("["):
            try:
                return " ".join(ast.literal_eval(x))
            except Exception:
                return x
        return x

    @staticmethod
    def read_data(file_path: str) -> pd.DataFrame:
        try:
            return pd.read_csv(file_path)
        except Exception as e:
            raise MyException(e, sys)

    # ---------------------------
    # PIPELINE STEPS
    # ---------------------------

    def drop_unused_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop ONLY columns not used in notebook logic
        (Do NOT drop title_norm, title_tokens, combined_text)
        """
        columns_to_drop = [
            "imdb_rating", "imdb_votes", "imdb_voting",
            "backdrop_url", "_id",
            "director_len", "cast_len",
            "genres_len", "keywords_len", "overview_len"
        ]

        existing_cols = [c for c in columns_to_drop if c in df.columns]
        if existing_cols:
            logging.info(f"Dropping unused columns: {existing_cols}")
            df = df.drop(columns=existing_cols)

        return df

    def handle_null_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Matches notebook cleaning"""
        logging.info("Handling null values...")

        # Drop rows with missing text features
        df = df.dropna(subset=self.TEXT_FEATURES)

        # Fill remaining nulls
        for col in self.TEXT_FEATURES:
            df[col] = df[col].fillna("")

        if "rating" in df.columns:
            df["rating"] = df["rating"].fillna(df["rating"].mean())

        if "poster_url" in df.columns:
            df["poster_url"] = df["poster_url"].fillna("")

        logging.info(f"✓ Null handling done. Shape: {df.shape}")
        return df

    def preprocess_text_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert list-like columns EXACTLY like notebook"""
        for col in ["genres", "cast", "keywords"]:
            if col in df.columns:
                df[col] = df[col].apply(self.join_list)
        return df

    def create_search_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Notebook search features"""
        if "title" in df.columns:
            df["title_norm"] = df["title"].apply(self.normalize_text)
            df["title_tokens"] = df["title_norm"].apply(lambda x: x.split())
        return df

    def create_combined_text(self, df: pd.DataFrame) -> pd.DataFrame:
        """EXACT notebook combined_text"""
        df["combined_text"] = (
            df["overview"] + " " +
            (df["genres"] + " ")+
            (df["keywords"] + " ") +
            df["cast"] + " " +
            df["director"]
        )
        return df

    def normalize_rating(self, df: pd.DataFrame) -> pd.DataFrame:
        """Notebook-style rating normalization"""
        if "rating" in df.columns:
            min_r = df["rating"].min()
            max_r = df["rating"].max()
            df["rating_norm"] = (
                (df["rating"] - min_r) / (max_r - min_r + 1e-8)
            )
        return df

    # ---------------------------
    # ENTRY POINT
    # ---------------------------

    def initiate_data_transformation(self) -> DataTransformationArtifact:
        try:
            if not self.data_validation_artifact.validation_status:
                raise Exception(self.data_validation_artifact.message)

            df = self.read_data(
                self.data_ingestion_artifact.ingested_data_file_path
            )
            logging.info(f"Loaded data: {df.shape}")

            df = self.drop_unused_columns(df)
            df = self.handle_null_values(df)
            df = self.preprocess_text_columns(df)
            df = self.create_search_features(df)
            df = self.create_combined_text(df)
            df = self.normalize_rating(df)

            transformed_path = self.data_transformation_config.transformed_data_path
            os.makedirs(os.path.dirname(transformed_path), exist_ok=True)

            df.to_csv(transformed_path, index=False)
            logging.info(f"✓ Transformed data saved at: {transformed_path}")

            return DataTransformationArtifact(
                transformed_data_file_path=transformed_path
            )

        except Exception as e:
            raise MyException(e, sys)
