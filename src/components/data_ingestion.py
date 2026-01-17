import os
import sys
import pandas as pd

from src.constants import COLLECTION_NAME
from src.entity.config_entity import DataIngestionConfig
from src.entity.artifact_entity import DataIngestionArtifact
from src.exception import MyException
from src.logger import logging
from src.data_access.proj1_data import MovieData


class DataIngestion:
    def __init__(
        self,
        data_ingestion_config: DataIngestionConfig = DataIngestionConfig()
    ):
        """
        Data Ingestion for Movie Recommendation System
        """
        try:
            self.data_ingestion_config = data_ingestion_config
        except Exception as e:
            raise MyException(e, sys)

    def export_data_into_feature_store(self) -> pd.DataFrame:
        """
        Fetch movie data from MongoDB and store it as a CSV artifact
        """
        try:
            logging.info("Starting data ingestion from MongoDB")

            movie_data = MovieData()
            dataframe = movie_data.export_collection_as_dataframe(collection_name=COLLECTION_NAME)

            logging.info(f"Fetched dataframe with shape: {dataframe.shape}")

            feature_store_path = self.data_ingestion_config.ingested_data_path
            os.makedirs(os.path.dirname(feature_store_path), exist_ok=True)

            dataframe.to_csv(feature_store_path, index=False, header=True)

            logging.info(
                f"Movie data saved to feature store at: {feature_store_path}"
            )

            return dataframe

        except Exception as e:
            raise MyException(e, sys)

    def initiate_data_ingestion(self) -> DataIngestionArtifact:
        """
        Initiates data ingestion pipeline
        """
        logging.info("Entered initiate_data_ingestion method")

        try:
            self.export_data_into_feature_store()

            data_ingestion_artifact = DataIngestionArtifact(
                ingested_data_file_path=self.data_ingestion_config.ingested_data_path
            )

            logging.info(
                f"Data ingestion completed successfully: {data_ingestion_artifact}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)