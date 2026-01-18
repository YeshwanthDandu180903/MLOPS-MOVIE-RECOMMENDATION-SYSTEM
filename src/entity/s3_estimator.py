import sys
import os
import numpy as np
import pandas as pd

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.estimator import MovieRecommenderEstimator


class MovieRecommenderS3Estimator:
    """
    Handles loading recommender artifacts from S3
    and serving recommendations
    """

    def __init__(self, bucket_name: str, s3_model_dir: str):
        """
        :param bucket_name: S3 bucket name
        :param s3_model_dir: S3 folder where recommender artifacts are stored
        """
        try:
            self.bucket_name = bucket_name
            self.s3_model_dir = s3_model_dir
            self.s3 = SimpleStorageService()
            self.estimator = None
        except Exception as e:
            raise MyException(e, sys)

    # -------------------------------------------------
    def is_model_present(self) -> bool:
        """
        Check whether recommender artifacts exist in S3
        """
        try:
            return self.s3.s3_key_path_available(
                bucket_name=self.bucket_name,
                s3_key=self.s3_model_dir
            )
        except Exception as e:
            raise MyException(e, sys)

    # -------------------------------------------------
    def load_model(self) -> MovieRecommenderEstimator:
        """
        Load recommender estimator (local artifacts are used
        after S3 download handled by pipeline)
        """
        try:
            logging.info("Loading MovieRecommenderEstimator")
            return MovieRecommenderEstimator()
        except Exception as e:
            raise MyException(e, sys)

    # -------------------------------------------------
    def recommend(self, movie_name: str, top_n: int = 10):
        """
        Generate movie recommendations
        """
        try:
            if self.estimator is None:
                self.estimator = self.load_model()

            return self.estimator.recommend(movie_name, top_n)

        except Exception as e:
            raise MyException(e, sys)