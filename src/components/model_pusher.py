import sys
import os

from src.cloud_storage.aws_storage import SimpleStorageService
from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import RecommenderModelPusherArtifact
from src.entity.config_entity import ModelPusherConfig


class ModelPusher:
    """
    Uploads recommender artifacts to S3
    """

    def __init__(self, model_pusher_config: ModelPusherConfig):
        try:
            self.model_pusher_config = model_pusher_config
            self.s3 = SimpleStorageService()
        except Exception as e:
            raise MyException(e, sys)

    def initiate_model_pusher(self) -> RecommenderModelPusherArtifact:
        """
        Upload recommender artifacts to S3
        """
        try:
            logging.info("Starting Model Pusher for Movie Recommendation System")

            local_artifact_dir = self.model_pusher_config.local_artifact_dir
            bucket_name = self.model_pusher_config.bucket_name
            s3_dir = self.model_pusher_config.s3_model_dir

            uploaded_files = 0

            for root, _, files in os.walk(local_artifact_dir):
                for file in files:
                    local_path = os.path.join(root, file)
                    s3_path = os.path.join(
                        s3_dir,
                        os.path.relpath(local_path, local_artifact_dir)
                    ).replace("\\", "/")

                    logging.info(f"Uploading {local_path} -> s3://{bucket_name}/{s3_path}")

                    self.s3.upload_file(
                        from_filename=local_path,
                        to_filename=s3_path,
                        bucket_name=bucket_name,
                        remove=False
                    )

                    uploaded_files += 1

            if uploaded_files == 0:
                raise Exception(
                    f"No files found to upload in local_artifact_dir: {local_artifact_dir}. "
                    "Ensure best_model artifacts exist before pushing."
                )

            model_pusher_artifact = RecommenderModelPusherArtifact(
                bucket_name=bucket_name,
                s3_model_path=s3_dir
            )

            logging.info("Model Pusher completed successfully")
            return model_pusher_artifact

        except Exception as e:
            raise MyException(e, sys)