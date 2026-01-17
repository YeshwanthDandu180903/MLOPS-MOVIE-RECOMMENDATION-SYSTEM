import sys
from src.logger import logging
from src.exception import MyException

# Components
from src.components.data_ingestion import DataIngestion
# from src.components.data_validation import DataValidation
# from src.components.data_transformation import DataTransformation
# from src.components.recommender_trainer import RecommenderTrainer
# from src.components.recommender_evaluation import RecommenderEvaluation
# from src.components.model_pusher import ModelPusher
# from src.pipline.prediction_pipeline import MovieRecommender
# Configs
from src.entity.config_entity import (
    DataIngestionConfig,
    # DataValidationConfig,
    # DataTransformationConfig,
    # RecommenderModelConfig,
    # ModelPusherConfig
)
from src.constants import MODEL_BUCKET_NAME, MODEL_PUSHER_S3_KEY

# Artifacts
from src.entity.artifact_entity import (
    DataIngestionArtifact,
    # DataValidationArtifact,
    # DataTransformationArtifact,
    # RecommenderModelArtifact,
    # RecommenderModelPusherArtifact
)


class TrainingPipeline:
    def __init__(self):
        try:
            logging.info("Initializing Movie Recommendation Training Pipeline")

            self.data_ingestion_config = DataIngestionConfig()
            # self.data_validation_config = DataValidationConfig()
            # self.data_transformation_config = DataTransformationConfig()
            # self.recommender_model_config = RecommenderModelConfig()
            # self.model_pusher_config = ModelPusherConfig(
            #     bucket_name=MODEL_BUCKET_NAME,
            #     s3_model_dir=MODEL_PUSHER_S3_KEY,
            #     local_artifact_dir=self.recommender_model_config.model_dir
            # )

        except Exception as e:
            raise MyException(e, sys)

    # =========================================================
    # Data Ingestion
    # =========================================================
    def start_data_ingestion(self) -> DataIngestionArtifact:
        try:
            logging.info("Starting Data Ingestion stage")

            data_ingestion = DataIngestion(
                data_ingestion_config=self.data_ingestion_config
            )
            data_ingestion_artifact = data_ingestion.initiate_data_ingestion()

            logging.info(
                f"Data Ingestion completed. Data stored at: "
                f"{data_ingestion_artifact.ingested_data_file_path}"
            )

            return data_ingestion_artifact

        except Exception as e:
            raise MyException(e, sys)

    # # =========================================================
    # # Data Validation
    # # =========================================================
    # def start_data_validation(
    #     self,
    #     data_ingestion_artifact: DataIngestionArtifact
    # ) -> DataValidationArtifact:
    #     try:
    #         logging.info("Starting Data Validation stage")

    #         data_validation = DataValidation(
    #             data_ingestion_artifact=data_ingestion_artifact,
    #             data_validation_config=self.data_validation_config
    #         )

    #         data_validation_artifact = data_validation.initiate_data_validation()

    #         if not data_validation_artifact.validation_status:
    #             raise Exception("Data validation failed")

    #         logging.info("Data Validation completed successfully")

    #         return data_validation_artifact

    #     except Exception as e:
    #         raise MyException(e, sys)

    # # =========================================================
    # # Data Transformation
    # # =========================================================
    # def start_data_transformation(
    #     self,
    #     data_ingestion_artifact: DataIngestionArtifact,
    #     data_validation_artifact: DataValidationArtifact
    # ) -> DataTransformationArtifact:
    #     try:
    #         logging.info("Starting Data Transformation stage")

    #         data_transformation = DataTransformation(
    #             data_ingestion_artifact=data_ingestion_artifact,
    #             data_validation_artifact=data_validation_artifact,
    #             data_transformation_config=self.data_transformation_config
    #         )

    #         data_transformation_artifact = (
    #             data_transformation.initiate_data_transformation()
    #         )

    #         logging.info(
    #             f"Data Transformation completed. Transformed data at: "
    #             f"{data_transformation_artifact.transformed_data_file_path}"
    #         )

    #         return data_transformation_artifact

    #     except Exception as e:
    #         raise MyException(e, sys)

    # # =========================================================
    # # Recommender Trainer
    # # =========================================================
    # def start_recommender_trainer(
    #     self,
    #     data_transformation_artifact: DataTransformationArtifact
    # ) -> RecommenderModelArtifact:
    #     try:
    #         logging.info("Starting Recommender Trainer stage")

    #         recommender_trainer = RecommenderTrainer(
    #             data_transformation_artifact=data_transformation_artifact,
    #             recommender_model_config=self.recommender_model_config
    #         )

    #         recommender_model_artifact = (
    #             recommender_trainer.initiate_recommender_trainer()
    #         )

    #         logging.info(
    #             "Recommender Trainer completed. Artifacts saved at:"
    #             f"\nTF-IDF: {recommender_model_artifact.tfidf_vectorizer_path}"
    #             f"\nMatrix: {recommender_model_artifact.tfidf_matrix_path}"
    #             f"\nCosine: {recommender_model_artifact.cosine_similarity_path}"
    #         )

    #         return recommender_model_artifact

    #     except Exception as e:
    #         raise MyException(e, sys)

    # # =========================================================
    # # Model Pusher
    # # =========================================================
    # def start_model_pusher(self) -> RecommenderModelPusherArtifact:
    #     try:
    #         logging.info("Starting Model Pusher stage")

    #         model_pusher = ModelPusher(
    #             model_pusher_config=self.model_pusher_config
    #         )

    #         model_pusher_artifact = model_pusher.initiate_model_pusher()

    #         logging.info(
    #             f"Model uploaded to s3://{model_pusher_artifact.bucket_name}/"
    #             f"{model_pusher_artifact.s3_model_path}"
    #         )

    #         return model_pusher_artifact

    #     except Exception as e:
    #         raise MyException(e, sys)

    # =========================================================
    # Run Entire Pipeline
    # =========================================================
    def run_pipeline(self) -> None:
        try:
            logging.info("===== Movie Recommendation Training Pipeline STARTED =====")

            data_ingestion_artifact = self.start_data_ingestion()

            # data_validation_artifact = self.start_data_validation(
            #     data_ingestion_artifact
            # )

            # data_transformation_artifact = self.start_data_transformation(
            #     data_ingestion_artifact,
            #     data_validation_artifact
            # )

            # recommender_trainer_artifact = self.start_recommender_trainer(
            #     data_transformation_artifact
            # )
            # # Load recommender for evaluation
            # recommender = MovieRecommender()

            # evaluator = RecommenderEvaluation(
            #     df=recommender.df,
            #     cosine_sim=recommender.cosine_sim,
            #     recommend_fn=recommender.recommend
            # )

            # precision, recall, f1 = evaluator.precision_recall_f1_at_k(k=10)
            # genre_precision = evaluator.genre_precision_at_k(k=10)

            # # Upload trained artifacts to S3
            # self.start_model_pusher()

            # logging.info("===== Movie Recommendation Training Pipeline COMPLETED =====")
            # logging.info(
            #     f"Final Evaluation -> "
            #     f"Precision@10={precision:.4f}, "
            #     f"Recall@10={recall:.4f}, "
            #     f"F1@10={f1:.4f}, "
            #     f"GenrePrecision@10={genre_precision:.4f}"
            # )

        except Exception as e:
            raise MyException(e, sys)