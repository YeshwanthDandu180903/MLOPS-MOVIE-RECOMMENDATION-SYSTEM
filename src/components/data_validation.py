import os
import sys
import yaml
import pandas as pd

from src.exception import MyException
from src.logger import logging
from src.entity.artifact_entity import  DataIngestionArtifact, DataValidationArtifact
from src.entity.config_entity import DataValidationConfig


class DataValidation:
    def __init__(
        self,
        data_ingestion_artifact: DataIngestionArtifact,
        data_validation_config: DataValidationConfig
    ):
        try:
            self.data_ingestion_artifact = data_ingestion_artifact
            self.data_validation_config = data_validation_config

            with open(self.data_validation_config.schema_file_path, "r") as f:
                self.schema = yaml.safe_load(f)

        except Exception as e:
            raise MyException(e, sys)

    def validate_schema(self, df: pd.DataFrame) -> bool:
        """
        Check whether required columns exist
        Extra columns are allowed (they may be from data processing)
        """
        expected_columns = set(self.schema["columns"])
        actual_columns = set(df.columns)

        missing_cols = expected_columns - actual_columns
        extra_cols = actual_columns - expected_columns

        if missing_cols:
            logging.error(f"Missing required columns: {missing_cols}")
            return False
        
        if extra_cols:
            logging.info(f"Extra columns found (will be ignored): {extra_cols}")
        
        logging.info("✓ All required columns present")
        return True

    def validate_text_columns(self, df: pd.DataFrame) -> bool:
        """
        Report null values in text columns (NOT a failure - handled in Data Transformation)
        """
        logging.info("Checking null values in text columns...")
        null_summary = {}
        
        for col in self.schema.get("required_text_columns", []):
            if col in df.columns:
                null_count = df[col].isnull().sum()
                null_summary[col] = null_count
                if null_count > 0:
                    null_pct = (null_count / len(df)) * 100
                    logging.warning(f"  {col}: {null_count} nulls ({null_pct:.2f}%)")
        
        # Always return True - nulls are handled in Data Transformation
        logging.info("Null value check completed (nulls will be handled in Data Transformation)")
        return True

    def initiate_data_validation(self) -> DataValidationArtifact:
        try:
            logging.info("=" * 70)
            logging.info("Starting Data Validation")
            logging.info("=" * 70)

            df = pd.read_csv(
                self.data_ingestion_artifact.ingested_data_file_path
            )
            
            logging.info(f"Dataframe shape: {df.shape}")
            logging.info(f"Columns: {list(df.columns)}")

            # Validate schema (check required columns exist)
            schema_status = self.validate_schema(df)
            
            # Check text columns for nulls (but don't fail)
            text_status = self.validate_text_columns(df)

            # Validation passes if schema is valid
            # Null values will be handled in Data Transformation
            validation_status = schema_status

            report = {
                "schema_validation": schema_status,
                "text_column_validation": text_status,
                "overall_status": validation_status,
                "dataframe_shape": str(df.shape),
                "total_rows": len(df),
                "total_columns": len(df.columns),
                "note": "Null values will be handled in Data Transformation stage"
            }

            os.makedirs(
                os.path.dirname(self.data_validation_config.validation_report_path),
                exist_ok=True
            )

            with open(self.data_validation_config.validation_report_path, "w") as f:
                yaml.dump(report, f)

            logging.info(f"Validation report saved at: "
                         f"{self.data_validation_config.validation_report_path}")
            
            logging.info("=" * 70)
            if validation_status:
                logging.info("✓ Data Validation PASSED - Proceeding to Data Transformation")
            else:
                logging.error("✗ Data Validation FAILED - Missing required columns")
            logging.info("=" * 70)

            return DataValidationArtifact(
                validation_status=validation_status,
                validation_report_file_path=self.data_validation_config.validation_report_path,
                message=None if validation_status else "Data validation failed - missing columns"
            )

        except Exception as e:
            raise MyException(e, sys)