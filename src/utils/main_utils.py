import os
import sys

import numpy as np
import dill
import yaml
from pandas import DataFrame

from src.exception import MyException
from src.logger import logging


def read_yaml_file(file_path: str) -> dict:
    """Load YAML configuration files for model training and validation."""
    try:
        with open(file_path, "rb") as yaml_file:
            return yaml.safe_load(yaml_file)
    except Exception as e:
        raise MyException(e, sys) from e


def write_yaml_file(file_path: str, content: object, replace: bool = False) -> None:
    """Save YAML configuration files."""
    try:
        if replace:
            if os.path.exists(file_path):
                os.remove(file_path)
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w") as file:
            yaml.dump(content, file)
    except Exception as e:
        raise MyException(e, sys) from e


def load_object(file_path: str) -> object:
    """Load trained models, preprocessors, and other serialized objects."""
    try:
        with open(file_path, "rb") as file_obj:
            obj = dill.load(file_obj)
        return obj
    except Exception as e:
        raise MyException(e, sys) from e


def save_object(file_path: str, obj: object) -> None:
    """Save trained models, preprocessors, and other objects for MLOPS pipeline."""
    logging.info("Saving object to: %s", file_path)
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "wb") as file_obj:
            dill.dump(obj, file_obj)
        logging.info("Object saved successfully")
    except Exception as e:
        raise MyException(e, sys) from e


def save_numpy_array_data(file_path: str, array: np.array) -> None:
    """Save feature matrices and embeddings as numpy arrays."""
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            np.save(file_obj, array)
        logging.info("Numpy array saved to: %s", file_path)
    except Exception as e:
        raise MyException(e, sys) from e


def load_numpy_array_data(file_path: str) -> np.array:
    """Load feature matrices and embeddings from numpy arrays."""
    try:
        with open(file_path, 'rb') as file_obj:
            return np.load(file_obj)
    except Exception as e:
        raise MyException(e, sys) from e


def save_dataframe(file_path: str, df: DataFrame, index: bool = False) -> None:
    """Save movie/user data as CSV for training and validation."""
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        df.to_csv(file_path, index=index)
        logging.info("DataFrame saved to: %s with shape %s", file_path, df.shape)
    except Exception as e:
        raise MyException(e, sys) from e


def load_dataframe(file_path: str) -> DataFrame:
    """Load movie/user data from CSV files."""
    try:
        df = pd.read_csv(file_path)
        logging.info("DataFrame loaded from: %s with shape %s", file_path, df.shape)
        return df
    except Exception as e:
        raise MyException(e, sys) from e


def drop_columns(df: DataFrame, cols: list) -> DataFrame:
    """Remove irrelevant movie/user features from dataset."""
    logging.info("Dropping columns: %s", cols)
    try:
        df = df.drop(columns=cols, axis=1)
        logging.info("Columns dropped successfully. New shape: %s", df.shape)
        return df
    except Exception as e:
        raise MyException(e, sys) from e
