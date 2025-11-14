import os
from dataclasses import dataclass
from typing import Optional

import logging
logger = logging.getLogger(__name__)

@dataclass
class DatabaseConfig:
    host: str = os.getenv("DB_HOST", "localhost")
    port: int = int(os.getenv("DB_PORT", "5432"))
    database: str = os.getenv("DB_NAME", "logistic_salesman_db")
    user: str = os.getenv("DB_USER", "postgres")
    password: str = os.getenv("DB_PASSWORD", "root")

@dataclass
class MinIOConfig:
    endpoint: str = os.getenv("MINIO_ENDPOINT", "localhost:9000")
    access_key: str = os.getenv("MINIO_ACCESS_KEY", "minioadmin")
    secret_key: str = os.getenv("MINIO_SECRET_KEY", "minioadmin")
    bucket_name: str = os.getenv("MINIO_BUCKET", "models")
    secure: bool = os.getenv("MINIO_SECURE", "False").lower() == "true"

@dataclass
class ModelConfig:
    model_type: str = os.getenv("MODEL_TYPE", "random_forest")
    test_size: float = float(os.getenv("TEST_SIZE", "0.2"))
    random_state: int = int(os.getenv("RANDOM_STATE", "42"))

class Config:
    def __init__(self):
        logger.info("=== CONFIG INITIALIZED ===")

    database: DatabaseConfig = DatabaseConfig()
    minio: MinIOConfig = MinIOConfig()
    model: ModelConfig = ModelConfig()