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
        logger.info("=== CONFIG INITIALIZATION ===")
        
        # Добавьте флаг для пропуска внешних подключений
        self.skip_external_connections = True  # Временно установите True
        
        # ML Model сервис
        self.ml_service = {
            "host": "localhost",
            "port": 9000,
            "timeout": 10,
            "retries": 0 if self.skip_external_connections else 3  # 0 ретраев если пропускаем
        }
        
        logger.info(f"ML Service config: {self.ml_service}")
        
        # ИНИЦИАЛИЗАЦИЯ АТРИБУТОВ КЛАССА (используйте self.)
        self.database = DatabaseConfig()
        self.minio = MinIOConfig()    # <-- ВАЖНО: атрибут minio (не minio_config)
        self.model = ModelConfig()
        
        logger.info(f"MinIO endpoint: {self.minio.endpoint}")
        logger.info(f"Database host: {self.database.host}")
        logger.info(f"Model type: {self.model.model_type}")

# Создайте глобальный экземпляр
config = Config()