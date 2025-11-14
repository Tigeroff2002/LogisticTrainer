from minio import Minio
from minio.error import S3Error
import joblib
import io
import logging
from typing import List, Optional
from app.config import Config

class ModelStorage:
    def __init__(self, config: Config):
        self.config = config.minio
        self.client = Minio(
            endpoint=self.config.endpoint,
            access_key=self.config.access_key,
            secret_key=self.config.secret_key,
            secure=self.config.secure
        )
        self.logger = logging.getLogger(__name__)
        self._ensure_bucket_exists()

    def _ensure_bucket_exists(self):
        """Создание bucket если не существует"""
        try:
            if not self.client.bucket_exists(self.config.bucket_name):
                self.client.make_bucket(self.config.bucket_name)
                self.logger.info(f"Created bucket: {self.config.bucket_name}")
        except S3Error as e:
            self.logger.error(f"Error creating bucket: {e}")
            raise

    def list_models(self, prefix: str = "") -> List[str]:
        """Список всех моделей в storage"""
        try:
            objects = self.client.list_objects(
                self.config.bucket_name,
                prefix=prefix,
                recursive=True
            )
            return [obj.object_name for obj in objects]
        except S3Error as e:
            self.logger.error(f"Error listing models: {e}")
            return []

    def get_model_count(self) -> int:
        """Получение количества существующих моделей"""
        models = self.list_models("model_")
        return len(models)

    def save_model(self, model, model_name: str, metadata: dict = None):
        """Сохранение модели в Object Storage"""
        try:
            # Сериализация модели
            model_bytes = io.BytesIO()
            joblib.dump(model, model_bytes)
            model_bytes.seek(0)
            
            # Загрузка в MinIO
            self.client.put_object(
                bucket_name=self.config.bucket_name,
                object_name=model_name,
                data=model_bytes,
                length=len(model_bytes.getvalue()),
                content_type='application/octet-stream',
                metadata=metadata
            )
            self.logger.info(f"Model saved successfully: {model_name}")
            
        except S3Error as e:
            self.logger.error(f"Error saving model {model_name}: {e}")
            raise

    def rotate_models(self, new_model, model_metadata: dict):
        """Ротация моделей: сохранение новой и переименование старых"""
        try:
            # Получаем текущие модели
            all_models = self.list_models("model_")
            model_count = len(all_models)
            
            self.logger.info(f"Current model count: {model_count}")
            
            # Сохраняем новую модель как model_latest.pkl
            self.save_model(new_model, "model_latest.pkl", model_metadata)
            self.logger.info("Saved new model as model_latest.pkl")
            
            # Переименовываем старые модели
            for old_name in all_models:
                if old_name.startswith("model_") and old_name != "model_latest.pkl":
                    # Извлекаем номер если есть
                    if old_name.startswith("model_") and old_name.endswith(".pkl"):
                        name_part = old_name[6:-4]  # убираем 'model_' и '.pkl'
                        if name_part.isdigit():
                            old_num = int(name_part)
                            new_num = old_num + 1
                            new_name = f"model_{new_num}.pkl"
                        else:
                            # Если нет номера, начинаем с 1
                            new_name = "model_1.pkl"
                    else:
                        new_name = "model_1.pkl"
                    
                    # Копируем с новым именем
                    self.client.copy_object(
                        self.config.bucket_name,
                        new_name,
                        f"{self.config.bucket_name}/{old_name}"
                    )
                    self.logger.info(f"Renamed {old_name} to {new_name}")
            
            # Удаляем старый model_latest если он был
            if "model_latest.pkl" in all_models:
                self.client.remove_object(
                    self.config.bucket_name, 
                    "model_latest.pkl"
                )
            
            self.logger.info("Model rotation completed successfully")
            
        except S3Error as e:
            self.logger.error(f"Error during model rotation: {e}")
            raise