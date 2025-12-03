from minio import Minio
from minio.error import S3Error
import io
import logging
from typing import List, Optional, Dict, Any
from app.config import Config
import pickle

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
        
    def get_model(self, model_name: str) -> bytes:
        """Получение модели из MinIO"""
        try:
            response = self.client.get_object(
                self.config.bucket_name,
                model_name
            )
            model_bytes = response.read()
            response.close()
            response.release_conn()
            return model_bytes
        except Exception as e:
            self.logger.error(f"Failed to get model {model_name}: {e}")
            raise

    def load_model_data(self, model_name: str) -> Dict[str, Any]:
        """Загрузка и десериализация модели"""
        try:
            self.logger.info(f"Loading model data: {model_name}")
            model_bytes = self.get_model(model_name)
            model_data = pickle.loads(model_bytes)
            return model_data
        except Exception as e:
            self.logger.error(f"Failed to load model data {model_name}: {e}")
            raise

    def load_model_to_predictor(self, model_name: str, predictor) -> Dict[str, Any]:
        """Загрузка модели в предиктор"""
        try:
            self.logger.info(f"Loading model {model_name} to predictor")
            model_data = self.load_model_data(model_name)
            
            predictor.model = model_data['model']
            predictor.label_encoders = model_data['label_encoders']
            predictor.feature_columns = model_data['feature_columns']
            predictor.scaler = model_data.get('scaler', None)
            predictor.model_type = model_data.get('model_type', 'random_forest')
            predictor.random_state = model_data.get('random_state', 42)
            predictor.training_date = model_data.get('training_date')
            predictor.metrics = model_data.get('metadata', {}).get('metrics', {})
            
            metadata = predictor.get_model_metadata() if hasattr(predictor, 'get_model_metadata') else model_data.get('metadata', {})
            
            self.logger.info(f"Model {model_name} loaded successfully to predictor")
            
            return metadata
            
        except Exception as e:
            self.logger.error(f"Failed to load model to predictor: {e}")
            raise

    def get_model_count(self) -> int:
        """Получение количества существующих моделей"""
        models = self.list_models("model_")
        return len(models)

    def save_model(self, model, model_name: str, metadata: dict = None):
        """Сохранение модели в Object Storage"""
        try:
            # Сериализация модели
            model_bytes = io.BytesIO()
            pickle.dump(model, model_bytes)
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

    def rotate_models(self, new_predictor, model_metadata: dict) -> str:
        """Ротация моделей: сохранение новой и переименование старых"""
        try:
            # Получаем текущие модели
            all_models = self.list_models("model_")
            model_count = len(all_models)
            
            self.logger.info(f"Current model count: {model_count}")
            self.logger.info(f"All models before rotation: {all_models}")
            
            # Проверяем, что new_predictor содержит модель
            if not hasattr(new_predictor, 'model') or new_predictor.model is None:
                self.logger.error("Predictor has no model!")
                raise ValueError("Predictor has no trained model")
            
            # Если есть старая модель model_latest.pkl, переименовываем ее
            if "model_latest.pkl" in all_models:
                try:
                    # Находим следующий номер
                    existing_numbers = []
                    for model_name in all_models:
                        if model_name.startswith("model_") and model_name.endswith(".pkl"):
                            name_part = model_name[6:-4]
                            if name_part.isdigit():
                                existing_numbers.append(int(name_part))
                    
                    next_number = 1
                    if existing_numbers:
                        next_number = max(existing_numbers) + 1
                    
                    new_name = f"model_{next_number}.pkl"
                    
                    # Копируем старый model_latest.pkl с новым именем
                    # Используем правильный формат CopySource
                    from minio.commonconfig import CopySource
                    
                    copy_source = CopySource(
                        bucket_name=self.config.bucket_name,
                        object_name="model_latest.pkl"
                    )
                    
                    self.client.copy_object(
                        bucket_name=self.config.bucket_name,
                        object_name=new_name,
                        source=copy_source
                    )
                    self.logger.info(f"Archived old model as: {new_name}")
                    
                except Exception as e:
                    self.logger.error(f"Error archiving old model: {e}")
            
            # Сохраняем новую модель
            model_data = {
                'model': new_predictor.model,
                'label_encoders': new_predictor.label_encoders,
                'feature_columns': new_predictor.feature_columns,
                'scaler': new_predictor.scaler,
                'model_type': new_predictor.model_type,
                'random_state': new_predictor.random_state,
                'training_date': new_predictor.training_date,
                'metadata': model_metadata
            }
            
            self.save_model(model_data, "model_latest.pkl", model_metadata)
            self.logger.info("Saved new model as model_latest.pkl")
            
            # Проверяем результат
            import time
            time.sleep(0.5)
            
            final_models = self.list_models("model_")
            self.logger.info(f"Final models: {final_models}")
            
            return "model_latest.pkl"
            
        except Exception as e:
            self.logger.error(f"Error in rotate_models: {e}", exc_info=True)
            raise

    def delete_all_models(self, prefix: str = "") -> dict:
        """Удаление всех моделей с заданным префиксом"""
        try:
            self.logger.info(f"Deleting all models with prefix: '{prefix}'")
            
            deleted_count = 0
            errors = []
            
            # Получаем список всех объектов с заданным префиксом
            objects = self.client.list_objects(
                self.config.bucket_name,
                prefix=prefix,
                recursive=True
            )
            
            for obj in objects:
                try:
                    self.client.remove_object(self.config.bucket_name, obj.object_name)
                    deleted_count += 1
                    self.logger.info(f"Deleted: {obj.object_name}")
                except Exception as e:
                    error_msg = f"Error deleting {obj.object_name}: {e}"
                    errors.append(error_msg)
                    self.logger.error(error_msg)
            
            return {
                'status': 'success' if not errors else 'partial_success',
                'deleted_count': deleted_count,
                'errors': errors if errors else None
            }
            
        except Exception as e:
            error_msg = f"Error deleting models: {e}"
            self.logger.error(error_msg)
            return {
                'status': 'error',
                'message': error_msg,
                'deleted_count': 0
            }