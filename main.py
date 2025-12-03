import sys
import logging
import tempfile
import os
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional
import pandas as pd

from app.rest_models.prediction_request import PredictionRequest
from app.rest_models.prediction_response import PredictionResponse
from app.rest_models.training_response import TrainingResponse

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info("=== LOGGING CONFIGURED ===")

# Импорт наших модулей
from app.config import Config
from app.database import DatabaseService
from app.storage import ModelStorage
from app.models import RouteTimePredictor

logger.info("=== MODULES IMPORTED ===")

# Глобальные объекты
config = Config()
db_service = DatabaseService(config)
storage = ModelStorage(config)
is_training = False
predictor = None
current_model_name = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Trainer Service")
    await db_service.connect()
    
    # Инициализируем предсказатель
    global predictor
    predictor = RouteTimePredictor(
        model_type=config.model.model_type,
        random_state=config.model.random_state
    )
    
    # Загружаем последнюю модель при старте, если она существует
    try:
        models = storage.list_models()
        if models:
            latest_model = models[-1]
            await load_model(latest_model)
            logger.info(f"Loaded latest model on startup: {latest_model}")
        else:
            logger.info("No models available on startup")
    except Exception as e:
        logger.warning(f"Could not load model on startup: {e}")
    
    logger.info("Service started successfully")
    
    yield
    
    # Shutdown
    await db_service.disconnect()
    logger.info("Service shutdown completed")

app = FastAPI(
    title="ML Model Trainer",
    description="Service for training route time prediction models",
    lifespan=lifespan
)

async def load_model(model_name: str):
    """Загрузка модели из хранилища"""
    global predictor, current_model_name
    
    try:
        # Скачиваем модель из хранилища во временный файл
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pkl') as tmp_file:
            model_bytes = storage.get_model(model_name)
            if not model_bytes:
                raise HTTPException(status_code=404, detail=f"Model '{model_name}' not found in storage")
            
            tmp_file.write(model_bytes)
            tmp_path = tmp_file.name
        
        # Загружаем модель
        predictor.load_model(tmp_path)
        current_model_name = model_name
        
        # Удаляем временный файл
        os.unlink(tmp_path)
        
        logger.info(f"Successfully loaded model: {model_name}")
        
    except Exception as e:
        logger.error(f"Failed to load model {model_name}: {e}")
        raise

@app.post("/train", response_model=TrainingResponse)
async def train_model(background_tasks: BackgroundTasks):
    """Запуск обучения модели"""
    global is_training
    
    if is_training:
        raise HTTPException(status_code=409, detail="Training already in progress")
    
    job_id = f"train_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    background_tasks.add_task(train_model_task)
    
    return TrainingResponse(
        job_id=job_id,
        status="started",
        message="Model training started in background"
    )

@app.get("/training-status")
async def get_training_status():
    """Статус обучения модели"""
    return {
        "is_finished_training": not is_training,
        "timestamp": datetime.now().isoformat()
    }

@app.get("/models")
async def list_models():
    """Список доступных моделей"""
    models = storage.list_models()
    return {
        "models": models,
        "count": len(models),
        "current_model": current_model_name
    }

@app.delete("/models")
async def list_models():
    """Удаление всех моделей"""
    return storage.delete_all_models()

@app.post("/predict", response_model=PredictionResponse)
async def predict_route_time(request: PredictionRequest):
    """
    Предсказание времени маршрута
    
    Использует последнюю доступную модель для предсказания времени поездки.
    """
    global predictor, current_model_name
    
    try:
        # Проверяем, загружена ли модель
        if predictor is None or predictor.model is None:
            # Пытаемся загрузить последнюю модель
            models = storage.list_models()
            if not models:
                raise HTTPException(
                    status_code=404, 
                    detail="No trained models available. Please train a model first."
                )
            
            latest_model = models[-1]
            await load_model(latest_model)
        
        # Подготавливаем данные для предсказания
        input_data = {
            'user_id': request.user_id,
            'start_fav_area_id': request.start_fav_area_id,
            'end_fav_area_id': request.end_fav_area_id,
            'month_of_year': request.month_of_year,
            'time_of_day': request.time_of_day,
            'day_of_week': request.day_of_week,
            'duration_seconds': 0  # Добавляем как placeholder
        }
        
        # Делаем предсказание
        predicted_duration = predictor.predict_single(input_data)
        
        # Формируем ответ
        return PredictionResponse(
            predicted_duration_seconds=float(predicted_duration),
            model_used=current_model_name if current_model_name else 'unknown_model',
            prediction_timestamp=datetime.now().isoformat()
        )
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Prediction failed: {str(e)}"
        )

@app.get("/")
async def root():
    """Корневой endpoint с информацией о сервисе"""
    return {
        "message": "ML Trainer and Predictor Service",
        "version": "1.0.0",
        "endpoints": {
            "train": "POST /train - Train a new model",
            "predict": "POST /predict - Make a prediction (uses latest model)",
            "predict_with_model": "POST /predict/{model_name} - Make prediction with specific model",
            "training_status": "GET /training-status - Check training status",
            "list_models": "GET /models - List available models",
            "current_model": "GET /model/current - Get current model info",
            "load_model": "POST /model/load/{model_name} - Load specific model",
            "health": "GET /health - Health check"
        }
    }

if __name__ == "__main__":
    import uvicorn

    config_server = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=None,
        access_log=False
    )
    server = uvicorn.Server(config_server)
    server.run()

async def train_model_task():
    """Фоновая задача обучения модели"""
    global is_training, predictor, current_model_name
    
    try:
        is_training = True
        logger.info("=== STARTING MODEL TRAINING ===")
        
        # 1. Получение данных
        logger.info("Step 1: Fetching training data from database")
        df = await db_service.get_training_data()
        
        if df.empty:
            logger.error("No training data available")
            return
        
        # 2. Обучение модели
        logger.info("Step 2: Training the model")
        predictor = RouteTimePredictor(
            model_type=config.model.model_type,
            random_state=config.model.random_state
        )
        
        metrics = predictor.train(df, test_size=config.model.test_size)
        
        # 3. Подготовка метаданных
        logger.info("Step 3: Preparing model metadata")
        metadata = predictor.get_model_metadata(metrics)
        
        # 4. Сохранение модели с ротацией
        logger.info("Step 4: Saving model with rotation")
        model_name = storage.rotate_models(predictor.model, metadata)
        current_model_name = model_name
        
        logger.info("=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final metrics: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        is_training = False