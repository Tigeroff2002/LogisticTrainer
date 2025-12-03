import sys
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Optional, Dict
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

# Храним все модели в памяти: имя_модели -> RouteTimePredictor
loaded_models: Dict[str, RouteTimePredictor] = {}
current_model_name: Optional[str] = None

async def load_all_models():
    """Загрузка всех моделей из хранилища в память"""
    global loaded_models, current_model_name
    
    try:
        models = storage.list_models()
        logger.info(f"Found {len(models)} models in storage: {models}")
        
        if not models:
            logger.info("No models found in storage")
            return
        
        logger.info(f"Loading {len(models)} models into memory...")
        
        for model_name in models:
            try:
                logger.info(f"Attempting to load model: {model_name}")
                predictor = RouteTimePredictor()
                metadata = storage.load_model_to_predictor(model_name, predictor)
                loaded_models[model_name] = predictor
                logger.info(f"Successfully loaded model: {model_name}")
                logger.info(f"Model {model_name} metadata: {metadata}")
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}", exc_info=True)
        
        # Устанавливаем последнюю модель как текущую
        if models:
            current_model_name = models[-1]
            logger.info(f"Current model set to: {current_model_name}")
            
    except Exception as e:
        logger.error(f"Error loading models: {e}", exc_info=True)

def get_current_predictor():
    """Получение текущего предиктора"""
    global loaded_models, current_model_name
    
    if current_model_name and current_model_name in loaded_models:
        return loaded_models[current_model_name]
    elif loaded_models:
        # Возвращаем первую доступную модель
        first_model_name = list(loaded_models.keys())[0]
        return loaded_models[first_model_name]
    else:
        return None

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Trainer Service")
    await db_service.connect()
    
    # Загружаем все модели в память
    await load_all_models()
    
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

async def train_model_task():
    """Фоновая задача обучения модели"""
    global is_training, loaded_models, current_model_name
    
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
        model_name = storage.rotate_models(predictor, metadata)
        current_model_name = model_name
        
        # 5. Загружаем новую модель в память
        # Сохраняем текущий предиктор в loaded_models
        loaded_models[model_name] = predictor
        
        logger.info("=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final metrics: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        logger.info(f"New model '{model_name}' saved and loaded to memory")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        is_training = False

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
    loaded_count = len(loaded_models)
    
    return {
        "models": models,
        "count": len(models),
        "loaded_count": loaded_count,
        "current_model": current_model_name,
        "loaded_models": list(loaded_models.keys())
    }

@app.delete("/models")
async def delete_models():
    """Удаление всех моделей"""
    global loaded_models, current_model_name
    
    result = storage.delete_all_models()
    
    # Очищаем модели из памяти
    loaded_models.clear()
    current_model_name = None
    
    return result

@app.post("/predict", response_model=PredictionResponse)
async def predict_route_time(request: PredictionRequest):
    """
    Предсказание времени маршрута
    
    Использует последнюю доступную модель для предсказания времени поездки.
    """
    try:
        predictor = get_current_predictor()
        
        if predictor is None or predictor.model is None:
            raise HTTPException(
                status_code=404, 
                detail="No trained models available. Please train a model first."
            )
        
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

@app.get("/model/current")
async def get_current_model_info():
    """Информация о текущей модели"""
    predictor = get_current_predictor()
    
    if not predictor:
        raise HTTPException(status_code=404, detail="No model loaded")
    
    return {
        "model_name": current_model_name,
        "feature_columns": predictor.feature_columns if predictor.feature_columns else [],
        "training_date": predictor.training_date.isoformat() if predictor.training_date else None,
        "metrics": predictor.metrics
    }

@app.get("/")
async def root():
    """Корневой endpoint с информацией о сервисе"""
    return {
        "message": "ML Trainer and Predictor Service",
        "version": "1.0.0",
        "endpoints": {
            "train": "POST /train - Train a new model",
            "predict": "POST /predict - Make a prediction (uses latest model)",
            "training_status": "GET /training-status - Check training status",
            "list_models": "GET /models - List available models",
            "delete_models": "DELETE /models - Delete all models",
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