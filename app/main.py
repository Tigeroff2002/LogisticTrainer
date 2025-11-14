import sys
import logging
from fastapi import FastAPI, BackgroundTasks, HTTPException
from contextlib import asynccontextmanager
import asyncio
from datetime import datetime
from pydantic import BaseModel
from typing import Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)
logger.info("=== LOGGING CONFIGURED ===")

# ТЕПЕРЬ импортируем наши модули
from app.config import Config
from app.database import DatabaseService
from app.storage import ModelStorage
from app.models import RouteTimePredictor

logger.info("=== MODULES IMPORTED ===")

# Модели Pydantic
class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str

# Глобальные объекты
config = Config()
db_service = DatabaseService(config)
storage = ModelStorage(config)
is_training = False

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ML Trainer Service")
    await db_service.connect()
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

# Простой тестовый endpoint для проверки
@app.get("/test-debug")
async def test_debug():
    logger.info("=== TEST DEBUG ENDPOINT CALLED ===")
    print("=== PRINT FROM TEST DEBUG ===")
    return {"message": "debug test", "timestamp": datetime.now().isoformat()}

@app.get("/training-status")
async def get_training_status():
    logger.info("=== GETTING TRAINING STATUS ===")
    print("=== PRINT FROM TRAINING STATUS ===")
    return {
        "is_training": is_training,
        "timestamp": datetime.now().isoformat()
    }

# Остальные endpoints остаются без изменений...
async def train_model_task():
    """Фоновая задача обучения модели"""
    global is_training
    
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
        storage.rotate_models(predictor.model, metadata)
        
        logger.info("=== MODEL TRAINING COMPLETED SUCCESSFULLY ===")
        logger.info(f"Final metrics: R² = {metrics['r2']:.4f}, RMSE = {metrics['rmse']:.2f}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        raise
    finally:
        is_training = False

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

@app.get("/models")
async def list_models():
    """Список доступных моделей"""
    models = storage.list_models()
    return {
        "models": models,
        "count": len(models)
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "database_connected": db_service.connection is not None
    }

@app.get("/")
async def root():
    return {"message": "ML Trainer Service is running"}

if __name__ == "__main__":
    import uvicorn

    # ТОЛЬКО ОДИН способ запуска!
    config = uvicorn.Config(
        app, 
        host="0.0.0.0", 
        port=8000,
        log_config=None,
        access_log=False
    )
    server = uvicorn.Server(config)
    server.run()