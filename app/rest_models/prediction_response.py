from pydantic import BaseModel, Field, validator

class PredictionResponse(BaseModel):
    predicted_duration_seconds: float = Field(..., description="Предсказанное время в секундах")
    model_used: str = Field(..., description="Имя использованной модели")
    prediction_timestamp: str = Field(..., description="Время предсказания")