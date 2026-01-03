from pydantic import BaseModel, Field, validator
from typing import List

class MatrixPredictionResponse(BaseModel):
    """Ответ с матрицей предсказанных времен (совместим с C#)"""
    predicted_durations: List[List[float]] = Field(
        ...,
        description="Матрица предсказанных времен в секундах (N x N)",
        example=[
            [0.0, 1825.5, 1943.2],
            [1789.1, 0.0, 1567.8],
            [2012.3, 1432.1, 0.0]
        ]
    )
    model_used: str = Field(..., description="Название использованной модели")
    prediction_timestamp: str = Field(..., description="Время предсказания в ISO формате")
    matrix_size: int = Field(..., description="Размер матрицы (N x N)")