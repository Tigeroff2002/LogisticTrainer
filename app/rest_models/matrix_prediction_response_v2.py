from pydantic import BaseModel, Field, validator
from typing import List

class MatrixPredictionResponseV2(BaseModel):
    predicted_durations: List[List[List[float]]] = Field(
        ...,
        description="Матрица предсказанных списков времен в секундах (N x N)",
        example=[
            [[0.0, 0.0, 0.0], [1825.5, 1825.5, 1825.5], [1943.2, 1943.2, 1943.2]],
            [[1789.1, 1789.1, 1789.1], [0.0, 0.0, 0.0], [1567.8, 1567.8, 1567.8]],
            [[2012.3, 2012.3, 2012.3], [1432.1, 1432.1, 1432.1], [0.0, 0.0, 0.0]]
        ]
    )
    model_used: str = Field(..., description="Название использованной модели")
    prediction_timestamp: str = Field(..., description="Время предсказания в ISO формате")
    matrix_size: int = Field(..., description="Размер матрицы (N x N)")