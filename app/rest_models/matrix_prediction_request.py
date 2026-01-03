from pydantic import BaseModel, Field, validator
from typing import List, Optional
from datetime import datetime

class MatrixElementInput(BaseModel):
    """Элемент матрицы на входе (совместим с C#)"""
    start_x: float = Field(..., description="Широта начальной точки")
    start_y: float = Field(..., description="Долгота начальной точки")
    end_x: float = Field(..., description="Широта конечной точки")
    end_y: float = Field(..., description="Долгота конечной точки")
    expected_duration_seconds: Optional[float] = Field(
        None,
        description="Сырое время для этой конкретной пары точек"
    )

class MatrixPredictionRequest(BaseModel):
    """Запрос для предсказания матрицы времени маршрутов"""
    user_id: int = Field(..., description="ID пользователя")
    month_of_year: str = Field(
        ..., 
        description="Месяц года",
        example="january"
    )
    time_of_day: str = Field(
        ...,
        description="Временной интервал дня",
        example="09:00:00:12:00:00"
    )
    day_of_week: str = Field(
        ...,
        description="День недели",
        example="monday"
    )
    matrix: List[List[MatrixElementInput]] = Field(
        ...,
        description="Матрица в виде списка списков (для совместимости с C#)",
        example=[
            [  # Строка 0
                MatrixElementInput(
                    start_x=55.7558, start_y=37.6176,
                    end_x=55.7558, end_y=37.6176,
                    expected_duration_seconds=0  # диагональ
                ),
                MatrixElementInput(
                    start_x=55.7558, start_y=37.6176,
                    end_x=59.9343, end_y=30.3351,
                    expected_duration_seconds=1800
                )
            ],
            [  # Строка 1
                MatrixElementInput(
                    start_x=59.9343, start_y=30.3351,
                    end_x=55.7558, end_y=37.6176,
                    expected_duration_seconds=1900
                ),
                MatrixElementInput(
                    start_x=59.9343, start_y=30.3351,
                    end_x=59.9343, end_y=30.3351,
                    expected_duration_seconds=0  # диагональ
                )
            ]
        ]
    )
    
    @validator('month_of_year')
    def validate_month(cls, v):
        valid_months = ['january', 'february', 'march', 'april', 'may', 'june', 
                       'july', 'august', 'september', 'october', 'november', 'december']
        if v.lower() not in valid_months:
            raise ValueError(f'month_of_year must be one of: {", ".join(valid_months)}')
        return v.lower()
    
    @validator('time_of_day')
    def validate_time_of_day(cls, v):
        valid_times = [
            '01:00:00:06:00:00', '06:00:00:09:00:00', '09:00:00:12:00:00',
            '12:00:00:14:00:00', '14:00:00:16:30:00', '16:30:00:19:00:00',
            '19:00:00:22:00:00', '22:00:00:01:00:00'
        ]
        if v not in valid_times:
            raise ValueError(f'time_of_day must be one of: {", ".join(valid_times)}')
        return v
    
    @validator('day_of_week')
    def validate_day_of_week(cls, v):
        valid_days = ['monday', 'tuesday', 'wednesday', 'thursday', 
                     'friday', 'saturday', 'sunday']
        if v.lower() not in valid_days:
            raise ValueError(f'day_of_week must be one of: {", ".join(valid_days)}')
        return v.lower()
    
    @validator('matrix')
    def validate_matrix_structure(cls, v):
        """Проверяем, что матрица квадратная"""
        n = len(v)
        for i, row in enumerate(v):
            if len(row) != n:
                raise ValueError(f'Row {i} has {len(row)} elements, but expected {n}')
        return v