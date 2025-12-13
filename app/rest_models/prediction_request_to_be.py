from pydantic import BaseModel, Field, validator
from typing import Optional

class PredictionRequestToBe(BaseModel):
    user_id: int = Field(..., description="ID пользователя")
    start_x: float = Field(..., description="Широта начальной точки")
    start_y: float = Field(..., description="Долгота начальной точки")
    end_x: float = Field(..., description="Широта конечной точки")
    end_y: float = Field(..., description="Долгота конечной точки")
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
    expected_duration_seconds: float = Field(..., description="Сырое время, предсказанное провайдером")
    
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