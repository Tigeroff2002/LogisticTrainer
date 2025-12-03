from pydantic import BaseModel, Field, validator

class TrainingResponse(BaseModel):
    job_id: str
    status: str
    message: str