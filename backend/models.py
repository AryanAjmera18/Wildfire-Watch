from pydantic import BaseModel
from typing import Optional

class PredictionResponse(BaseModel):
    label: str
    confidence: float
    model_used: str

class ExplainResponse(BaseModel):
    original_image: str # Base64
    gradcam_image: str # Base64
    label: str
    confidence: float

class ReportRequest(BaseModel):
    model_choice: str
    label: str
    confidence: float

class ReportResponse(BaseModel):
    report: str
