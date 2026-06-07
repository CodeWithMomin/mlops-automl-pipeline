from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

class TokenResponse(BaseModel):
    access_token: str
    token_type: str

class PredictionInput(BaseModel):
    age: float = Field(..., ge=18, le=100, description="Age of the borrower")
    income: float = Field(..., ge=0, description="Annual income of the borrower")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score of the borrower")
    debt_to_income: float = Field(..., ge=0, le=100, description="Debt-to-income ratio in percentage")
    employment_years: float = Field(..., ge=0, description="Years in current employment")
    prior_defaults: int = Field(..., ge=0, le=1, description="Prior default flag (0 or 1)")

    class Config:
        json_schema_extra = {
            "example": {
                "age": 35.0,
                "income": 55000.0,
                "credit_score": 710.0,
                "debt_to_income": 25.5,
                "employment_years": 4.5,
                "prior_defaults": 0
            }
        }

class PredictionRequest(BaseModel):
    samples: List[PredictionInput]

class PredictionOutput(BaseModel):
    prediction: int
    probability: float

class PredictionResponse(BaseModel):
    predictions: List[PredictionOutput]
    model_name: str
    model_version: str

class FeedbackItem(PredictionInput):
    target: int = Field(..., ge=0, le=1, description="Ground truth label (0 or 1)")

class FeedbackRequest(BaseModel):
    samples: List[FeedbackItem]

class FeedbackResponse(BaseModel):
    performance_f1: float
    performance_accuracy: float
    threshold_f1: float
    degradation_detected: bool
    drift_detected: bool
    retraining_triggered: bool
    message: str
