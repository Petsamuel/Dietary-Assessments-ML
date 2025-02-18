from enum import Enum
from typing import List, Optional, Dict, Any
from pydantic import BaseModel, field_validator, Field
from typing import Union

# Enums for validation
class Gender(str, Enum):
    MALE = "Male"
    FEMALE = "Female"
    OTHER = "Other"

class ActivityLevel(str, Enum):
    SEDENTARY = "Sedentary"
    LIGHTLY_ACTIVE = "Lightly active"
    MODERATELY_ACTIVE = "Moderately active"
    VERY_ACTIVE = "Very active"
    SUPER_ACTIVE = "Super active"

class StressLevel(str, Enum):
    LOW = "Low"
    MODERATE = "Moderate"
    HIGH = "High"

class DietaryPreference(str, Enum):
    VEGETARIAN = "Vegetarian"
    VEGAN = "Vegan"
    OMNIVORE = "Omnivore"
    KETO = "Keto"
    PALEO = "Paleo"
    OTHER = "Other"

class ProteinSource(str, Enum):
    MEAT = "Meat"
    FISH = "Fish"
    PLANT_BASED = "Plant-Based"
    DAIRY = "Dairy"

# Input model
class DietAnalysisInput(BaseModel):
    age: int = Field(..., ge=0, le=120)
    gender: Gender
    height: float = Field(..., ge=50, le=300)  # cm
    weight: float = Field(..., ge=20, le=300)  # kg
    activity_level: ActivityLevel
    sleep_duration: float = Field(..., ge=0, le=24)
    stress_level: StressLevel
    smoking_status: bool
    alcohol_consumption: str = Field(..., pattern="^(None|Occasional|Regular)$")
    diabetes: str = Field(..., pattern="^(No|Pre-Diabetic|Yes)$")
    hypertension: bool
    high_cholesterol: bool
    obesity: str = Field(..., pattern="^(No|Overweight|Underweight|Yes)$")
    food_allergies: List[str] = []
    dietary_preference: DietaryPreference
    preferred_protein_sources: List[ProteinSource] = []
    intolerances: List[str] = []

    @field_validator('food_allergies', 'intolerances')
    def convert_none_to_empty_list(cls, v):
        if v == ['None'] or v == ['none'] or v == None:
            return []
        return v

# Response models
class DietAnalysisResponse(BaseModel):
    nutritional_score: float
    interpretation: str
    calculated_values: dict
    recommendations: List[str]

class ProcessingStep(BaseModel):
    step_name: str
    description: str
    parameters: Dict[str, Any]

class EnhancedDietAnalysisResponse(DietAnalysisResponse):
    processing_steps: List[ProcessingStep]
    total_processing_time: float

class PreparationResponse(BaseModel):
    data: Union[Dict[str, float], List[Dict[str, float]]]
    processing_time: float
    status: str
    source_type: str
    records_processed: int

class PredictionResponse(BaseModel):
    score: float
    processing_time: float
    status: str

class InterpretationResponse(BaseModel):
    interpretation: str
    recommendations: List[str]
    processing_time: float
    status: str