from fastapi import FastAPI, HTTPException
import pandas as pd
import numpy as np
from typing import List, Optional
import joblib
from typing import Union
from fastapi.middleware.cors import CORSMiddleware
from fastapi_cache import FastAPICache
from fastapi_cache.backends.inmemory import InMemoryBackend
from fastapi_cache.decorator import cache
from models.model import DietAnalysisInput, DietAnalysisResponse, StressLevel, Gender, ActivityLevel, DietaryPreference, ProteinSource


# Create FastAPI app
app = FastAPI(
    title="Diet Analysis API",
    description="API for analyzing dietary patterns and providing nutritional recommendations",
    version="0.0.1",
    docs_url="/",
    redoc_url="/redoc",
    openapi_url="/openapi.json"
)

# Load the model at startup
try:
    model = joblib.load('best_regression_model.pkl')
except FileNotFoundError:
    model = None

def prepare_input_data(input_data: DietAnalysisInput) -> pd.DataFrame:
    """Prepare input data for model prediction"""
    # Calculate BMI
    height_m = input_data.height / 100
    weight_kg = input_data.weight
    bmi = weight_kg / (height_m ** 2)
    
    # Calculate estimated caloric needs
    if input_data.gender == Gender.FEMALE:
        bmr = 447.593 + (9.247 * weight_kg) + (3.098 * input_data.height) - (4.330 * input_data.age)
    else:
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * input_data.height) - (5.677 * input_data.age)
    
    # Activity level multiplier
    activity_multipliers = {
        ActivityLevel.SEDENTARY: 1.2,
        ActivityLevel.LIGHTLY_ACTIVE: 1.375,
        ActivityLevel.MODERATELY_ACTIVE: 1.55,
        ActivityLevel.VERY_ACTIVE: 1.725,
        ActivityLevel.SUPER_ACTIVE: 1.9
    }
    
    kcal = bmr * activity_multipliers[input_data.activity_level]
    
    # Estimate macronutrients based on dietary preference
    if input_data.dietary_preference == DietaryPreference.KETO:
        prot = (kcal * 0.20) / 4
        carb = (kcal * 0.05) / 4
        fat = (kcal * 0.75) / 9
    elif input_data.dietary_preference in [DietaryPreference.VEGETARIAN, DietaryPreference.VEGAN]:
        prot = (kcal * 0.15) / 4
        carb = (kcal * 0.60) / 4
        fat = (kcal * 0.25) / 9
    else:
        prot = (kcal * 0.30) / 4
        carb = (kcal * 0.45) / 4
        fat = (kcal * 0.25) / 9
    
    # Estimate servings
    vegsrv = 6 if input_data.dietary_preference in [DietaryPreference.VEGETARIAN, DietaryPreference.VEGAN] else 3
    grainsrv = 6 if 'Gluten' not in input_data.intolerances else 2
    fruitsrv = 4
    
    # Estimate minerals
    calc = kcal * 0.4
    phos = kcal * 0.3
    fe = kcal * 0.006
    
    return pd.DataFrame({
        'KCAL': [kcal],
        'PROT': [prot],
        'FAT': [fat],
        'CARB': [carb],
        'CALC': [calc],
        'PHOS': [phos],
        'FE': [fe],
        'VEGSRV': [vegsrv],
        'GRAINSRV': [grainsrv],
        'FRUITSRV': [fruitsrv]
    })

def get_interpretation(score: float) -> tuple[str, List[str]]:
    """Get interpretation and recommendations based on nutritional score"""
    if score > 75:
        return ("Excellent nutritional balance", [
            "Maintain current dietary patterns",
            "Consider adding variety to maintain interest",
            "Monitor portion sizes to maintain balance"
        ])
    elif score > 50:
        return ("Good nutritional balance", [
            "Consider increasing protein intake",
            "Add more variety to vegetable choices",
            "Monitor processed food intake"
        ])
    elif score > 25:
        return ("Fair nutritional balance", [
            "Increase fruit and vegetable intake",
            "Consider reducing processed food consumption",
            "Add more whole grains to diet",
            "Consider consulting a nutritionist"
        ])
    else:
        return ("Poor nutritional balance", [
            "Strongly consider consulting a nutritionist",
            "Increase protein intake",
            "Add more fruits and vegetables",
            "Reduce processed food consumption",
            "Consider tracking daily food intake"
        ])

@app.get("/")
async def read_root():
    """Root endpoint"""
    return {"message": "Welcome to the Diet Analysis API!"}

@app.post("/analyze", response_model=DietAnalysisResponse)
@cache(expire=300)
async def analyze_diet(input_data: DietAnalysisInput):
    """Analyze diet and provide recommendations"""
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    try:
        # Prepare input data
        X_input = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(X_input)[0]
        
        # Get interpretation and recommendations
        interpretation, recommendations = get_interpretation(prediction)
        
        # Prepare response
        return DietAnalysisResponse(
            nutritional_score=float(prediction),
            interpretation=interpretation,
            calculated_values={
                column: float(X_input[column].values[0])
                for column in X_input.columns
            },
            recommendations=recommendations
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    

@app.get("/contact")
async def contact():
    """Contact endpoint"""
    return {"message": "Please contact us for any questions or feedback."}

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)