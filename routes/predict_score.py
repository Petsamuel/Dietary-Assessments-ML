from utils.helper import get_interpretation

# Prediction endpoint
@app.post("/predict-score", response_model=PredictionResponse)
@cache(expire=300)
async def predict_score(data: dict):
    """Endpoint for model prediction"""
    if not model:
        raise HTTPException(500, "Model not loaded")
    
    start_time = time.time()
    try:
        # Convert input dict to DataFrame
        X_input = pd.DataFrame([data])
        
        # Validate features
        required_features = ['KCAL', 'PROT', 'FAT', 'CARB', 'CALC', 
                            'PHOS', 'FE', 'VEGSRV', 'GRAINSRV', 
                            'FRUITSRV', 'BMI', 'SLEEP', 'STRESS', 
                            'SMOKING', 'ALCOHOL', 'DIABETES', 
                            'HYPERTENSION', 'CHOLESTEROL', 'OBESITY']
        
        if not all(col in X_input.columns for col in required_features):
            raise HTTPException(400, "Missing required features for prediction")

        prediction = model.predict(X_input)[0]
        return {
            "score": float(prediction),
            "processing_time": round(time.time() - start_time, 4),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(500, "Prediction failed")

# Interpretation endpoint
@app.post("/interpret-score", response_model=InterpretationResponse)
@cache(expire=300)
async def interpret_score(score: float = Body(..., embed=True)):
    """Endpoint for score interpretation"""
    start_time = time.time()
    try:
        interpretation, recommendations = get_interpretation(score)
        return {
            "interpretation": interpretation,
            "recommendations": recommendations,
            "processing_time": round(time.time() - start_time, 4),
            "status": "success"
        }
    except Exception as e:
        logger.error(f"Interpretation error: {e}")
        raise HTTPException(500, "Interpretation failed")

