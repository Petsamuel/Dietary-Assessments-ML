from utils.helper import get_interpretation

# Full analysis endpoint
@app.post("/full-analysis", response_model=EnhancedDietAnalysisResponse)
async def full_analysis(input_data: DietAnalysisInput):
    """Complete analysis pipeline"""
    processing_steps = []
    total_time = 0.0
    
    try:
        # Step 1: Data Preparation
        step_start = time.time()
        X_input = prepare_input_data(input_data)
        step_time = time.time() - step_start
        processing_steps.append(ProcessingStep(
            step_name="data_preparation",
            description="Processed user input data",
            parameters=input_data.dict(),
            processing_time=round(step_time, 4),
            status="success"
        ))
        total_time += step_time

        # Step 2: Prediction
        step_start = time.time()
        prediction = model.predict(X_input)[0] if model else None
        step_time = time.time() - step_start
        processing_steps.append(ProcessingStep(
            step_name="model_prediction",
            description="Generated nutritional score",
            parameters={"input_features": X_input.columns.tolist()},
            processing_time=round(step_time, 4),
            status="success"
        ))
        total_time += step_time

        # Step 3: Interpretation
        step_start = time.time()
        interpretation, recommendations = get_interpretation(prediction)
        step_time = time.time() - step_start
        processing_steps.append(ProcessingStep(
            step_name="interpretation",
            description="Generated recommendations",
            parameters={"score": prediction},
            processing_time=round(step_time, 4),
            status="success"
        ))
        total_time += step_time

        return EnhancedDietAnalysisResponse(
            nutritional_score=float(prediction),
            interpretation=interpretation,
            calculated_values=X_input.iloc[0].to_dict(),
            recommendations=recommendations,
            processing_steps=processing_steps,
            total_processing_time=round(total_time, 4)
        )

    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Full analysis error: {e}")
        raise HTTPException(500, "Analysis failed")