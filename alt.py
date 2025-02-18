from enum import Enum
from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field, field_validator
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
import pandas as pd
import numpy as np
import time
import json
import joblib
import logging
import io
import csv
from models.model import Gender, ActivityLevel, StressLevel, DietAnalysisInput, DietaryPreference, ProteinSource, ProcessingStep,DietAnalysisResponse, EnhancedDietAnalysisResponse, PreparationResponse, PredictionResponse, InterpretationResponse

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="Diet Analysis API",
    description="API for analyzing dietary patterns with enhanced tracking",
    version="0.0.1",
    docs_url="/",
    redoc_url="/redoc"
)

# Load model
model = None
try:
    model = joblib.load('best_regression_model.pkl')
    logger.info("Model loaded successfully")
except FileNotFoundError:
    logger.error("Model file not found")


# Update the /prepare-data endpoint
@app.post("/prepare-data", response_model=PreparationResponse)
async def prepare_data(
    input_data: Optional[str] = Form(None),  # Accept JSON string as form data
    csv_file: Optional[UploadFile] = File(None)  # Accept CSV file
):
    """Endpoint for data preparation from either JSON input or CSV file"""
    start_time = time.time()
    processed_data = []
    source_type = "json"
    
    try:
        if csv_file:
            source_type = "csv"
            # Process CSV file
            contents = await csv_file.read()
            csv_text = io.StringIO(contents.decode('utf-8'))
            reader = csv.DictReader(csv_text)
            
            for row in reader:
                try:
                    # Convert CSV row to DietAnalysisInput
                    input_data = DietAnalysisInput(
                        age=int(row['age']),
                        height=float(row['height']),
                        weight=float(row['weight']),
                        gender=Gender[row['gender'].upper()],
                        activity_level=ActivityLevel[row['activity_level'].upper()],
                        sleep_duration=float(row['sleep_duration']),
                        stress_level=StressLevel[row['stress_level'].upper()],
                        smoking_status=row['smoking_status'].lower() == 'true',
                        alcohol_consumption=row['alcohol_consumption'],
                        diabetes=row['diabetes'],
                        hypertension=row['hypertension'].lower() == 'true',
                        high_cholesterol=row['high_cholesterol'].lower() == 'true',
                        obesity=row['obesity'],
                        food_allergies=json.loads(row['food_allergies']),
                        dietary_preference=DietaryPreference[row['dietary_preference'].upper()],
                        preferred_protein_sources=[ProteinSource[ps.upper()] for ps in json.loads(row['preferred_protein_sources'])],
                        intolerances=json.loads(row['intolerances'])
                    )
                    # Prepare data for this row
                    prepared = prepare_input_data(input_data)
                    processed_data.append(prepared.iloc[0].to_dict())
                except Exception as e:
                    logger.warning(f"Skipping invalid row: {e}")
                    continue
        elif input_data:
            # Process single JSON input
            try:
                # Parse JSON string into dictionary
                input_dict = json.loads(input_data)
                # Convert dictionary to DietAnalysisInput
                input_data = DietAnalysisInput(**input_dict)
                # Prepare data
                prepared = prepare_input_data(input_data)
                processed_data.append(prepared.iloc[0].to_dict())
            except json.JSONDecodeError as e:
                raise HTTPException(400, "Invalid JSON input")
            except Exception as e:
                raise HTTPException(400, f"Invalid input data: {e}")
        else:
            raise HTTPException(400, "Either JSON input or CSV file must be provided")
        
        processing_time = time.time() - start_time
        
        return {
            "data": processed_data[0] if source_type == "json" else processed_data,
            "processing_time": round(processing_time, 4),
            "status": "success",
            "source_type": source_type,
            "records_processed": len(processed_data)
        }
    
    except HTTPException as e:
        raise e
    except Exception as e:
        logger.error(f"Preparation error: {e}")
        raise HTTPException(500, "Data preparation failed")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=5000)