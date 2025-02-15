import pandas as pd
import joblib
import numpy as np

def prepare_test_data(personal_info, lifestyle_factors, medical_history, dietary_prefs):
    """
    Convert raw input data into format required by the models
    """
    # Calculate BMI
    height_m = personal_info['height_cm'] / 100  # convert cm to meters
    weight_kg = personal_info['weight_kg']
    bmi = weight_kg / (height_m ** 2)
    
    # Map activity level to numeric scale (1-5)
    activity_mapping = {
        'Sedentary': 1,
        'Lightly active': 2,
        'Moderately active': 3,
        'Very active': 4,
        'Super active': 5
    }
    
    # Calculate approximate body fat percentage using BMI
    # This is a rough estimation - in real application you'd want actual measurements
    if personal_info['gender'].lower() == 'male':
        body_fat = (1.20 * bmi) + (0.23 * personal_info['age']) - 16.2
    else:  # female
        body_fat = (1.20 * bmi) + (0.23 * personal_info['age']) - 5.4
    
    # Estimate waist circumference based on BMI (rough approximation)
    # In real application, you'd want actual measurements
    if personal_info['gender'].lower() == 'male':
        waist = 75 + (0.74 * bmi)
    else:  # female
        waist = 64 + (0.74 * bmi)
    
    # Estimate calories based on activity level and body metrics
    # Basic Harris-Benedict equation
    if personal_info['gender'].lower() == 'male':
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * personal_info['height_cm']) - (5.677 * personal_info['age'])
    else:  # female
        bmr = 447.593 + (9.247 * weight_kg) + (3.098 * personal_info['height_cm']) - (4.330 * personal_info['age'])
    
    activity_multiplier = {
        'Sedentary': 1.2,
        'Lightly active': 1.375,
        'Moderately active': 1.55,
        'Very active': 1.725,
        'Super active': 1.9
    }
    
    kcal = bmr * activity_multiplier[lifestyle_factors['activity_level']]
    
    # Estimate macronutrients based on dietary preference
    if dietary_prefs['diet_type'].lower() == 'keto':
        prot = (kcal * 0.20) / 4  # 20% of calories from protein
        carb = (kcal * 0.05) / 4  # 5% of calories from carbs
        fat = (kcal * 0.75) / 9   # 75% of calories from fat
    elif dietary_prefs['diet_type'].lower() == 'vegan':
        prot = (kcal * 0.15) / 4  # 15% of calories from protein
        carb = (kcal * 0.60) / 4  # 60% of calories from carbs
        fat = (kcal * 0.25) / 9   # 25% of calories from fat
    else:  # balanced
        prot = (kcal * 0.30) / 4  # 30% of calories from protein
        carb = (kcal * 0.45) / 4  # 45% of calories from carbs
        fat = (kcal * 0.25) / 9   # 25% of calories from fat
    
    # Estimate fiber based on carbs
    dfib = carb * 0.1  # Assuming 10% of carbs are fiber
    
    # Create test data dictionary matching model features
    test_data = {
        'Age': personal_info['age'],
        'BMI': bmi,
        '% Body fat': body_fat,
        'Waist': waist,
        'KCAL': kcal,
        'PROT': prot,
        'FAT': fat,
        'CARB': carb,
        'DFIB': dfib,
        'Moderate physical activity': activity_mapping[lifestyle_factors['activity_level']]
    }
    
    return pd.DataFrame([test_data])

def test_models(personal_info, lifestyle_factors, medical_history, dietary_prefs):
    """
    Test all three models with the provided data
    """
    # Prepare test data
    test_df = prepare_test_data(personal_info, lifestyle_factors, medical_history, dietary_prefs)
    
    # Load saved models
    try:
        obesity_model = joblib.load('best_regression_model.pkl')
        diet_model = joblib.load('best_regression_model.pkl')
        nutrition_model = joblib.load('best_regression_model.pkl')
        
        # Make predictions
        obesity_pred = obesity_model.predict(test_df)
        diet_pred = diet_model.predict(test_df)
        nutrition_score = nutrition_model.predict(test_df)
        
        # Format results
        results = {
            'Obesity Risk': 'High' if obesity_pred[0] == 1 else 'Low',
            'Dietary Classification': diet_pred[0],
            'Nutritional Score': round(nutrition_score[0], 2)
        }
        
        return results
        
    except FileNotFoundError:
        return "Error: Model files not found. Please ensure the models have been trained and saved."

# Example usage with hardcoded data
if __name__ == "__main__":
    # Sample input data
    personal_info = {
        'age': 35,
        'gender': 'Female',
        'height_cm': 165,  # 5'5" in cm
        'weight_kg': 65    # 143 lbs in kg
    }
    
    lifestyle_factors = {
        'activity_level': 'Moderately active',
        'sleep_duration': 7,
        'stress_level': 'Moderate',
        'smoking_status': 'No',
        'alcohol_consumption': 'Occasional'
    }
    
    medical_history = {
        'diabetes': 'No',
        'hypertension': 'No',
        'high_cholesterol': 'No',
        'obesity': 'No',
        'food_allergies': ['None'],
        'other_conditions': ''
    }
    
    dietary_prefs = {
        'diet_type': 'Balanced',
        'protein_sources': ['Meat', 'Fish', 'Dairy'],
        'intolerances': ['None']
    }
    
    # Test the models
    results = test_models(personal_info, lifestyle_factors, medical_history, dietary_prefs)
    print("\nModel Predictions:")
    for key, value in results.items():
        print(f"{key}: {value}")