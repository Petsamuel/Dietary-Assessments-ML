import pandas as pd
import numpy as np
import joblib

def prepare_input_data(input_data):
    """
    Prepare input data for model prediction by calculating required features
    """
    # Calculate BMI
    height_m = input_data['Height'] / 100
    weight_kg = input_data['Weight']
    bmi = weight_kg / (height_m ** 2)
    
    # Calculate estimated caloric needs (Harris-Benedict equation)
    if input_data['Gender'] == 'Female':
        bmr = 447.593 + (9.247 * weight_kg) + (3.098 * input_data['Height']) - (4.330 * input_data['Age'])
    else:
        bmr = 88.362 + (13.397 * weight_kg) + (4.799 * input_data['Height']) - (5.677 * input_data['Age'])
    
    # Activity level multiplier
    activity_multipliers = {
        'Sedentary': 1.2,
        'Lightly active': 1.375,
        'Moderately active': 1.55,
        'Very active': 1.725,
        'Super active': 1.9
    }
    
    kcal = bmr * activity_multipliers[input_data['Activity Level']]
    
    # Estimate macronutrients based on dietary preference
    if input_data['Dietary Preference'] == 'Keto':
        prot = (kcal * 0.20) / 4  # 20% protein
        carb = (kcal * 0.05) / 4  # 5% carbs
        fat = (kcal * 0.75) / 9   # 75% fat
    elif input_data['Dietary Preference'] in ['Vegetarian', 'Vegan']:
        prot = (kcal * 0.15) / 4  # 15% protein
        carb = (kcal * 0.60) / 4  # 60% carbs
        fat = (kcal * 0.25) / 9   # 25% fat
    else:  # balanced
        prot = (kcal * 0.30) / 4  # 30% protein
        carb = (kcal * 0.45) / 4  # 45% carbs
        fat = (kcal * 0.25) / 9   # 25% fat
    
    # Estimate other nutritional values
    vegsrv = 6 if input_data['Dietary Preference'] in ['Vegetarian', 'Vegan'] else 3
    grainsrv = 6 if 'Gluten' not in input_data['Intolerances'] else 2
    fruitsrv = 4  # recommended daily servings
    
    # Estimate minerals based on caloric intake
    calc = kcal * 0.4  # mg per calorie
    phos = kcal * 0.3  # mg per calorie
    fe = kcal * 0.006  # mg per calorie
    
    # Create DataFrame with required features
    prepared_data = pd.DataFrame({
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
    
    return prepared_data

# Example usage
if __name__ == "__main__":
    # Sample input data
    input_data = {
        'Age': 30,
        'Gender': 'Female',
        'Height': 160,
        'Weight': 60,
        'Activity Level': 'Moderately active',
        'Sleep Duration': 7,
        'Stress Level': 'Moderate',
        'Smoking Status': 'No',
        'Alcohol Consumption': 'Occasional',
        'Diabetes': 'No',
        'Hypertension': 'No',
        'High Cholesterol': 'No',
        'Obesity': 'No',
        'Food Allergies': 'None',
        'Dietary Preference': 'Vegetarian',
        'Preferred Protein Sources': 'Plant-Based',
        'Intolerances': 'None'
    }
    
    try:
        # Load the model
        model = joblib.load('best_regression_model.pkl')
        
        # Prepare the input data
        X_input = prepare_input_data(input_data)
        
        # Make prediction
        prediction = model.predict(X_input)
        
        print("\nPrediction Results:")
        print(f"Nutritional Score: {prediction[0]:.2f}")
        
        # Provide interpretation
        if prediction[0] > 75:
            print("Interpretation: Excellent nutritional balance")
        elif prediction[0] > 50:
            print("Interpretation: Good nutritional balance")
        elif prediction[0] > 25:
            print("Interpretation: Fair nutritional balance - consider dietary adjustments")
        else:
            print("Interpretation: Poor nutritional balance - consider consulting a nutritionist")
            
        # Print the calculated features for verification
        print("\nCalculated Nutritional Values:")
        for column in X_input.columns:
            print(f"{column}: {X_input[column].values[0]:.2f}")
            
    except FileNotFoundError:
        print("Error: Model file 'best_regression_model.pkl' not found. Please ensure the model has been trained and saved.")
    except Exception as e:
        print(f"Error occurred: {str(e)}")