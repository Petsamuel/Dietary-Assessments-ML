import pandas as pd

def prepare_input_data(input_data: DietAnalysisInput) -> pd.DataFrame:
    """Prepare input data for model prediction with updated fields"""
    try:
        # Calculate BMI
        height_m = input_data.height / 100
        weight_kg = input_data.weight
        bmi = weight_kg / (height_m ** 2)
        
        # Calculate estimated caloric needs using Harris-Benedict equation
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
        
        # Adjust for stress level
        stress_factors = {
            StressLevel.LOW: 1.0,
            StressLevel.MODERATE: 1.1,
            StressLevel.HIGH: 1.2
        }
        kcal *= stress_factors[input_data.stress_level]
        
        # Adjust for sleep duration
        if input_data.sleep_duration < 6:
            kcal *= 1.1  # Increase caloric needs for poor sleep
        elif input_data.sleep_duration > 8:
            kcal *= 0.95  # Slightly reduce for excessive sleep
        
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
        
        # Adjust for health conditions
        if input_data.diabetes == "Yes":
            carb *= 0.8  # Reduce carbs for diabetics
        if input_data.hypertension:
            fat *= 0.9  # Reduce fat for hypertension
        if input_data.high_cholesterol:
            fat *= 0.85  # Further reduce fat for high cholesterol
        
        # Estimate servings
        vegsrv = 6 if input_data.dietary_preference in [DietaryPreference.VEGETARIAN, DietaryPreference.VEGAN] else 3
        grainsrv = 6 if 'Gluten' not in input_data.intolerances else 2
        fruitsrv = 4
        
        # Estimate minerals
        calc = kcal * 0.4
        phos = kcal * 0.3
        fe = kcal * 0.006
        
        # Create DataFrame with all features
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
            'FRUITSRV': [fruitsrv],
            'BMI': [bmi],
            'SLEEP': [input_data.sleep_duration],
            'STRESS': [input_data.stress_level.value],
            'SMOKING': [int(input_data.smoking_status)],
            'ALCOHOL': [input_data.alcohol_consumption],
            'DIABETES': [input_data.diabetes],
            'HYPERTENSION': [int(input_data.hypertension)],
            'CHOLESTEROL': [int(input_data.high_cholesterol)],
            'OBESITY': [input_data.obesity]
        })
    
    except Exception as e:
        logger.error(f"Error preparing input data: {e}")
        raise HTTPException(status_code=400, detail="Invalid input data")
    
    
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