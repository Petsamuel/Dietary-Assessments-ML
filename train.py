import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import numpy as np

# ----------------------------
# 1. Load and Preprocess the Dataset
# ----------------------------
try:
    data = pd.read_csv('diet.csv', encoding='latin1')
except UnicodeDecodeError:
    print("Unable to read the file with the specified encoding. Please check the file encoding.")
    exit()

# Handle missing values
numeric_columns = data.select_dtypes(include=['number']).columns
data[numeric_columns] = data[numeric_columns].fillna(data[numeric_columns].mean())
categorical_columns = data.select_dtypes(exclude=['number']).columns
data[categorical_columns] = data[categorical_columns].fillna(data[categorical_columns].mode().iloc[0])
data.dropna(subset=['BMI', 'VEGSRV', 'PROT', 'CARB', 'FAT', 'DFIB'], inplace=True)

# ----------------------------
# 2. Define Target Variables and Features
# ----------------------------
if 'Sugars' not in data.columns:
    if 'Total Sugars' in data.columns:
        data['Nutritional Score'] = (data['PROT'] * 0.3) + (data['CARB'] * 0.2) + (data['FAT'] * 0.1) - (data['Total Sugars'] * 0.4) + (data['DFIB'] * 0.2)
    else:
        data['Nutritional Score'] = (data['PROT'] * 0.3) + (data['CARB'] * 0.2) + (data['FAT'] * 0.1) + (data['DFIB'] * 0.2)
else:
    data['Nutritional Score'] = (data['PROT'] * 0.3) + (data['CARB'] * 0.2) + (data['FAT'] * 0.1) - (data['Sugars'] * 0.4) + (data['DFIB'] * 0.2)

nutrition_features = ['KCAL', 'PROT', 'FAT', 'CARB', 'CALC', 'PHOS', 'FE', 'VEGSRV', 'GRAINSRV', 'FRUITSRV']
X_nutrition = data[nutrition_features]
y_nutrition = data['Nutritional Score']

# ----------------------------
# 3. Train and Compare Models
# ----------------------------
models = [
    ('Linear Regression', LinearRegression()),
    ('Random Forest', RandomForestRegressor(random_state=42)),
    ('Gradient Boosting', GradientBoostingRegressor(random_state=42)),
    ('SVR', SVR()),
    ('KNN', KNeighborsRegressor()),
    ('Decision Tree', DecisionTreeRegressor(random_state=42)),
    ('AdaBoost', AdaBoostRegressor(random_state=42)),
]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X_nutrition, y_nutrition, test_size=0.2, random_state=42)

# Initialize results storage
results = []

# Train and evaluate each model
for name, model in models:
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    rmse = mean_squared_error(y_test, y_pred) ** 0.5
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    results.append({
        'Model': name,
        'RMSE': rmse,
        'MAE': mae,
        'R²': r2
    })
    
    print(f"{name} Results:")
    print(f"RMSE: {rmse}, MAE: {mae}, R²: {r2}")

# Convert results to DataFrame
results_df = pd.DataFrame(results)

# Plot R² Score Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='R²', data=results_df)
plt.title('R² Score Comparison')
plt.xticks(rotation=45)
plt.savefig('r2_comparison.png')

# Plot MAE Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='MAE', data=results_df)
plt.title('MAE Comparison')
plt.xticks(rotation=45)
plt.savefig('mae_comparison.png')

# Plot RMSE Comparison
plt.figure(figsize=(10, 6))
sns.barplot(x='Model', y='RMSE', data=results_df)
plt.title('RMSE Comparison')
plt.xticks(rotation=45)
plt.savefig('rmse_comparison.png')

# Save the best model
best_model_info = results_df.loc[results_df['R²'].idxmax()]
best_model_name = best_model_info['Model']
best_model = next(model for name, model in models if name == best_model_name)
joblib.dump(best_model, 'best_regression_model.pkl')
print(f"Best model saved: {best_model_name}")

# Plot Actual vs Predicted Values for the Best Model
y_pred_best = best_model.predict(X_test)
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred_best, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title(f'Actual vs Predicted for {best_model_name}')
plt.savefig('actual_vs_predicted_best_model.png')

# Residual Plot for the Best Model
residuals = y_test - y_pred_best
plt.figure(figsize=(10, 6))
sns.histplot(residuals, kde=True)
plt.title(f'Residual Distribution for {best_model_name}')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.savefig('residual_distribution_best_model.png')

# Feature Importance Heatmap (for models that support it)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.Series(best_model.feature_importances_, index=X_nutrition.columns).sort_values(ascending=False)
    plt.figure(figsize=(10, 6))
    sns.heatmap(feature_importance.to_frame(), annot=True, cmap='viridis')
    plt.title(f'Feature Importance Heatmap for {best_model_name}')
    plt.savefig('feature_importance_heatmap.png')
