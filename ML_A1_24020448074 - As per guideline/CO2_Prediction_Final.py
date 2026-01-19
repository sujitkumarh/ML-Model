"""
CO2 Emission Prediction using Linear Regression
===============================================
This script predicts CO2 emissions based on vehicle features using linear regression.
Structure based on Train_and_save.ipynb - Portable and executable on any PC.

Author: ML Assignment
Date: January 2026
"""

# ============================================================================
# BLOCK 1: PREREQUISITES & ENVIRONMENT SETUP
# ============================================================================
"""
Prerequisites:
1. Python 3.7 or higher
2. Required packages: numpy, pandas, matplotlib, scikit-learn
3. CSV data file in the same directory
4. Install packages: pip install numpy pandas matplotlib scikit-learn
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("CO2 EMISSION PREDICTION - LINEAR REGRESSION MODEL")
print("="*80)
print(f"Python Version: {sys.version}")
print(f"Working Directory: {os.getcwd()}")
print("="*80 + "\n")

# ============================================================================
# BLOCK 2: IMPORT LIBRARIES
# ============================================================================
print("[STEP 1] Import Libraries...")
print("-"*80)

try:
    import numpy as np  # Numerical computing library
    print("✓ NumPy imported successfully")
except ImportError:
    print("✗ NumPy not found. Install using: pip install numpy")
    sys.exit(1)

try:
    import pandas as pd  # Data manipulation and analysis library
    print("✓ Pandas imported successfully")
except ImportError:
    print("✗ Pandas not found. Install using: pip install pandas")
    sys.exit(1)

try:
    import matplotlib.pyplot as plt  # Plotting and visualization library
    print("✓ Matplotlib imported successfully")
except ImportError:
    print("✗ Matplotlib not found. Install using: pip install matplotlib")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split  # Split data into train and test sets
    from sklearn.linear_model import LinearRegression  # Linear regression model
    from sklearn.tree import DecisionTreeRegressor  # Decision tree model for comparison
    from sklearn.ensemble import RandomForestRegressor  # Random forest model for comparison
    from sklearn.preprocessing import PolynomialFeatures  # Polynomial features for non-linear relationships
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score  # Evaluation metrics
    print("✓ Scikit-learn imported successfully")
except ImportError:
    print("✗ Scikit-learn not found. Install using: pip install scikit-learn")
    sys.exit(1)

try:
    import pickle  # Serialize and save Python objects
    print("✓ Pickle imported successfully")
except ImportError:
    print("✗ Pickle not found (should be built-in)")
    sys.exit(1)

print("\n✓ All libraries imported successfully!\n")

# ============================================================================
# BLOCK 3: LOAD DATASET
# ============================================================================
print("[STEP 2] Load Dataset...")
print("-"*80)

# Define data path (flexible for portability)
DATA_FILE = "FuelConsumptionCo2.csv"  # CSV file name

# Check if file exists in current directory
if os.path.exists(DATA_FILE):
    data_path = DATA_FILE
    print(f"✓ Data file found: {os.path.abspath(DATA_FILE)}")
else:
    # Try absolute path
    data_path = r"C:\Users\shujare\OneDrive - Capgemini\Attachments\SIMS\SEM4\ML\ML_A1_24020448074\FuelConsumptionCo2.csv"
    if os.path.exists(data_path):
        print(f"✓ Data file found: {data_path}")
    else:
        print(f"✗ ERROR: Data file not found!")
        print(f"  Please ensure '{DATA_FILE}' is in the same directory as this script")
        sys.exit(1)

# Load data
try:
    df = pd.read_csv(data_path)  # Read CSV file into pandas DataFrame
    print(f"✓ Data loaded successfully!")
    print(f"  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
except Exception as e:
    print(f"✗ ERROR loading data: {e}")
    sys.exit(1)

# Display dataset information
print("\n📊 Dataset Preview:")
print(df.describe())  # Print statistical summary
print("\nFirst 5 rows:")
print(df.head())  # Print first 5 rows

print("\n" + "="*80)

# ============================================================================
# BLOCK 4: DATA EXPLORATION & UNDERSTANDING
# ============================================================================
print("\n[STEP 3] Data Exploration & Understanding...")
print("-"*80)

print("\n📊 Dataset Information:")
print(f"  Total records: {len(df)}")
print(f"  Total features: {len(df.columns)}")
print(f"  Column names: {list(df.columns)}")

print("\n🔍 Data Types:")
print(df.dtypes)

print("\n❓ Missing Values Check:")
missing = df.isnull().sum()
if missing.sum() == 0:
    print("  ✓ No missing values found!")
else:
    print(missing[missing > 0])

print("\n📈 Random Sample (5 records):")
print(df.sample(5))  # Verify successful load with randomly selected records

print("\n" + "="*80)

# ============================================================================
# BLOCK 5: FEATURE SELECTION & PREPROCESSING
# ============================================================================
print("\n[STEP 4] Feature Selection & Preprocessing...")
print("-"*80)

# Select relevant features for modeling
selected_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
cdf = df[selected_features]  # Extract selected columns from DataFrame

print(f"\n✓ Selected features: {selected_features}")
print(f"  Preprocessed dataset shape: {cdf.shape}")

print("\nSample of preprocessed data (9 records):")
print(cdf.sample(9))  # Display 9 random samples from preprocessed data

print("\n" + "="*80)

# ============================================================================
# BLOCK 6: DATA VISUALIZATION
# ============================================================================
print("\n[STEP 5] Data Visualization...")
print("-"*80)

# Create visualizations directory
viz_dir = "visualizations"
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
    print(f"✓ Created directory: {viz_dir}/")

# 6.1: Histogram of all features
print("\n📊 Creating histograms for all features...")
viz = cdf[['CYLINDERS', 'ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
fig = plt.figure(figsize=(12, 8))
viz.hist(bins=20, edgecolor='black')
plt.suptitle('Distribution of Features', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/feature_distributions.png")
plt.show()

# 6.2: Scatter plot - Fuel Consumption vs CO2 Emissions
print("\n📈 Creating scatter plots...")
plt.figure(figsize=(10, 6))
plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue', alpha=0.5)
plt.xlabel("Fuel Consumption Combined (L/100km)", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title('Fuel Consumption vs CO2 Emissions', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/fuel_vs_co2.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/fuel_vs_co2.png")
plt.show()

# 6.3: Scatter plot - Engine Size vs CO2 Emissions
plt.figure(figsize=(10, 6))
plt.scatter(cdf.ENGINESIZE, cdf.CO2EMISSIONS, color='green', alpha=0.5)
plt.xlabel("Engine Size (L)", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title('Engine Size vs CO2 Emissions', fontsize=14, fontweight='bold')
plt.xlim(0, 8)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/engine_vs_co2.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/engine_vs_co2.png")
plt.show()

# 6.4: Scatter plot - Cylinders vs CO2 Emissions
plt.figure(figsize=(10, 6))
plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red', alpha=0.5)
plt.xlabel("Number of Cylinders", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title('Cylinders vs CO2 Emissions', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/cylinders_vs_co2.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/cylinders_vs_co2.png")
plt.show()

print("\n✓ All visualizations completed!")
print("="*80)

# ============================================================================
# BLOCK 7: PREPARE DATA - ENGINE SIZE MODEL
# ============================================================================
print("\n[STEP 6] Prepare Data - ENGINE SIZE Feature...")
print("-"*80)

# Separate features (X) and target (y)
X_engine = cdf.ENGINESIZE.to_numpy()  # Extract engine size as feature
y = cdf.CO2EMISSIONS.to_numpy()  # Extract CO2 emissions as target variable

print(f"\n✓ Feature (X): ENGINESIZE - shape {X_engine.shape}")
print(f"✓ Target (y): CO2EMISSIONS - shape {y.shape}")

print("\n" + "="*80)

# ============================================================================
# BLOCK 8: SPLIT DATA - ENGINE SIZE MODEL
# ============================================================================
print("\n[STEP 7] Split Data into Training and Testing Sets...")
print("-"*80)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X_engine, y, test_size=0.2, random_state=42  # 80% train, 20% test with fixed random state
)

print(f"\n📊 Data Split:")
print(f"  Training set: {X_train.shape}")
print(f"  Testing set: {X_test.shape}")
print(f"  Target train: {y_train.shape}")
print(f"  Target test: {y_test.shape}")
print(f"  Split ratio: 80% training, 20% testing")

print("\n" + "="*80)

# ============================================================================
# BLOCK 9: TRAIN LINEAR REGRESSION MODEL - ENGINE SIZE
# ============================================================================
print("\n[STEP 8] Train the Linear Regression Model - ENGINE SIZE...")
print("-"*80)

# Create and train model
regressor_engine = LinearRegression()  # Create a new Linear Regression model instance
regressor_engine.fit(X_train.reshape(-1, 1), y_train)  # Fit the model on training data

print("\n✓ Model trained successfully!")
print(f"  Coefficient (slope): {regressor_engine.coef_[0]:.4f}")
print(f"  Intercept: {regressor_engine.intercept_:.4f}")
print(f"  Equation: CO2 = {regressor_engine.coef_[0]:.4f} × EngineSize + {regressor_engine.intercept_:.4f}")

# Visualize training results
plt.figure(figsize=(10, 6))
plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
plt.plot(X_train, regressor_engine.coef_[0] * X_train + regressor_engine.intercept_, 
         color='red', linewidth=2, label='Regression Line')
plt.xlabel("Engine Size (L)", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title('Linear Regression: Engine Size (Training Data)', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/model_engine_training.png', dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {viz_dir}/model_engine_training.png")
plt.show()

print("="*80)

# ============================================================================
# BLOCK 10: MAKE PREDICTIONS - ENGINE SIZE
# ============================================================================
print("\n[STEP 9] Make Predictions on Test Set - ENGINE SIZE...")
print("-"*80)

# Make predictions on test set
y_pred_engine = regressor_engine.predict(X_test.reshape(-1, 1))  # Predict target values for test features

print("\n✓ Predictions completed!")
print(f"  Sample predictions (first 5):")
for i in range(min(5, len(y_pred_engine))):
    print(f"    Actual: {y_test[i]:.2f} g/km  →  Predicted: {y_pred_engine[i]:.2f} g/km")

print("\n" + "="*80)

# ============================================================================
# BLOCK 11: EVALUATE MODEL - ENGINE SIZE
# ============================================================================
print("\n[STEP 10] Evaluate Model Accuracy - ENGINE SIZE...")
print("-"*80)

# Calculate evaluation metrics
accuracy_engine = regressor_engine.score(X_test.reshape(-1, 1), y_test)  # Calculate R² score on test set
mae_engine = mean_absolute_error(y_test, y_pred_engine)
mse_engine = mean_squared_error(y_test, y_pred_engine)
rmse_engine = np.sqrt(mse_engine)
r2_engine = r2_score(y_test, y_pred_engine)

print("\n📊 Model Performance Metrics:")
print(f"  Model Accuracy (R²):           {accuracy_engine:.4f}")  # Print the accuracy score
print(f"  R² Score:                      {r2_engine:.4f}")
print(f"  Mean Absolute Error (MAE):     {mae_engine:.2f}")
print(f"  Mean Squared Error (MSE):      {mse_engine:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_engine:.2f}")

# Interpret results
print(f"\n💡 Model Interpretation:")
print(f"  The model explains {r2_engine*100:.2f}% of the variance in CO2 emissions")
if r2_engine > 0.7:
    print(f"  ✓ Good fit!")
elif r2_engine > 0.5:
    print(f"  ⚠ Moderate fit")
else:
    print(f"  ✗ Poor fit - consider other features or models")

# Visualize test results
plt.figure(figsize=(10, 6))
plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
plt.plot(X_test, regressor_engine.coef_[0] * X_test + regressor_engine.intercept_, 
         color='red', linewidth=2, label='Prediction Line')
plt.xlabel("Engine Size (L)", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title(f'Model Evaluation: Engine Size (R²={r2_engine:.4f})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/model_engine_testing.png', dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {viz_dir}/model_engine_testing.png")
plt.show()

print("="*80)

# ============================================================================
# BLOCK 12: SAVE MODEL - ENGINE SIZE
# ============================================================================
print("\n[STEP 11] Save the Trained Model - ENGINE SIZE...")
print("-"*80)

# Save the trained model to a file
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"✓ Created directory: {model_dir}/")

model_filename_engine = f'{model_dir}/co2_engine_model.pkl'
with open(model_filename_engine, 'wb') as f:  # Open file in write-binary mode
    pickle.dump(regressor_engine, f)  # Serialize and save the model object

print(f"\n✓ Model saved successfully!")
print(f"  File: {model_filename_engine}")
print(f"  Size: {os.path.getsize(model_filename_engine)} bytes")

# Verify model can be loaded
print("\n🔍 Verifying saved model...")
with open(model_filename_engine, 'rb') as f:
    loaded_model = pickle.load(f)
print("✓ Model loaded successfully from file!")

print("="*80)

# ============================================================================
# BLOCK 13: SAMPLE PREDICTION - ENGINE SIZE
# ============================================================================
print("\n[STEP 12] Test Prediction with Sample Data - ENGINE SIZE...")
print("-"*80)

# Test prediction with sample engine sizes
print("\n🔮 Making predictions with sample data...")
sample_engine_sizes = [2.0, 3.5, 5.0, 6.5]  # Sample engine sizes in liters

for engine_size in sample_engine_sizes:
    prediction = regressor_engine.predict([[engine_size]])  # Make prediction with sample input
    print(f"  Engine Size: {engine_size}L  →  Predicted CO2: {prediction[0]:.2f} g/km")

print("\n💡 Try your own prediction:")
print("   Example: regressor_engine.predict([[4.2]]) for 4.2L engine")

print("="*80)

# ============================================================================
# BLOCK 14: PREPARE DATA - FUEL CONSUMPTION MODEL
# ============================================================================
print("\n[STEP 13] Prepare Data - FUEL CONSUMPTION Feature...")
print("-"*80)

# Select fuel consumption feature from the dataframe
X_fuel = cdf.FUELCONSUMPTION_COMB.to_numpy()  # Extract fuel consumption as feature

print(f"\n✓ Feature (X): FUELCONSUMPTION_COMB - shape {X_fuel.shape}")
print(f"✓ Target (y): CO2EMISSIONS - shape {y.shape}")

print("\n" + "="*80)

# ============================================================================
# BLOCK 15: SPLIT DATA - FUEL CONSUMPTION MODEL
# ============================================================================
print("\n[STEP 14] Split Data into Training and Testing Sets - FUEL CONSUMPTION...")
print("-"*80)

# Split data 80%/20% into training and testing sets
X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(
    X_fuel, y, test_size=0.2, random_state=42  # 80% train, 20% test with fixed random state
)

print(f"\n📊 Data Split:")
print(f"  Training set: {X_train_fuel.shape}")
print(f"  Testing set: {X_test_fuel.shape}")
print(f"  Target train: {y_train_fuel.shape}")
print(f"  Target test: {y_test_fuel.shape}")

print("\n" + "="*80)

# ============================================================================
# BLOCK 16: TRAIN LINEAR REGRESSION MODEL - FUEL CONSUMPTION
# ============================================================================
print("\n[STEP 15] Train the Linear Regression Model - FUEL CONSUMPTION...")
print("-"*80)

# Train a linear regression model using the training data
regressor_fuel = LinearRegression()  # Create a new Linear Regression model instance
regressor_fuel.fit(X_train_fuel.reshape(-1, 1), y_train_fuel)  # Fit the model on training data

print("\n✓ Model trained successfully!")
print(f"  Coefficient (slope): {regressor_fuel.coef_[0]:.4f}")
print(f"  Intercept: {regressor_fuel.intercept_:.4f}")
print(f"  Equation: CO2 = {regressor_fuel.coef_[0]:.4f} × FuelConsumption + {regressor_fuel.intercept_:.4f}")

print("="*80)

# ============================================================================
# BLOCK 17: MAKE PREDICTIONS - FUEL CONSUMPTION
# ============================================================================
print("\n[STEP 16] Make Predictions on Test Set - FUEL CONSUMPTION...")
print("-"*80)

# Use the model to make test predictions on the fuel consumption testing data
y_pred_fuel = regressor_fuel.predict(X_test_fuel.reshape(-1, 1))  # Predict target values for test features

print("\n✓ Predictions completed!")
print(f"  Sample predictions (first 5):")
for i in range(min(5, len(y_pred_fuel))):
    print(f"    Actual: {y_test_fuel[i]:.2f} g/km  →  Predicted: {y_pred_fuel[i]:.2f} g/km")

print("\n" + "="*80)

# ============================================================================
# BLOCK 18: EVALUATE MODEL - FUEL CONSUMPTION
# ============================================================================
print("\n[STEP 17] Evaluate Model Accuracy - FUEL CONSUMPTION...")
print("-"*80)

# Evaluate model accuracy
accuracy_fuel = regressor_fuel.score(X_test_fuel.reshape(-1, 1), y_test_fuel)  # Calculate R² score on test set
mae_fuel = mean_absolute_error(y_test_fuel, y_pred_fuel)
mse_fuel = mean_squared_error(y_test_fuel, y_pred_fuel)
rmse_fuel = np.sqrt(mse_fuel)
r2_fuel = r2_score(y_test_fuel, y_pred_fuel)

print("\n📊 Model Performance Metrics:")
print(f"  Model Accuracy (R²):           {accuracy_fuel:.4f}")  # Print the accuracy score
print(f"  R² Score:                      {r2_fuel:.4f}")
print(f"  Mean Absolute Error (MAE):     {mae_fuel:.2f}")
print(f"  Mean Squared Error (MSE):      {mse_fuel:.2f}")
print(f"  Root Mean Squared Error (RMSE): {rmse_fuel:.2f}")

print(f"\n💡 Model Interpretation:")
print(f"  The model explains {r2_fuel*100:.2f}% of the variance in CO2 emissions")
if r2_fuel > 0.7:
    print(f"  ✓ Good fit!")
elif r2_fuel > 0.5:
    print(f"  ⚠ Moderate fit")
else:
    print(f"  ✗ Poor fit")

# Visualize results
plt.figure(figsize=(10, 6))
plt.scatter(X_test_fuel, y_test_fuel, color='blue', alpha=0.5, label='Test Data')
plt.plot(X_test_fuel, regressor_fuel.coef_[0] * X_test_fuel + regressor_fuel.intercept_, 
         color='red', linewidth=2, label='Prediction Line')
plt.xlabel("Fuel Consumption Combined (L/100km)", fontsize=12, fontweight='bold')
plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
plt.title(f'Model Evaluation: Fuel Consumption (R²={r2_fuel:.4f})', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/model_fuel_testing.png', dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {viz_dir}/model_fuel_testing.png")
plt.show()

print("="*80)

# ============================================================================
# BLOCK 19: SAVE MODEL - FUEL CONSUMPTION
# ============================================================================
print("\n[STEP 18] Save the Trained Model - FUEL CONSUMPTION...")
print("-"*80)

# Save the trained model to a file
model_filename_fuel = f'{model_dir}/co2_fuel_model.pkl'
with open(model_filename_fuel, 'wb') as f:  # Open file in write-binary mode
    pickle.dump(regressor_fuel, f)  # Serialize and save the model object

print(f"\n✓ Model saved successfully!")
print(f"  File: {model_filename_fuel}")
print(f"  Size: {os.path.getsize(model_filename_fuel)} bytes")

print("="*80)

# ============================================================================
# BLOCK 20: SAMPLE PREDICTION - FUEL CONSUMPTION
# ============================================================================
print("\n[STEP 19] Test Prediction with Sample Data - FUEL CONSUMPTION...")
print("-"*80)

# Test prediction with sample fuel consumption values
print("\n🔮 Making predictions with sample data...")
sample_fuel_consumptions = [8.0, 10.5, 12.0, 15.5]  # Sample fuel consumption in L/100km

for fuel_consumption in sample_fuel_consumptions:
    prediction = regressor_fuel.predict([[fuel_consumption]])  # Make prediction with sample input
    print(f"  Fuel Consumption: {fuel_consumption}L/100km  →  Predicted CO2: {prediction[0]:.2f} g/km")

print("\n💡 Try your own prediction:")
print("   Example: regressor_fuel.predict([[11.5]]) for 11.5L/100km fuel consumption")

print("="*80)

# ============================================================================
# BLOCK 21: MODEL COMPARISON
# ============================================================================
print("\n[STEP 20] Model Comparison...")
print("-"*80)

comparison_data = {
    'Feature': ['Engine Size', 'Fuel Consumption'],
    'MAE': [mae_engine, mae_fuel],
    'MSE': [mse_engine, mse_fuel],
    'RMSE': [rmse_engine, rmse_fuel],
    'R² Score': [r2_engine, r2_fuel],
    'Accuracy': [accuracy_engine, accuracy_fuel]
}

comparison_df = pd.DataFrame(comparison_data)
print("\n📊 Model Performance Comparison:")
print(comparison_df.to_string(index=False))

# Determine best model
best_model = 'Engine Size' if r2_engine > r2_fuel else 'Fuel Consumption'
best_r2 = max(r2_engine, r2_fuel)

print(f"\n🏆 Best Performing Feature: {best_model} (R²={best_r2:.4f})")

print("\n" + "="*80)

# ============================================================================
# BLOCK 22: ALTERNATIVE MODELS EXPLORATION
# ============================================================================
print("\n[STEP 21] Alternative Models Exploration...")
print("-"*80)
print("\n💡 MODEL SELECTION JUSTIFICATION:")
print("   Testing multiple ML algorithms to identify the best fit for CO2 prediction")
print("   Comparing: Linear Regression, Decision Tree, Random Forest, Polynomial Regression\n")

# Prepare data for all alternative models (using Engine Size feature)
X_train_reshaped = X_train.reshape(-1, 1)
X_test_reshaped = X_test.reshape(-1, 1)

# ============================================================================
# MODEL 1: Linear Regression (Already Trained)
# ============================================================================
print("\n[Model 1] Linear Regression - Already Trained")
print("  Pros: Simple, interpretable, fast, works well for linear relationships")
print("  Cons: Cannot capture non-linear patterns")
print(f"  R² Score: {r2_engine:.4f}")
print(f"  RMSE: {rmse_engine:.2f}")

# ============================================================================
# MODEL 2: Decision Tree Regressor
# ============================================================================
print("\n[Model 2] Training Decision Tree Regressor...")
dt_model = DecisionTreeRegressor(random_state=42, max_depth=5)
dt_model.fit(X_train_reshaped, y_train)
y_pred_dt = dt_model.predict(X_test_reshaped)

# Evaluate Decision Tree
dt_mae = mean_absolute_error(y_test, y_pred_dt)
dt_mse = mean_squared_error(y_test, y_pred_dt)
dt_rmse = np.sqrt(dt_mse)
dt_r2 = r2_score(y_test, y_pred_dt)

print("  Pros: Handles non-linearity, easy to visualize, no scaling needed")
print("  Cons: Prone to overfitting, unstable with small data changes")
print(f"  R² Score: {dt_r2:.4f}")
print(f"  RMSE: {dt_rmse:.2f}")

# ============================================================================
# MODEL 3: Random Forest Regressor
# ============================================================================
print("\n[Model 3] Training Random Forest Regressor...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42, max_depth=5)
rf_model.fit(X_train_reshaped, y_train)
y_pred_rf = rf_model.predict(X_test_reshaped)

# Evaluate Random Forest
rf_mae = mean_absolute_error(y_test, y_pred_rf)
rf_mse = mean_squared_error(y_test, y_pred_rf)
rf_rmse = np.sqrt(rf_mse)
rf_r2 = r2_score(y_test, y_pred_rf)

print("  Pros: Reduces overfitting, handles non-linearity, robust")
print("  Cons: More complex, slower training, less interpretable")
print(f"  R² Score: {rf_r2:.4f}")
print(f"  RMSE: {rf_rmse:.2f}")

# ============================================================================
# MODEL 4: Polynomial Regression (Degree 2)
# ============================================================================
print("\n[Model 4] Training Polynomial Regression (Degree 2)...")
poly_features = PolynomialFeatures(degree=2)
X_train_poly = poly_features.fit_transform(X_train_reshaped)
X_test_poly = poly_features.transform(X_test_reshaped)

poly_model = LinearRegression()
poly_model.fit(X_train_poly, y_train)
y_pred_poly = poly_model.predict(X_test_poly)

# Evaluate Polynomial Regression
poly_mae = mean_absolute_error(y_test, y_pred_poly)
poly_mse = mean_squared_error(y_test, y_pred_poly)
poly_rmse = np.sqrt(poly_mse)
poly_r2 = r2_score(y_test, y_pred_poly)

print("  Pros: Captures non-linear relationships while staying interpretable")
print("  Cons: Risk of overfitting with higher degrees, needs careful tuning")
print(f"  R² Score: {poly_r2:.4f}")
print(f"  RMSE: {poly_rmse:.2f}")

# ============================================================================
# COMPREHENSIVE MODEL COMPARISON TABLE
# ============================================================================
print("\n" + "="*80)
print("📊 COMPREHENSIVE MODEL COMPARISON - ENGINE SIZE FEATURE")
print("="*80)

comparison_data_all = {
    'Model': ['Linear Regression', 'Decision Tree', 'Random Forest', 'Polynomial (Deg 2)'],
    'R² Score': [r2_engine, dt_r2, rf_r2, poly_r2],
    'MAE': [mae_engine, dt_mae, rf_mae, poly_mae],
    'MSE': [mse_engine, dt_mse, rf_mse, poly_mse],
    'RMSE': [rmse_engine, dt_rmse, rf_rmse, poly_rmse],
    'Training Speed': ['Very Fast', 'Fast', 'Medium', 'Fast'],
    'Interpretability': ['High', 'Medium', 'Low', 'Medium']
}

comparison_all_df = pd.DataFrame(comparison_data_all)
print("\n" + comparison_all_df.to_string(index=False))

# Determine best model
best_model_idx = comparison_all_df['R² Score'].idxmax()
best_model_name = comparison_all_df.loc[best_model_idx, 'Model']
best_model_r2 = comparison_all_df.loc[best_model_idx, 'R² Score']

print(f"\n🏆 Best Performing Model: {best_model_name} (R²={best_model_r2:.4f})")

# ============================================================================
# MODEL SELECTION JUSTIFICATION
# ============================================================================
print("\n" + "="*80)
print("💡 MODEL SELECTION JUSTIFICATION")
print("="*80)
print("\nWhy we chose LINEAR REGRESSION as the primary model:\n")
print("  ✓ Strong Performance: R² = {:.4f} indicates good predictive power".format(r2_engine))
print("  ✓ Simplicity: Easy to understand and explain to stakeholders")
print("  ✓ Interpretability: Clear equation showing exact relationship")
print("  ✓ Fast Training & Prediction: Suitable for real-time applications")
print("  ✓ Low Resource Requirements: Can run on any device")
print("  ✓ Well-Established: Proven method with extensive documentation")
print("  ✓ No Hyperparameter Tuning: Reduces complexity and maintenance")

if poly_r2 > r2_engine:
    improvement = ((poly_r2 - r2_engine) / r2_engine) * 100
    print(f"\n⚠ Note: Polynomial Regression shows {improvement:.1f}% improvement")
    print("   Consider using for production if non-linearity is significant")

print("\n📋 When to consider alternatives:")
print("   • Decision Tree: When you need feature importance analysis")
print("   • Random Forest: When maximum accuracy is critical (production systems)")
print("   • Polynomial: When clear non-linear patterns exist in residuals")

print("\n" + "="*80)

# ============================================================================
# BLOCK 23: SUMMARY & OUTPUTS
# ============================================================================
print("\n[STEP 22] Summary & Outputs...")
print("-"*80)

print("\n✓ Analysis Complete!")
print("\n📁 Generated Files:")
print(f"  Visualizations ({viz_dir}/):")
print("    1. feature_distributions.png")
print("    2. fuel_vs_co2.png")
print("    3. engine_vs_co2.png")
print("    4. cylinders_vs_co2.png")
print("    5. model_engine_training.png")
print("    6. model_engine_testing.png")
print("    7. model_fuel_testing.png")
print(f"  Saved Models ({model_dir}/):")
print("    8. co2_engine_model.pkl")
print("    9. co2_fuel_model.pkl")

print("\n📈 Key Findings:")
print(f"  • Dataset contains {len(df)} vehicle records")
print(f"  • Engine Size model R²: {r2_engine:.4f} (Accuracy: {accuracy_engine:.4f})")
print(f"  • Fuel Consumption model R²: {r2_fuel:.4f} (Accuracy: {accuracy_fuel:.4f})")
print(f"  • Best predictor: {best_model}")

print("\n� RECOMMENDATIONS FOR EFFICIENCY & IMPROVEMENT:")
print("   1. Feature Engineering:")
print("      • Create interaction features (Engine Size × Cylinders)")
print("      • Add polynomial features for non-linear relationships")
print("      • Engineer ratio features (Power-to-Weight, Fuel Efficiency Index)")
print("\n   2. Model Optimization:")
print("      • Hyperparameter tuning using GridSearchCV or RandomizedSearchCV")
print("      • Try ensemble methods (Gradient Boosting, XGBoost)")
print("      • Implement cross-validation for robust performance estimation")
print("\n   3. Data Augmentation:")
print("      • Collect more recent vehicle data for better generalization")
print("      • Include electric/hybrid vehicles for comprehensive coverage")
print("      • Add external factors (temperature, driving conditions)")
print("\n   4. Deployment Strategy:")
print("      • Create REST API for real-time predictions")
print("      • Implement model monitoring for performance degradation")
print("      • Set up automated retraining pipeline with new data")
print("\n   5. Performance Enhancements:")
print("      • Use feature scaling (StandardScaler) for better convergence")
print("      • Implement outlier detection and removal")
print("      • Try dimensionality reduction (PCA) for high-dimensional data")

print("\n🚀 Next Steps:")
print("  1. Load saved models:")
print("     model = pickle.load(open('models/co2_engine_model.pkl', 'rb'))")
print("  2. Make predictions with new data:")
print("     model.predict([[engine_size]])")
print("  3. Try Multiple Linear Regression with all features")
print("  4. Experiment with best alternative model from comparison")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
print("✓ All models saved and ready for deployment!")
print("="*80)
