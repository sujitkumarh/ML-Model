"""
CO2 Emission Prediction - Interactive Block Execution
====================================================
Execute individual blocks or groups of blocks interactively.
Based on Train_and_save.ipynb structure.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Global variables to store state across blocks
df = None
cdf = None
X_engine = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
regressor_engine = None
y_pred_engine = None
accuracy_engine = None
mae_engine = None
mse_engine = None
rmse_engine = None
r2_engine = None
X_fuel = None
X_train_fuel = None
X_test_fuel = None
y_train_fuel = None
y_test_fuel = None
regressor_fuel = None
y_pred_fuel = None
accuracy_fuel = None
mae_fuel = None
mse_fuel = None
rmse_fuel = None
r2_fuel = None
viz_dir = "visualizations"
model_dir = "models"

# ============================================================================
# BLOCK FUNCTIONS
# ============================================================================

def block_1_import_libraries():
    """Block 1: Import Libraries"""
    global np, pd, plt, train_test_split, LinearRegression, mean_absolute_error, mean_squared_error, r2_score, pickle
    
    print("\n" + "="*80)
    print("BLOCK 1: Import Libraries")
    print("="*80)
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        from sklearn.model_selection import train_test_split
        from sklearn.linear_model import LinearRegression
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
        import pickle
        
        print("✓ NumPy imported successfully")
        print("✓ Pandas imported successfully")
        print("✓ Matplotlib imported successfully")
        print("✓ Scikit-learn imported successfully")
        print("✓ Pickle imported successfully")
        print("\n✓ All libraries imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Error importing libraries: {e}")
        print("Run: pip install numpy pandas matplotlib scikit-learn")
        return False

def block_2_load_dataset():
    """Block 2: Load Dataset"""
    global df
    
    print("\n" + "="*80)
    print("BLOCK 2: Load Dataset")
    print("="*80)
    
    DATA_FILE = "FuelConsumptionCo2.csv"
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(script_dir, DATA_FILE)
    
    try:
        df = pd.read_csv(data_path)
        print(f"✓ Data loaded successfully!")
        print(f"  Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        print(f"  File: {os.path.abspath(data_path)}")
        return True
    except Exception as e:
        print(f"✗ ERROR loading data: {e}")
        return False

def block_3_explore_data():
    """Block 3: Data Exploration"""
    global df
    
    print("\n" + "="*80)
    print("BLOCK 3: Data Exploration & Understanding")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    print(f"\n📊 Dataset Information:")
    print(f"  Total records: {len(df)}")
    print(f"  Total features: {len(df.columns)}")
    print(f"  Columns: {list(df.columns)}")
    
    print("\n📈 Statistical Summary:")
    print(df.describe())
    
    print("\n🔍 Data Types:")
    print(df.dtypes)
    
    print("\n❓ Missing Values:")
    missing = df.isnull().sum()
    if missing.sum() == 0:
        print("  ✓ No missing values!")
    else:
        print(missing[missing > 0])
    
    print("\n📋 Sample Data (5 random records):")
    print(df.sample(5))
    
    return True

def block_4_feature_selection():
    """Block 4: Feature Selection"""
    global df, cdf
    
    print("\n" + "="*80)
    print("BLOCK 4: Feature Selection & Preprocessing")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    selected_features = ['ENGINESIZE', 'CYLINDERS', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']
    cdf = df[selected_features]
    
    print(f"✓ Selected features: {selected_features}")
    print(f"  Shape: {cdf.shape}")
    print("\nSample (9 records):")
    print(cdf.sample(9))
    
    return True

def block_5_visualize_data():
    """Block 5: Data Visualization"""
    global cdf, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 5: Data Visualization")
    print("="*80)
    
    if cdf is None:
        print("✗ Error: Features not selected. Run Block 4 first.")
        return False
    
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"✓ Created directory: {viz_dir}/")
    
    # Histograms
    print("\n📊 Creating histograms...")
    viz = cdf[['CYLINDERS', 'ENGINESIZE', 'FUELCONSUMPTION_COMB', 'CO2EMISSIONS']]
    fig = plt.figure(figsize=(12, 8))
    viz.hist(bins=20, edgecolor='black')
    plt.suptitle('Distribution of Features', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/feature_distributions.png")
    plt.show()
    
    # Scatter plots
    print("\n📈 Creating scatter plots...")
    
    # Fuel vs CO2
    plt.figure(figsize=(10, 6))
    plt.scatter(cdf.FUELCONSUMPTION_COMB, cdf.CO2EMISSIONS, color='blue', alpha=0.5)
    plt.xlabel("Fuel Consumption (L/100km)", fontsize=12, fontweight='bold')
    plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
    plt.title('Fuel Consumption vs CO2 Emissions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/fuel_vs_co2.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/fuel_vs_co2.png")
    plt.show()
    
    # Engine vs CO2
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
    
    # Cylinders vs CO2
    plt.figure(figsize=(10, 6))
    plt.scatter(cdf.CYLINDERS, cdf.CO2EMISSIONS, color='red', alpha=0.5)
    plt.xlabel("Cylinders", fontsize=12, fontweight='bold')
    plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
    plt.title('Cylinders vs CO2 Emissions', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/cylinders_vs_co2.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/cylinders_vs_co2.png")
    plt.show()
    
    print("\n✓ All visualizations completed!")
    return True

def block_6_train_engine_model():
    """Block 6: Train Engine Size Model"""
    global cdf, X_engine, y, X_train, X_test, y_train, y_test, regressor_engine, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 6: Train Linear Regression Model - ENGINE SIZE")
    print("="*80)
    
    if cdf is None:
        print("✗ Error: Features not selected. Run Block 4 first.")
        return False
    
    # Prepare data
    X_engine = cdf.ENGINESIZE.to_numpy()
    y = cdf.CO2EMISSIONS.to_numpy()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_engine, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data Split: {X_train.shape}, {X_test.shape}, {y_train.shape}, {y_test.shape}")
    
    # Train model
    regressor_engine = LinearRegression()
    regressor_engine.fit(X_train.reshape(-1, 1), y_train)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  Coefficient: {regressor_engine.coef_[0]:.4f}")
    print(f"  Intercept: {regressor_engine.intercept_:.4f}")
    print(f"  Equation: CO2 = {regressor_engine.coef_[0]:.4f} × EngineSize + {regressor_engine.intercept_:.4f}")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_train, y_train, color='blue', alpha=0.5, label='Training Data')
    plt.plot(X_train, regressor_engine.coef_[0] * X_train + regressor_engine.intercept_, 
             color='red', linewidth=2, label='Regression Line')
    plt.xlabel("Engine Size (L)", fontsize=12, fontweight='bold')
    plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
    plt.title('Training: Engine Size Model', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/model_engine_training.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/model_engine_training.png")
    plt.show()
    
    return True

def block_7_evaluate_engine_model():
    """Block 7: Evaluate Engine Model"""
    global regressor_engine, X_test, y_test, y_pred_engine, accuracy_engine, mae_engine, mse_engine, rmse_engine, r2_engine, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 7: Evaluate Model - ENGINE SIZE")
    print("="*80)
    
    if regressor_engine is None:
        print("✗ Error: Model not trained. Run Block 6 first.")
        return False
    
    # Predict
    y_pred_engine = regressor_engine.predict(X_test.reshape(-1, 1))
    
    # Calculate metrics
    accuracy_engine = regressor_engine.score(X_test.reshape(-1, 1), y_test)
    mae_engine = mean_absolute_error(y_test, y_pred_engine)
    mse_engine = mean_squared_error(y_test, y_pred_engine)
    rmse_engine = np.sqrt(mse_engine)
    r2_engine = r2_score(y_test, y_pred_engine)
    
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy (R²): {accuracy_engine:.4f}")
    print(f"  MAE: {mae_engine:.2f}")
    print(f"  MSE: {mse_engine:.2f}")
    print(f"  RMSE: {rmse_engine:.2f}")
    print(f"  R² Score: {r2_engine:.4f}")
    
    print(f"\n💡 Model explains {r2_engine*100:.2f}% of variance")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test, y_test, color='blue', alpha=0.5, label='Test Data')
    plt.plot(X_test, regressor_engine.coef_[0] * X_test + regressor_engine.intercept_, 
             color='red', linewidth=2, label='Prediction')
    plt.xlabel("Engine Size (L)", fontsize=12, fontweight='bold')
    plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
    plt.title(f'Evaluation: Engine Size (R²={r2_engine:.4f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/model_engine_testing.png', dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {viz_dir}/model_engine_testing.png")
    plt.show()
    
    return True

def block_8_save_engine_model():
    """Block 8: Save Engine Model"""
    global regressor_engine, model_dir
    
    print("\n" + "="*80)
    print("BLOCK 8: Save Model - ENGINE SIZE")
    print("="*80)
    
    if regressor_engine is None:
        print("✗ Error: Model not trained. Run Block 6 first.")
        return False
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = f'{model_dir}/co2_engine_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(regressor_engine, f)
    
    print(f"✓ Model saved: {filename}")
    print(f"  Size: {os.path.getsize(filename)} bytes")
    
    return True

def block_9_test_engine_predictions():
    """Block 9: Test Engine Predictions"""
    global regressor_engine
    
    print("\n" + "="*80)
    print("BLOCK 9: Sample Predictions - ENGINE SIZE")
    print("="*80)
    
    if regressor_engine is None:
        print("✗ Error: Model not trained. Run Block 6 first.")
        return False
    
    print("\n🔮 Sample Predictions:")
    samples = [2.0, 3.5, 5.0, 6.5]
    for size in samples:
        pred = regressor_engine.predict([[size]])
        print(f"  Engine {size}L → CO2: {pred[0]:.2f} g/km")
    
    return True

def block_10_train_fuel_model():
    """Block 10: Train Fuel Consumption Model"""
    global cdf, y, X_fuel, X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel, regressor_fuel
    
    print("\n" + "="*80)
    print("BLOCK 10: Train Linear Regression Model - FUEL CONSUMPTION")
    print("="*80)
    
    if cdf is None:
        print("✗ Error: Features not selected. Run Block 4 first.")
        return False
    
    # Prepare data
    X_fuel = cdf.FUELCONSUMPTION_COMB.to_numpy()
    
    # Split data
    X_train_fuel, X_test_fuel, y_train_fuel, y_test_fuel = train_test_split(
        X_fuel, y, test_size=0.2, random_state=42
    )
    
    print(f"📊 Data Split: {X_train_fuel.shape}, {X_test_fuel.shape}")
    
    # Train model
    regressor_fuel = LinearRegression()
    regressor_fuel.fit(X_train_fuel.reshape(-1, 1), y_train_fuel)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  Coefficient: {regressor_fuel.coef_[0]:.4f}")
    print(f"  Intercept: {regressor_fuel.intercept_:.4f}")
    print(f"  Equation: CO2 = {regressor_fuel.coef_[0]:.4f} × FuelConsumption + {regressor_fuel.intercept_:.4f}")
    
    return True

def block_11_evaluate_fuel_model():
    """Block 11: Evaluate Fuel Model"""
    global regressor_fuel, X_test_fuel, y_test_fuel, y_pred_fuel, accuracy_fuel, mae_fuel, mse_fuel, rmse_fuel, r2_fuel, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 11: Evaluate Model - FUEL CONSUMPTION")
    print("="*80)
    
    if regressor_fuel is None:
        print("✗ Error: Model not trained. Run Block 10 first.")
        return False
    
    # Predict
    y_pred_fuel = regressor_fuel.predict(X_test_fuel.reshape(-1, 1))
    
    # Calculate metrics
    accuracy_fuel = regressor_fuel.score(X_test_fuel.reshape(-1, 1), y_test_fuel)
    mae_fuel = mean_absolute_error(y_test_fuel, y_pred_fuel)
    mse_fuel = mean_squared_error(y_test_fuel, y_pred_fuel)
    rmse_fuel = np.sqrt(mse_fuel)
    r2_fuel = r2_score(y_test_fuel, y_pred_fuel)
    
    print(f"\n📊 Model Performance:")
    print(f"  Accuracy (R²): {accuracy_fuel:.4f}")
    print(f"  MAE: {mae_fuel:.2f}")
    print(f"  MSE: {mse_fuel:.2f}")
    print(f"  RMSE: {rmse_fuel:.2f}")
    print(f"  R² Score: {r2_fuel:.4f}")
    
    print(f"\n💡 Model explains {r2_fuel*100:.2f}% of variance")
    
    # Visualize
    plt.figure(figsize=(10, 6))
    plt.scatter(X_test_fuel, y_test_fuel, color='blue', alpha=0.5, label='Test Data')
    plt.plot(X_test_fuel, regressor_fuel.coef_[0] * X_test_fuel + regressor_fuel.intercept_, 
             color='red', linewidth=2, label='Prediction')
    plt.xlabel("Fuel Consumption (L/100km)", fontsize=12, fontweight='bold')
    plt.ylabel("CO2 Emissions (g/km)", fontsize=12, fontweight='bold')
    plt.title(f'Evaluation: Fuel Consumption (R²={r2_fuel:.4f})', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/model_fuel_testing.png', dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {viz_dir}/model_fuel_testing.png")
    plt.show()
    
    return True

def block_12_save_fuel_model():
    """Block 12: Save Fuel Model"""
    global regressor_fuel, model_dir
    
    print("\n" + "="*80)
    print("BLOCK 12: Save Model - FUEL CONSUMPTION")
    print("="*80)
    
    if regressor_fuel is None:
        print("✗ Error: Model not trained. Run Block 10 first.")
        return False
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = f'{model_dir}/co2_fuel_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(regressor_fuel, f)
    
    print(f"✓ Model saved: {filename}")
    print(f"  Size: {os.path.getsize(filename)} bytes")
    
    return True

def block_13_test_fuel_predictions():
    """Block 13: Test Fuel Predictions"""
    global regressor_fuel
    
    print("\n" + "="*80)
    print("BLOCK 13: Sample Predictions - FUEL CONSUMPTION")
    print("="*80)
    
    if regressor_fuel is None:
        print("✗ Error: Model not trained. Run Block 10 first.")
        return False
    
    print("\n🔮 Sample Predictions:")
    samples = [8.0, 10.5, 12.0, 15.5]
    for fuel in samples:
        pred = regressor_fuel.predict([[fuel]])
        print(f"  Fuel {fuel}L/100km → CO2: {pred[0]:.2f} g/km")
    
    return True

def block_14_compare_models():
    """Block 14: Compare Models"""
    global mae_engine, mse_engine, rmse_engine, r2_engine, accuracy_engine
    global mae_fuel, mse_fuel, rmse_fuel, r2_fuel, accuracy_fuel
    
    print("\n" + "="*80)
    print("BLOCK 14: Model Comparison")
    print("="*80)
    
    if r2_engine is None or r2_fuel is None:
        print("✗ Error: Both models must be evaluated first.")
        print("  Run Blocks 7 and 11.")
        return False
    
    comparison_data = {
        'Feature': ['Engine Size', 'Fuel Consumption'],
        'MAE': [mae_engine, mae_fuel],
        'MSE': [mse_engine, mse_fuel],
        'RMSE': [rmse_engine, rmse_fuel],
        'R²': [r2_engine, r2_fuel],
        'Accuracy': [accuracy_engine, accuracy_fuel]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    best = 'Engine Size' if r2_engine > r2_fuel else 'Fuel Consumption'
    best_r2 = max(r2_engine, r2_fuel)
    print(f"\n🏆 Best Model: {best} (R²={best_r2:.4f})")
    
    return True

# ============================================================================
# MENU SYSTEM
# ============================================================================

BLOCKS = {
    1: ("Import Libraries", block_1_import_libraries),
    2: ("Load Dataset", block_2_load_dataset),
    3: ("Explore Data", block_3_explore_data),
    4: ("Feature Selection", block_4_feature_selection),
    5: ("Visualize Data", block_5_visualize_data),
    6: ("Train Engine Model", block_6_train_engine_model),
    7: ("Evaluate Engine Model", block_7_evaluate_engine_model),
    8: ("Save Engine Model", block_8_save_engine_model),
    9: ("Test Engine Predictions", block_9_test_engine_predictions),
    10: ("Train Fuel Model", block_10_train_fuel_model),
    11: ("Evaluate Fuel Model", block_11_evaluate_fuel_model),
    12: ("Save Fuel Model", block_12_save_fuel_model),
    13: ("Test Fuel Predictions", block_13_test_fuel_predictions),
    14: ("Compare Models", block_14_compare_models),
}

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*80)
    print("CO2 PREDICTION - BLOCK EXECUTION MENU")
    print("="*80)
    
    for num, (name, _) in BLOCKS.items():
        print(f"{num:2d}. {name}")
    
    print("\n15. Run All Blocks")
    print(" 0. Exit")
    print("="*80)

def parse_input(user_input):
    """Parse user input (e.g., '1', '1,2,3', '1-5')"""
    blocks_to_run = []
    
    parts = user_input.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            # Range input (e.g., '1-5')
            try:
                start, end = map(int, part.split('-'))
                blocks_to_run.extend(range(start, end + 1))
            except:
                print(f"Invalid range: {part}")
        else:
            # Single block
            try:
                blocks_to_run.append(int(part))
            except:
                print(f"Invalid block number: {part}")
    
    return sorted(set(blocks_to_run))

def run_all_blocks():
    """Execute all blocks in sequence"""
    print("\n" + "="*80)
    print("RUNNING ALL BLOCKS")
    print("="*80)
    
    for num in sorted(BLOCKS.keys()):
        name, func = BLOCKS[num]
        print(f"\n{'='*80}")
        print(f"Executing Block {num}: {name}")
        print(f"{'='*80}")
        
        success = func()
        if not success:
            print(f"\n✗ Block {num} failed. Stopping execution.")
            return False
    
    print("\n" + "="*80)
    print("✓ ALL BLOCKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    return True

def main():
    """Main interactive loop"""
    print("\n" + "="*80)
    print("CO2 EMISSION PREDICTION - INTERACTIVE MODE")
    print("="*80)
    print("Execute individual blocks or groups of blocks.")
    print("Each block performs a specific task in the ML pipeline.")
    print("="*80)
    
    while True:
        show_menu()
        
        try:
            user_input = input("\nEnter block number(s) to execute (e.g., 1 or 1,2,3 or 1-5): ").strip()
            
            if user_input == '0':
                print("\n✓ Exiting. Goodbye!")
                break
            
            if user_input == '15':
                run_all_blocks()
                continue
            
            blocks_to_run = parse_input(user_input)
            
            if not blocks_to_run:
                print("✗ No valid blocks specified.")
                continue
            
            # Execute selected blocks
            for block_num in blocks_to_run:
                if block_num in BLOCKS:
                    name, func = BLOCKS[block_num]
                    print(f"\n{'='*80}")
                    print(f"Executing Block {block_num}: {name}")
                    print(f"{'='*80}")
                    func()
                else:
                    print(f"✗ Invalid block number: {block_num}")
            
            print("\n" + "="*80)
            print("✓ Execution completed!")
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted by user. Exiting.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
