# CO2 Emission Prediction - Linear Regression

**Machine Learning Assignment**  
Predicts CO2 emissions based on vehicle features using Linear Regression

---

## 📋 Table of Contents
- [Overview](#overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [File Structure](#file-structure)
- [How to Execute](#how-to-execute)
- [Execution Modes](#execution-modes)
- [Output Files](#output-files)
- [Model Usage](#model-usage)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

This project implements a complete machine learning pipeline for predicting CO2 emissions from vehicles based on:
- **Engine Size**
- **Number of Cylinders**
- **Fuel Consumption**

The project includes:
- ✅ Data loading and exploration
- ✅ Data visualization
- ✅ Feature selection and preprocessing
- ✅ Model training (2 models)
- ✅ Model evaluation
- ✅ Model saving (pickle format)
- ✅ Sample predictions

---

## 💻 Prerequisites

### Required Software:
- **Python 3.7 or higher**
- **pip** (Python package installer)

### Required Packages:
```bash
numpy
pandas
matplotlib
scikit-learn
```

---

## 📦 Installation

### Step 1: Verify Python Installation
```powershell
python --version
```

### Step 2: Install Required Packages
```powershell
pip install numpy pandas matplotlib scikit-learn
```

Or install specific versions:
```powershell
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.9.3 scikit-learn==1.3.0
```

### Step 3: Verify Installation
```powershell
python -c "import numpy, pandas, matplotlib, sklearn; print('All packages installed successfully!')"
```

---

## 📁 File Structure

```
ML_A1_24020448074/
│
├── FuelConsumptionCo2.csv          # Dataset (required)
├── CO2_Prediction_Final.py         # Main script (complete execution)
├── CO2_Prediction_Interactive.py   # Interactive menu-driven script
├── README.md                        # This file
│
├── models/                          # Generated after execution
│   ├── co2_engine_model.pkl        # Saved Engine Size model
│   └── co2_fuel_model.pkl          # Saved Fuel Consumption model
│
└── visualizations/                  # Generated after execution
    ├── feature_distributions.png
    ├── fuel_vs_co2.png
    ├── engine_vs_co2.png
    ├── cylinders_vs_co2.png
    ├── model_engine_training.png
    ├── model_engine_testing.png
    └── model_fuel_testing.png
```

---

## 🚀 How to Execute

### **Option 1: Complete Execution (All Blocks)**
Run the entire script from start to finish:

```powershell
python CO2_Prediction_Final.py
```

This will:
- Execute all 22 blocks sequentially
- Generate all visualizations
- Train and save both models
- Display all results

**Time:** ~2-3 minutes

---

### **Option 2: Interactive Block Execution**
Run blocks individually with a menu:

```powershell
python CO2_Prediction_Interactive.py
```

**Menu Options:**
```
=== CO2 PREDICTION - BLOCK EXECUTION MENU ===
1.  Import Libraries
2.  Load Dataset
3.  Explore Data
4.  Feature Selection
5.  Visualize Data
6.  Train Engine Model
7.  Evaluate Engine Model
8.  Save Engine Model
9.  Test Engine Predictions
10. Train Fuel Model
11. Evaluate Fuel Model
12. Save Fuel Model
13. Test Fuel Predictions
14. Compare Models
15. Run All Blocks
0.  Exit

Enter block number(s) to execute (e.g., 1 or 1,2,3 or 1-5): 
```

**Examples:**
- Execute single block: `1`
- Execute multiple blocks: `1,2,3`
- Execute range: `1-5` (blocks 1 through 5)
- Execute all: `15`

---

### **Option 3: Jupyter Notebook (Recommended for Block-by-Block)**

Convert to Jupyter Notebook for cell-by-cell execution:

1. **Install Jupyter:**
```powershell
pip install jupyter
```

2. **Create notebook version:**
```powershell
# Open VS Code
# Right-click on CO2_Prediction_Final.py
# Select "Export to Jupyter Notebook"
```

3. **Run Jupyter:**
```powershell
jupyter notebook
```

4. **Open the notebook and run cells individually** (Shift + Enter)

---

## 🎮 Execution Modes

### **Mode 1: Full Automatic Execution**
```powershell
python CO2_Prediction_Final.py
```
- **Pros:** Complete analysis in one run, no interaction needed
- **Cons:** Can't skip blocks, runs everything
- **Best for:** Final execution, demonstrations, production

### **Mode 2: Interactive Menu Execution**
```powershell
python CO2_Prediction_Interactive.py
```
- **Pros:** Choose which blocks to run, flexible, rerun specific blocks
- **Cons:** Requires user input
- **Best for:** Learning, debugging, customization

### **Mode 3: Jupyter Notebook**
```powershell
jupyter notebook
```
- **Pros:** Visual, cell-by-cell execution, can edit inline
- **Cons:** Requires Jupyter installation
- **Best for:** Exploration, learning, presentations

---

## 📊 Output Files

### **Generated Directories:**

#### `visualizations/` (7 PNG files)
1. **feature_distributions.png** - Histogram of all features
2. **fuel_vs_co2.png** - Scatter plot: Fuel Consumption vs CO2
3. **engine_vs_co2.png** - Scatter plot: Engine Size vs CO2
4. **cylinders_vs_co2.png** - Scatter plot: Cylinders vs CO2
5. **model_engine_training.png** - Training data with regression line
6. **model_engine_testing.png** - Test data evaluation (Engine)
7. **model_fuel_testing.png** - Test data evaluation (Fuel)

#### `models/` (2 PKL files)
1. **co2_engine_model.pkl** - Trained Engine Size model
2. **co2_fuel_model.pkl** - Trained Fuel Consumption model

---

## 🔮 Model Usage

### **Load Saved Model:**
```python
import pickle

# Load engine model
with open('models/co2_engine_model.pkl', 'rb') as f:
    engine_model = pickle.load(f)

# Load fuel model
with open('models/co2_fuel_model.pkl', 'rb') as f:
    fuel_model = pickle.load(f)
```

### **Make Predictions:**
```python
# Predict CO2 for 4.2L engine
engine_prediction = engine_model.predict([[4.2]])
print(f"CO2 Emissions: {engine_prediction[0]:.2f} g/km")

# Predict CO2 for 11.5 L/100km fuel consumption
fuel_prediction = fuel_model.predict([[11.5]])
print(f"CO2 Emissions: {fuel_prediction[0]:.2f} g/km")
```

### **Example: Interactive Prediction Script**
```python
import pickle

# Load model
model = pickle.load(open('models/co2_engine_model.pkl', 'rb'))

# Get user input
engine_size = float(input("Enter engine size (L): "))

# Predict
co2 = model.predict([[engine_size]])
print(f"Predicted CO2 Emissions: {co2[0]:.2f} g/km")
```

---

## 🔧 Troubleshooting

### **Issue 1: "Module not found" Error**
```
ModuleNotFoundError: No module named 'sklearn'
```
**Solution:**
```powershell
pip install scikit-learn
```

### **Issue 2: "File not found" Error**
```
FileNotFoundError: FuelConsumptionCo2.csv
```
**Solution:**
- Ensure `FuelConsumptionCo2.csv` is in the same directory as the script
- Or update the `data_path` variable in the script with the correct path

### **Issue 3: Matplotlib Display Issues**
**Solution:**
```python
# Add at the top of script if plots don't show
import matplotlib
matplotlib.use('TkAgg')  # or 'Qt5Agg'
```

### **Issue 4: Permission Denied (Saving Files)**
**Solution:**
- Run PowerShell as Administrator
- Or save files to a different directory with write permissions

### **Issue 5: Python Version Too Old**
```
SyntaxError: invalid syntax
```
**Solution:**
```powershell
# Check Python version
python --version

# Upgrade Python to 3.7+
# Download from: https://www.python.org/downloads/
```

---

## 📈 Expected Results

### **Model Performance:**
- **Engine Size Model:**
  - R² Score: ~0.75-0.85
  - RMSE: ~25-30 g/km
  
- **Fuel Consumption Model:**
  - R² Score: ~0.85-0.95
  - RMSE: ~15-20 g/km

*Note: Actual values may vary slightly due to random state in train/test split*

---

## 🎓 Block Descriptions

| Block | Name | Description | Dependencies |
|-------|------|-------------|--------------|
| 1 | Import Libraries | Load required packages | None |
| 2 | Load Dataset | Read CSV file into DataFrame | Block 1 |
| 3 | Explore Data | Statistical summary, data types | Block 2 |
| 4 | Feature Selection | Extract relevant columns | Block 2 |
| 5 | Visualize Data | Create histograms and scatter plots | Block 4 |
| 6 | Train Engine Model | Build regression model (Engine) | Block 4 |
| 7 | Evaluate Engine Model | Calculate performance metrics | Block 6 |
| 8 | Save Engine Model | Export model as .pkl file | Block 6 |
| 9 | Test Engine Predictions | Sample predictions | Block 6 |
| 10 | Train Fuel Model | Build regression model (Fuel) | Block 4 |
| 11 | Evaluate Fuel Model | Calculate performance metrics | Block 10 |
| 12 | Save Fuel Model | Export model as .pkl file | Block 10 |
| 13 | Test Fuel Predictions | Sample predictions | Block 10 |
| 14 | Compare Models | Performance comparison table | Blocks 7, 11 |

---

## 📝 Notes

- **Portability:** All scripts are self-contained and portable
- **Data Path:** Update `data_path` variable if CSV is in different location
- **Random State:** `random_state=42` ensures reproducible results
- **Train/Test Split:** 80% training, 20% testing
- **Model Format:** Pickle (.pkl) format for easy loading

---

## 🚀 Quick Start

```powershell
# 1. Navigate to project directory
cd "C:\Users\shujare\OneDrive - Capgemini\Attachments\SIMS\SEM4\ML\ML_A1_24020448074"

# 2. Install packages
pip install numpy pandas matplotlib scikit-learn

# 3. Run complete analysis
python CO2_Prediction_Final.py

# OR run interactive mode
python CO2_Prediction_Interactive.py
```

---

## 📞 Support

For issues or questions:
1. Check the [Troubleshooting](#troubleshooting) section
2. Verify all prerequisites are installed
3. Ensure `FuelConsumptionCo2.csv` is in the correct location

---

## 📄 License

This project is for educational purposes (Machine Learning Assignment).

---

**Created:** January 2026  
**Assignment:** ML_A1_24020448074  
**Language:** Python 3.7+
