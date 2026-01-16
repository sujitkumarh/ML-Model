# Drug Prediction - Decision Tree Classification

**Machine Learning Assignment**  
Predicts drug prescriptions based on patient features using Decision Tree Classifier

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

This project implements a complete machine learning pipeline for predicting drug prescriptions based on patient characteristics:
- **Age**
- **Sex** (Male/Female)
- **Blood Pressure** (HIGH/NORMAL/LOW)
- **Cholesterol** (HIGH/NORMAL)
- **Na to K Ratio** (Sodium to Potassium ratio)

**Target Drugs:** drugA, drugB, drugC, drugX, drugY

The project includes:
- ✅ Data loading and exploration
- ✅ Data visualization
- ✅ Feature encoding and preprocessing
- ✅ Model training (Decision Tree Classifier)
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
seaborn
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
pip install numpy pandas matplotlib seaborn scikit-learn
```

Or install specific versions:
```powershell
pip install numpy==1.24.3 pandas==2.0.3 matplotlib==3.9.3 seaborn==0.12.2 scikit-learn==1.3.0
```

### Step 3: Verify Installation
```powershell
python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('All packages installed successfully!')"
```

---

## 📁 File Structure

```
ML_A1_24020448074/
│
├── drug_identification.csv          # Dataset (required)
├── drug_prediction_final.py         # Main script (complete execution)
├── drug_prediction_interactive.py   # Interactive menu-driven script
├── README.md                         # This file
│
├── models/                           # Generated after execution
│   ├── drug_classifier_model.pkl    # Saved Decision Tree model
│   └── drug_encoders.pkl            # Saved label encoders
│
└── visualizations/                   # Generated after execution
    ├── drug_distribution.png
    ├── age_vs_na_to_k.png
    ├── feature_distributions.png
    ├── correlation_heatmap.png
    ├── feature_importance.png
    ├── decision_tree.png
    └── confusion_matrix.png
```

---

## 🚀 How to Execute

### Option 1: Complete Execution (Recommended for First Run)
Run the complete pipeline in one go:

```powershell
python drug_prediction_final.py
```

**This will:**
1. Load the dataset
2. Perform data exploration
3. Create visualizations
4. Preprocess and encode data
5. Train the Decision Tree model
6. Evaluate model performance
7. Save the trained model
8. Generate all output files

**Expected output:**
- 7 visualization PNG files in `visualizations/` folder
- 2 model files in `models/` folder
- Console output with detailed metrics and progress

---

### Option 2: Interactive Block Execution (For Learning/Debugging)
Execute specific blocks or experiment step-by-step:

```powershell
python drug_prediction_interactive.py
```

**Interactive Menu:**
```
DRUG PREDICTION - BLOCK EXECUTION MENU
================================================================================
 1. Import Libraries
 2. Load Dataset
 3. Explore Data
 4. Visualize Data
 5. Preprocess & Encode Data
 6. Prepare Features & Target
 7. Split Train/Test Data
 8. Train Decision Tree
 9. Visualize Decision Tree
10. Make Predictions
11. Evaluate Model
12. Save Model & Encoders
13. Test Sample Predictions

14. Run All Blocks
 0. Exit
================================================================================
```

**Usage Examples:**
- Run single block: `1`
- Run multiple blocks: `1,2,3`
- Run range of blocks: `1-5`
- Run all blocks: `14`

---

## 📊 Execution Modes

### Mode 1: Full Automated Execution
```powershell
python drug_prediction_final.py
```
- **Duration:** ~30-60 seconds
- **Use Case:** Complete analysis, production deployment
- **Output:** All visualizations and models

### Mode 2: Interactive Mode
```powershell
python drug_prediction_interactive.py
```
- **Duration:** Variable (user-controlled)
- **Use Case:** Learning, debugging, experimenting
- **Output:** Selected blocks only

### Mode 3: Custom Script
Load the saved model and make predictions:
```python
import pickle
import numpy as np

# Load model and encoders
model = pickle.load(open('models/drug_classifier_model.pkl', 'rb'))
encoders = pickle.load(open('models/drug_encoders.pkl', 'rb'))

# Prepare new patient data
# Format: [Age, Sex(F=0/M=1), BP(HIGH=0/LOW=1/NORMAL=2), Cholesterol(HIGH=0/NORMAL=1), Na_to_K]
patient = [35, 1, 0, 1, 14.5]  # Example: 35 year old male, HIGH BP, NORMAL cholesterol, Na_to_K=14.5

# Make prediction
prediction = model.predict([patient])
predicted_drug = encoders['le_drug'].inverse_transform(prediction)[0]

print(f"Predicted Drug: {predicted_drug}")
```

---

## 📈 Output Files

### Visualizations (visualizations/ folder)

1. **drug_distribution.png**
   - Bar chart showing count of each drug type
   - Helps understand target variable distribution

2. **age_vs_na_to_k.png**
   - Scatter plot of Age vs Na_to_K ratio
   - Color-coded by drug type
   - Shows relationship between features and target

3. **feature_distributions.png**
   - 4-panel plot showing distribution of all features
   - Age histogram, Na_to_K histogram
   - Sex distribution, Blood Pressure distribution

4. **correlation_heatmap.png**
   - Correlation matrix for numerical features
   - Shows relationships between Age and Na_to_K

5. **feature_importance.png**
   - Horizontal bar chart showing feature importance scores
   - Identifies most influential features

6. **decision_tree.png**
   - Complete visualization of the trained decision tree
   - Shows decision rules and leaf nodes
   - Color-coded by predicted class

7. **confusion_matrix.png**
   - Heatmap showing actual vs predicted classifications
   - Diagonal shows correct predictions
   - Off-diagonal shows misclassifications

### Models (models/ folder)

1. **drug_classifier_model.pkl**
   - Trained Decision Tree Classifier
   - Can be loaded for predictions
   - Size: ~5-10 KB

2. **drug_encoders.pkl**
   - Label encoders for categorical features
   - Required to encode new data before prediction
   - Contains: le_sex, le_bp, le_chol, le_drug

---

## 🔬 Model Details

### Algorithm: Decision Tree Classifier
- **Criterion:** Entropy (Information Gain)
- **Max Depth:** 4 (prevents overfitting)
- **Random State:** 42 (for reproducibility)

### Features Used:
1. **Age** (Numerical, 15-74 years)
2. **Sex** (Categorical: F=0, M=1)
3. **BP** (Categorical: HIGH=0, LOW=1, NORMAL=2)
4. **Cholesterol** (Categorical: HIGH=0, NORMAL=1)
5. **Na_to_K** (Numerical, ratio of Sodium to Potassium)

### Target Variable:
- **Drug** (Categorical: drugA, drugB, drugC, drugX, drugY)

### Performance Metrics:
The model typically achieves:
- **Training Accuracy:** 98-100%
- **Testing Accuracy:** 95-98%
- **Precision, Recall, F1-Score:** 0.90+ for most classes

---

## 💡 Model Usage

### Loading the Model
```python
import pickle

# Load model
with open('models/drug_classifier_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load encoders
with open('models/drug_encoders.pkl', 'rb') as f:
    encoders = pickle.load(f)

le_sex = encoders['le_sex']
le_bp = encoders['le_bp']
le_chol = encoders['le_chol']
le_drug = encoders['le_drug']
```

### Making Predictions
```python
# Example patient data
patient_data = {
    'Age': 45,
    'Sex': 'M',
    'BP': 'HIGH',
    'Cholesterol': 'NORMAL',
    'Na_to_K': 12.5
}

# Encode categorical features
patient_encoded = [
    patient_data['Age'],
    le_sex.transform([patient_data['Sex']])[0],
    le_bp.transform([patient_data['BP']])[0],
    le_chol.transform([patient_data['Cholesterol']])[0],
    patient_data['Na_to_K']
]

# Make prediction
prediction = model.predict([patient_encoded])
predicted_drug = le_drug.inverse_transform(prediction)[0]

# Get prediction probabilities
prediction_proba = model.predict_proba([patient_encoded])

print(f"Predicted Drug: {predicted_drug}")
print(f"Confidence: {prediction_proba[0][prediction[0]]*100:.1f}%")
```

### Batch Predictions
```python
import pandas as pd

# Multiple patients
patients = pd.DataFrame([
    {'Age': 25, 'Sex': 'F', 'BP': 'HIGH', 'Cholesterol': 'HIGH', 'Na_to_K': 15.5},
    {'Age': 50, 'Sex': 'M', 'BP': 'NORMAL', 'Cholesterol': 'NORMAL', 'Na_to_K': 10.2},
    {'Age': 68, 'Sex': 'F', 'BP': 'LOW', 'Cholesterol': 'HIGH', 'Na_to_K': 28.5}
])

# Encode all patients
X_new = []
for _, patient in patients.iterrows():
    encoded = [
        patient['Age'],
        le_sex.transform([patient['Sex']])[0],
        le_bp.transform([patient['BP']])[0],
        le_chol.transform([patient['Cholesterol']])[0],
        patient['Na_to_K']
    ]
    X_new.append(encoded)

# Predict
predictions = model.predict(X_new)
predicted_drugs = le_drug.inverse_transform(predictions)

# Display results
patients['Predicted_Drug'] = predicted_drugs
print(patients)
```

---

## 🔧 Troubleshooting

### Issue 1: "ModuleNotFoundError: No module named 'xxx'"
**Solution:**
```powershell
pip install numpy pandas matplotlib seaborn scikit-learn
```

### Issue 2: "FileNotFoundError: drug_identification.csv not found"
**Solution:**
- Ensure `drug_identification.csv` is in the same directory as the script
- Or update the file path in the script to absolute path

### Issue 3: "Permission denied" when creating folders
**Solution:**
- Run PowerShell as Administrator
- Or change the working directory to a location with write permissions

### Issue 4: Plots not showing
**Solution:**
- Plots are automatically saved to `visualizations/` folder
- If using Jupyter/IDE, ensure matplotlib backend is configured
- Check if `plt.show()` is being blocked by environment

### Issue 5: Low model accuracy
**Solution:**
- Verify data quality (no missing/corrupted values)
- Try different hyperparameters (max_depth, criterion)
- Consider feature engineering or additional features
- Try ensemble methods (Random Forest, Gradient Boosting)

### Issue 6: Model file size too large
**Solution:**
- Decision Tree models are typically small (<10 KB)
- If large, reduce max_depth or prune the tree
- Consider model compression techniques

---

## 📚 Understanding the Output

### Classification Report Explanation
```
              precision    recall  f1-score   support

      drugA       0.95      0.93      0.94         15
      drugB       0.92      0.95      0.93         20
      drugC       0.97      0.94      0.95         18
      drugX       0.96      0.98      0.97         25
      drugY       0.98      0.97      0.98         30

   accuracy                           0.96        108
  macro avg       0.96      0.95      0.95        108
weighted avg       0.96      0.96      0.96        108
```

- **Precision:** Of all positive predictions, how many were correct?
- **Recall:** Of all actual positives, how many were found?
- **F1-Score:** Harmonic mean of precision and recall
- **Support:** Number of actual occurrences of each class

### Confusion Matrix Interpretation
```
[[14  1  0  0  0]   <- Actual drugA
 [ 1 19  0  0  0]   <- Actual drugB
 [ 0  0 17  1  0]   <- Actual drugC
 [ 0  0  1 24  0]   <- Actual drugX
 [ 0  0  0  1 29]]  <- Actual drugY
```
- Diagonal elements: Correct predictions
- Off-diagonal: Misclassifications

---

## 🎓 Learning Objectives

This project demonstrates:
1. **Data Loading & Exploration** - Understanding dataset structure
2. **Data Visualization** - Creating meaningful plots
3. **Data Preprocessing** - Encoding categorical variables
4. **Feature Engineering** - Selecting relevant features
5. **Model Training** - Decision Tree Classifier
6. **Model Evaluation** - Accuracy, precision, recall, F1-score
7. **Model Persistence** - Saving/loading models with pickle
8. **Prediction Pipeline** - End-to-end inference

---

## 🚀 Next Steps & Improvements

### Model Improvements:
1. **Hyperparameter Tuning**
   - GridSearchCV for optimal parameters
   - Try different max_depth values
   - Experiment with min_samples_split, min_samples_leaf

2. **Alternative Algorithms**
   - Random Forest Classifier
   - Support Vector Machine (SVM)
   - Neural Networks
   - Ensemble methods (Voting, Stacking)

3. **Feature Engineering**
   - Create interaction features (Age × Na_to_K)
   - Binning continuous variables
   - Polynomial features

4. **Cross-Validation**
   - K-Fold cross-validation
   - Stratified K-Fold
   - Time series cross-validation (if temporal data)

5. **Model Interpretability**
   - SHAP values
   - LIME explanations
   - Feature interaction plots

### Code Improvements:
1. Add logging functionality
2. Implement command-line arguments
3. Create web API using Flask/FastAPI
4. Add unit tests
5. Create Docker container
6. Deploy to cloud (AWS, Azure, GCP)

---

## 📞 Support

For issues or questions:
1. Check the Troubleshooting section
2. Review the code comments
3. Consult scikit-learn documentation: https://scikit-learn.org/
4. Check Python package documentation

---

## 📄 License

This project is for educational purposes as part of Machine Learning Assignment.

---

## 🙏 Acknowledgments

- Dataset: Drug Identification Dataset (drug_identification.csv)
- Libraries: NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
- Algorithm: Decision Tree Classifier (CART algorithm)

---

**Last Updated:** January 2026  
**Python Version:** 3.7+  
**Scikit-learn Version:** 1.3.0+

---

## Quick Reference Card

### Encoding Mappings:
```python
Sex:         F=0, M=1
BP:          HIGH=0, LOW=1, NORMAL=2
Cholesterol: HIGH=0, NORMAL=1
Drug:        drugA=0, drugB=1, drugC=2, drugX=3, drugY=4
```

### Quick Commands:
```powershell
# Install packages
pip install numpy pandas matplotlib seaborn scikit-learn

# Run full pipeline
python drug_prediction_final.py

# Run interactive mode
python drug_prediction_interactive.py

# Check Python version
python --version

# Verify packages
python -c "import numpy, pandas, matplotlib, seaborn, sklearn; print('OK')"
```

### File Sizes (Approximate):
- drug_identification.csv: ~5 KB
- drug_classifier_model.pkl: ~5-10 KB
- drug_encoders.pkl: ~2-5 KB
- Each visualization PNG: 50-200 KB

---

**✅ Project Status:** Complete and Ready for Execution
