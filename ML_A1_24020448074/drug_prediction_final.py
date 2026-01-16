"""
Drug Prediction using Decision Tree Classifier
==============================================
This script predicts drug prescriptions based on patient features using decision tree classification.
Adapted from CO2 Prediction structure for Drug Identification dataset.

Author: ML Assignment
Date: January 2026
"""

# ============================================================================
# BLOCK 1: PREREQUISITES & ENVIRONMENT SETUP
# ============================================================================
"""
Prerequisites:
1. Python 3.7 or higher
2. Required packages: numpy, pandas, matplotlib, seaborn, scikit-learn
3. CSV data file in the same directory
4. Install packages: pip install numpy pandas matplotlib seaborn scikit-learn
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

print("="*80)
print("DRUG PREDICTION - DECISION TREE CLASSIFICATION MODEL")
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
    import seaborn as sns  # Statistical data visualization
    print("✓ Seaborn imported successfully")
except ImportError:
    print("✗ Seaborn not found. Install using: pip install seaborn")
    sys.exit(1)

try:
    from sklearn.model_selection import train_test_split  # Split data into train and test sets
    from sklearn.tree import DecisionTreeClassifier, plot_tree  # Decision tree classifier
    from sklearn.preprocessing import LabelEncoder  # Encode categorical features
    from sklearn.metrics import accuracy_score, classification_report, confusion_matrix  # Evaluation metrics
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
DATA_FILE = "drug_identification.csv"  # CSV file name

# Check if file exists in current directory
if os.path.exists(DATA_FILE):
    data_path = DATA_FILE
    print(f"✓ Data file found: {os.path.abspath(DATA_FILE)}")
else:
    # Try absolute path
    data_path = r"C:\Users\shujare\OneDrive - Capgemini\Attachments\SIMS\SEM4\ML\ML_A1_24020448074\drug_identification.csv"
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

print("\n📊 Target Variable Distribution:")
print(df['Drug'].value_counts())  # Count of each drug type

print("\n📊 Categorical Features Distribution:")
print("\nSex Distribution:")
print(df['Sex'].value_counts())
print("\nBP Distribution:")
print(df['BP'].value_counts())
print("\nCholesterol Distribution:")
print(df['Cholesterol'].value_counts())

print("\n" + "="*80)

# ============================================================================
# BLOCK 5: DATA VISUALIZATION
# ============================================================================
print("\n[STEP 4] Data Visualization...")
print("-"*80)

# Create visualizations directory
viz_dir = "visualizations"
if not os.path.exists(viz_dir):
    os.makedirs(viz_dir)
    print(f"✓ Created directory: {viz_dir}/")

# 5.1: Drug distribution (target variable)
print("\n📊 Creating drug distribution chart...")
plt.figure(figsize=(10, 6))
drug_counts = df['Drug'].value_counts()
plt.bar(drug_counts.index, drug_counts.values, color=['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FECA57'])
plt.xlabel("Drug Type", fontsize=12, fontweight='bold')
plt.ylabel("Count", fontsize=12, fontweight='bold')
plt.title('Distribution of Drug Types', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{viz_dir}/drug_distribution.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/drug_distribution.png")
plt.show()

# 5.2: Age vs Na_to_K scatter plot by drug
print("\n📈 Creating scatter plots...")
plt.figure(figsize=(12, 6))
for drug in df['Drug'].unique():
    drug_data = df[df['Drug'] == drug]
    plt.scatter(drug_data['Age'], drug_data['Na_to_K'], label=drug, alpha=0.6, s=50)
plt.xlabel("Age", fontsize=12, fontweight='bold')
plt.ylabel("Na to K Ratio", fontsize=12, fontweight='bold')
plt.title('Age vs Na_to_K Ratio by Drug Type', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{viz_dir}/age_vs_na_to_k.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/age_vs_na_to_k.png")
plt.show()

# 5.3: Feature distributions
print("\n📊 Creating feature distribution plots...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Age distribution
axes[0, 0].hist(df['Age'], bins=20, edgecolor='black', color='skyblue')
axes[0, 0].set_xlabel('Age', fontweight='bold')
axes[0, 0].set_ylabel('Frequency', fontweight='bold')
axes[0, 0].set_title('Age Distribution', fontweight='bold')
axes[0, 0].grid(True, alpha=0.3)

# Na_to_K distribution
axes[0, 1].hist(df['Na_to_K'], bins=20, edgecolor='black', color='lightgreen')
axes[0, 1].set_xlabel('Na to K Ratio', fontweight='bold')
axes[0, 1].set_ylabel('Frequency', fontweight='bold')
axes[0, 1].set_title('Na_to_K Distribution', fontweight='bold')
axes[0, 1].grid(True, alpha=0.3)

# Sex distribution
sex_counts = df['Sex'].value_counts()
axes[1, 0].bar(sex_counts.index, sex_counts.values, color=['lightcoral', 'lightblue'])
axes[1, 0].set_xlabel('Sex', fontweight='bold')
axes[1, 0].set_ylabel('Count', fontweight='bold')
axes[1, 0].set_title('Sex Distribution', fontweight='bold')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# BP distribution
bp_counts = df['BP'].value_counts()
axes[1, 1].bar(bp_counts.index, bp_counts.values, color=['#FF6B6B', '#FFA07A', '#98D8C8'])
axes[1, 1].set_xlabel('Blood Pressure', fontweight='bold')
axes[1, 1].set_ylabel('Count', fontweight='bold')
axes[1, 1].set_title('Blood Pressure Distribution', fontweight='bold')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(f'{viz_dir}/feature_distributions.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/feature_distributions.png")
plt.show()

# 5.4: Correlation heatmap (numerical features only)
print("\n📊 Creating correlation heatmap...")
plt.figure(figsize=(8, 6))
numerical_cols = ['Age', 'Na_to_K']
corr_matrix = df[numerical_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=1)
plt.title('Correlation Heatmap - Numerical Features', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/correlation_heatmap.png")
plt.show()

print("\n✓ All visualizations completed!")
print("="*80)

# ============================================================================
# BLOCK 6: DATA PREPROCESSING & ENCODING
# ============================================================================
print("\n[STEP 5] Data Preprocessing & Encoding...")
print("-"*80)

# Create a copy for preprocessing
df_processed = df.copy()

# Initialize label encoders
le_sex = LabelEncoder()
le_bp = LabelEncoder()
le_chol = LabelEncoder()
le_drug = LabelEncoder()

# Encode categorical features
print("\n🔄 Encoding categorical features...")
df_processed['Sex'] = le_sex.fit_transform(df['Sex'])  # F=0, M=1
df_processed['BP'] = le_bp.fit_transform(df['BP'])  # HIGH=0, LOW=1, NORMAL=2
df_processed['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])  # HIGH=0, NORMAL=1
df_processed['Drug'] = le_drug.fit_transform(df['Drug'])  # drugA=0, drugB=1, drugC=2, drugX=3, drugY=4

print(f"✓ Sex encoding: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
print(f"✓ BP encoding: {dict(zip(le_bp.classes_, le_bp.transform(le_bp.classes_)))}")
print(f"✓ Cholesterol encoding: {dict(zip(le_chol.classes_, le_chol.transform(le_chol.classes_)))}")
print(f"✓ Drug encoding: {dict(zip(le_drug.classes_, le_drug.transform(le_drug.classes_)))}")

print("\nSample of encoded data (5 records):")
print(df_processed.head())

print("\n" + "="*80)

# ============================================================================
# BLOCK 7: PREPARE DATA - SPLIT FEATURES AND TARGET
# ============================================================================
print("\n[STEP 6] Prepare Data - Split Features and Target...")
print("-"*80)

# Separate features (X) and target (y)
X = df_processed[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values  # All features
y = df_processed['Drug'].values  # Target variable

print(f"\n✓ Features (X): shape {X.shape}")
print(f"  Feature names: Age, Sex, BP, Cholesterol, Na_to_K")
print(f"✓ Target (y): shape {y.shape}")
print(f"  Classes: {le_drug.classes_}")

print("\n" + "="*80)

# ============================================================================
# BLOCK 8: SPLIT DATA - TRAIN AND TEST SETS
# ============================================================================
print("\n[STEP 7] Split Data into Training and Testing Sets...")
print("-"*80)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y  # 80% train, 20% test with stratification
)

print(f"\n📊 Data Split:")
print(f"  Training set: {X_train.shape}")
print(f"  Testing set: {X_test.shape}")
print(f"  Target train: {y_train.shape}")
print(f"  Target test: {y_test.shape}")
print(f"  Split ratio: 80% training, 20% testing")
print(f"  Stratification: Enabled (preserves class distribution)")

print("\n📊 Class distribution in train set:")
unique, counts = np.unique(y_train, return_counts=True)
for drug_id, count in zip(unique, counts):
    print(f"  {le_drug.inverse_transform([drug_id])[0]}: {count} samples")

print("\n" + "="*80)

# ============================================================================
# BLOCK 9: TRAIN DECISION TREE CLASSIFIER
# ============================================================================
print("\n[STEP 8] Train the Decision Tree Classifier...")
print("-"*80)

# Create and train model
classifier = DecisionTreeClassifier(
    criterion='entropy',  # Use entropy for information gain
    max_depth=4,  # Limit tree depth to prevent overfitting
    random_state=42  # For reproducibility
)
classifier.fit(X_train, y_train)  # Fit the model on training data

print("\n✓ Model trained successfully!")
print(f"  Criterion: {classifier.criterion}")
print(f"  Max depth: {classifier.max_depth}")
print(f"  Number of leaves: {classifier.get_n_leaves()}")
print(f"  Tree depth: {classifier.get_depth()}")
print(f"  Number of features: {classifier.n_features_in_}")

# Feature importance
feature_importance = classifier.feature_importances_
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']

print("\n📊 Feature Importance:")
for name, importance in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
    print(f"  {name:15s}: {importance:.4f}")

# Visualize feature importance
plt.figure(figsize=(10, 6))
sorted_idx = np.argsort(feature_importance)
plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='skyblue')
plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
plt.xlabel('Importance', fontsize=12, fontweight='bold')
plt.ylabel('Features', fontsize=12, fontweight='bold')
plt.title('Feature Importance - Decision Tree', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig(f'{viz_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {viz_dir}/feature_importance.png")
plt.show()

# Visualize decision tree
print("\n📊 Visualizing decision tree...")
plt.figure(figsize=(20, 12))
plot_tree(classifier, 
          feature_names=feature_names,
          class_names=le_drug.classes_,
          filled=True,
          rounded=True,
          fontsize=10)
plt.title('Decision Tree Visualization', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/decision_tree.png', dpi=300, bbox_inches='tight')
print(f"  ✓ Saved: {viz_dir}/decision_tree.png")
plt.show()

print("="*80)

# ============================================================================
# BLOCK 10: MAKE PREDICTIONS
# ============================================================================
print("\n[STEP 9] Make Predictions on Test Set...")
print("-"*80)

# Make predictions on test set
y_pred = classifier.predict(X_test)  # Predict target values for test features
y_pred_proba = classifier.predict_proba(X_test)  # Prediction probabilities

print("\n✓ Predictions completed!")
print(f"  Sample predictions (first 5):")
for i in range(min(5, len(y_pred))):
    actual_drug = le_drug.inverse_transform([y_test[i]])[0]
    predicted_drug = le_drug.inverse_transform([y_pred[i]])[0]
    confidence = y_pred_proba[i][y_pred[i]] * 100
    print(f"    Actual: {actual_drug}  →  Predicted: {predicted_drug} (Confidence: {confidence:.1f}%)")

print("\n" + "="*80)

# ============================================================================
# BLOCK 11: EVALUATE MODEL
# ============================================================================
print("\n[STEP 10] Evaluate Model Performance...")
print("-"*80)

# Calculate evaluation metrics
train_accuracy = classifier.score(X_train, y_train)  # Training accuracy
test_accuracy = accuracy_score(y_test, y_pred)  # Test accuracy

print("\n📊 Model Performance Metrics:")
print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  Testing Accuracy:  {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")

# Interpret results
print(f"\n💡 Model Interpretation:")
if test_accuracy > 0.9:
    print(f"  ✓ Excellent performance!")
elif test_accuracy > 0.8:
    print(f"  ✓ Good performance!")
elif test_accuracy > 0.7:
    print(f"  ⚠ Moderate performance")
else:
    print(f"  ✗ Poor performance - consider feature engineering or model tuning")

# Detailed classification report
print("\n📋 Detailed Classification Report:")
print(classification_report(y_test, y_pred, target_names=le_drug.classes_))

# Confusion matrix
print("\n📊 Confusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)

# Visualize confusion matrix
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=le_drug.classes_, 
            yticklabels=le_drug.classes_,
            cbar_kws={'label': 'Count'})
plt.xlabel('Predicted Drug', fontsize=12, fontweight='bold')
plt.ylabel('Actual Drug', fontsize=12, fontweight='bold')
plt.title(f'Confusion Matrix (Accuracy: {test_accuracy:.2%})', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{viz_dir}/confusion_matrix.png', dpi=300, bbox_inches='tight')
print(f"\n  ✓ Saved: {viz_dir}/confusion_matrix.png")
plt.show()

print("="*80)

# ============================================================================
# BLOCK 12: SAVE MODEL AND ENCODERS
# ============================================================================
print("\n[STEP 11] Save the Trained Model and Encoders...")
print("-"*80)

# Create models directory
model_dir = "models"
if not os.path.exists(model_dir):
    os.makedirs(model_dir)
    print(f"✓ Created directory: {model_dir}/")

# Save the trained model
model_filename = f'{model_dir}/drug_classifier_model.pkl'
with open(model_filename, 'wb') as f:  # Open file in write-binary mode
    pickle.dump(classifier, f)  # Serialize and save the model object

print(f"\n✓ Model saved successfully!")
print(f"  File: {model_filename}")
print(f"  Size: {os.path.getsize(model_filename)} bytes")

# Save the label encoders
encoders = {
    'le_sex': le_sex,
    'le_bp': le_bp,
    'le_chol': le_chol,
    'le_drug': le_drug
}

encoders_filename = f'{model_dir}/drug_encoders.pkl'
with open(encoders_filename, 'wb') as f:
    pickle.dump(encoders, f)

print(f"✓ Encoders saved successfully!")
print(f"  File: {encoders_filename}")
print(f"  Size: {os.path.getsize(encoders_filename)} bytes")

# Verify model and encoders can be loaded
print("\n🔍 Verifying saved files...")
with open(model_filename, 'rb') as f:
    loaded_model = pickle.load(f)
with open(encoders_filename, 'rb') as f:
    loaded_encoders = pickle.load(f)
print("✓ Model and encoders loaded successfully from files!")

print("="*80)

# ============================================================================
# BLOCK 13: SAMPLE PREDICTIONS
# ============================================================================
print("\n[STEP 12] Test Prediction with Sample Data...")
print("-"*80)

# Test prediction with sample patient data
print("\n🔮 Making predictions with sample patient data...\n")

# Sample 1: Young female with high BP, high cholesterol
sample_1 = {
    'Age': 25,
    'Sex': 'F',
    'BP': 'HIGH',
    'Cholesterol': 'HIGH',
    'Na_to_K': 15.5
}

# Sample 2: Middle-aged male with normal BP, normal cholesterol
sample_2 = {
    'Age': 45,
    'Sex': 'M',
    'BP': 'NORMAL',
    'Cholesterol': 'NORMAL',
    'Na_to_K': 10.2
}

# Sample 3: Elderly female with low BP, high cholesterol
sample_3 = {
    'Age': 68,
    'Sex': 'F',
    'BP': 'LOW',
    'Cholesterol': 'HIGH',
    'Na_to_K': 28.5
}

samples = [sample_1, sample_2, sample_3]

for i, sample in enumerate(samples, 1):
    # Encode the sample
    sample_encoded = [
        sample['Age'],
        le_sex.transform([sample['Sex']])[0],
        le_bp.transform([sample['BP']])[0],
        le_chol.transform([sample['Cholesterol']])[0],
        sample['Na_to_K']
    ]
    
    # Make prediction
    prediction = classifier.predict([sample_encoded])
    prediction_proba = classifier.predict_proba([sample_encoded])
    predicted_drug = le_drug.inverse_transform(prediction)[0]
    confidence = prediction_proba[0][prediction[0]] * 100
    
    print(f"Sample {i}:")
    print(f"  Patient: Age={sample['Age']}, Sex={sample['Sex']}, BP={sample['BP']}, "
          f"Cholesterol={sample['Cholesterol']}, Na_to_K={sample['Na_to_K']}")
    print(f"  → Predicted Drug: {predicted_drug} (Confidence: {confidence:.1f}%)")
    print(f"  → Probability Distribution:")
    for drug, prob in zip(le_drug.classes_, prediction_proba[0]):
        print(f"      {drug}: {prob*100:.1f}%")
    print()

print("💡 Try your own prediction:")
print("   Example: classifier.predict([[30, 1, 0, 1, 12.5]])")
print("   Format: [Age, Sex(F=0/M=1), BP(HIGH=0/LOW=1/NORMAL=2), Cholesterol(HIGH=0/NORMAL=1), Na_to_K]")

print("="*80)

# ============================================================================
# BLOCK 14: SUMMARY & OUTPUTS
# ============================================================================
print("\n[STEP 13] Summary & Outputs...")
print("-"*80)

print("\n✓ Analysis Complete!")
print("\n📁 Generated Files:")
print(f"  Visualizations ({viz_dir}/):")
print("    1. drug_distribution.png")
print("    2. age_vs_na_to_k.png")
print("    3. feature_distributions.png")
print("    4. correlation_heatmap.png")
print("    5. feature_importance.png")
print("    6. decision_tree.png")
print("    7. confusion_matrix.png")
print(f"  Saved Models ({model_dir}/):")
print("    8. drug_classifier_model.pkl")
print("    9. drug_encoders.pkl")

print("\n📈 Key Findings:")
print(f"  • Dataset contains {len(df)} patient records")
print(f"  • Number of drug types: {len(le_drug.classes_)} ({', '.join(le_drug.classes_)})")
print(f"  • Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
print(f"  • Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"  • Most important feature: {feature_names[np.argmax(feature_importance)]}")

print("\n🚀 Next Steps:")
print("  1. Load saved model:")
print("     model = pickle.load(open('models/drug_classifier_model.pkl', 'rb'))")
print("     encoders = pickle.load(open('models/drug_encoders.pkl', 'rb'))")
print("  2. Make predictions with new patient data")
print("  3. Try different classifiers (Random Forest, SVM, etc.)")
print("  4. Experiment with hyperparameter tuning")
print("  5. Apply cross-validation for robust evaluation")

print("\n" + "="*80)
print("✓ ANALYSIS COMPLETED SUCCESSFULLY!")
print("✓ Model saved and ready for deployment!")
print("="*80)
