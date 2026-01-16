"""
Drug Prediction - Interactive Block Execution
=============================================
Execute individual blocks or groups of blocks interactively.
Adapted from CO2 Prediction structure for Drug Identification dataset.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Global variables to store state across blocks
df = None
df_processed = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
classifier = None
y_pred = None
y_pred_proba = None
train_accuracy = None
test_accuracy = None
le_sex = None
le_bp = None
le_chol = None
le_drug = None
feature_names = ['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']
viz_dir = "visualizations"
model_dir = "models"

# =======================================================================
# BLOCK FUNCTIONS
# =======================================================================

def block_1_import_libraries():
    """Block 1: Import Libraries"""
    global np, pd, plt, sns, train_test_split, DecisionTreeClassifier, plot_tree
    global LabelEncoder, accuracy_score, classification_report, confusion_matrix, pickle
    
    print("\n" + "="*80)
    print("BLOCK 1: Import Libraries")
    print("="*80)
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.tree import DecisionTreeClassifier, plot_tree
        from sklearn.preprocessing import LabelEncoder
        from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
        import pickle
        
        print("✓ NumPy imported successfully")
        print("✓ Pandas imported successfully")
        print("✓ Matplotlib imported successfully")
        print("✓ Seaborn imported successfully")
        print("✓ Scikit-learn imported successfully")
        print("✓ Pickle imported successfully")
        print("\n✓ All libraries imported successfully!")
        return True
    except ImportError as e:
        print(f"✗ Error importing libraries: {e}")
        print("Run: pip install numpy pandas matplotlib seaborn scikit-learn")
        return False

def block_2_load_dataset():
    """Block 2: Load Dataset"""
    global df
    
    print("\n" + "="*80)
    print("BLOCK 2: Load Dataset")
    print("="*80)
    
    DATA_FILE = "drug_identification.csv"
    
    if os.path.exists(DATA_FILE):
        data_path = DATA_FILE
    else:
        data_path = r"C:\Users\shujare\OneDrive - Capgemini\Attachments\SIMS\SEM4\ML\ML_A1_24020448074\drug_identification.csv"
    
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
    
    print("\n📊 Target Variable Distribution:")
    print(df['Drug'].value_counts())
    
    print("\n📊 Categorical Features:")
    print("\nSex:", df['Sex'].value_counts().to_dict())
    print("BP:", df['BP'].value_counts().to_dict())
    print("Cholesterol:", df['Cholesterol'].value_counts().to_dict())
    
    return True

def block_4_visualize_data():
    """Block 4: Data Visualization"""
    global df, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 4: Data Visualization")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"✓ Created directory: {viz_dir}/")
    
    # Drug distribution
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
    
    # Age vs Na_to_K scatter plot
    print("\n📈 Creating scatter plot...")
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
    
    # Feature distributions
    print("\n📊 Creating feature distributions...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    axes[0, 0].hist(df['Age'], bins=20, edgecolor='black', color='skyblue')
    axes[0, 0].set_xlabel('Age', fontweight='bold')
    axes[0, 0].set_ylabel('Frequency', fontweight='bold')
    axes[0, 0].set_title('Age Distribution', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    
    axes[0, 1].hist(df['Na_to_K'], bins=20, edgecolor='black', color='lightgreen')
    axes[0, 1].set_xlabel('Na to K Ratio', fontweight='bold')
    axes[0, 1].set_ylabel('Frequency', fontweight='bold')
    axes[0, 1].set_title('Na_to_K Distribution', fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    
    sex_counts = df['Sex'].value_counts()
    axes[1, 0].bar(sex_counts.index, sex_counts.values, color=['lightcoral', 'lightblue'])
    axes[1, 0].set_xlabel('Sex', fontweight='bold')
    axes[1, 0].set_ylabel('Count', fontweight='bold')
    axes[1, 0].set_title('Sex Distribution', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
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
    
    # Correlation heatmap
    print("\n📊 Creating correlation heatmap...")
    plt.figure(figsize=(8, 6))
    numerical_cols = ['Age', 'Na_to_K']
    corr_matrix = df[numerical_cols].corr()
    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, square=True, linewidths=1)
    plt.title('Correlation Heatmap', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/correlation_heatmap.png")
    plt.show()
    
    print("\n✓ All visualizations completed!")
    return True

def block_5_preprocess_data():
    """Block 5: Data Preprocessing & Encoding"""
    global df, df_processed, le_sex, le_bp, le_chol, le_drug
    
    print("\n" + "="*80)
    print("BLOCK 5: Data Preprocessing & Encoding")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    # Create copy
    df_processed = df.copy()
    
    # Initialize encoders
    le_sex = LabelEncoder()
    le_bp = LabelEncoder()
    le_chol = LabelEncoder()
    le_drug = LabelEncoder()
    
    # Encode features
    print("\n🔄 Encoding categorical features...")
    df_processed['Sex'] = le_sex.fit_transform(df['Sex'])
    df_processed['BP'] = le_bp.fit_transform(df['BP'])
    df_processed['Cholesterol'] = le_chol.fit_transform(df['Cholesterol'])
    df_processed['Drug'] = le_drug.fit_transform(df['Drug'])
    
    print(f"✓ Sex: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    print(f"✓ BP: {dict(zip(le_bp.classes_, le_bp.transform(le_bp.classes_)))}")
    print(f"✓ Cholesterol: {dict(zip(le_chol.classes_, le_chol.transform(le_chol.classes_)))}")
    print(f"✓ Drug: {dict(zip(le_drug.classes_, le_drug.transform(le_drug.classes_)))}")
    
    print("\nSample of encoded data:")
    print(df_processed.head())
    
    return True

def block_6_prepare_data():
    """Block 6: Prepare Features and Target"""
    global df_processed, X, y, le_drug
    
    print("\n" + "="*80)
    print("BLOCK 6: Prepare Features and Target")
    print("="*80)
    
    if df_processed is None:
        print("✗ Error: Data not preprocessed. Run Block 5 first.")
        return False
    
    X = df_processed[['Age', 'Sex', 'BP', 'Cholesterol', 'Na_to_K']].values
    y = df_processed['Drug'].values
    
    print(f"\n✓ Features (X): {X.shape}")
    print(f"  Feature names: Age, Sex, BP, Cholesterol, Na_to_K")
    print(f"✓ Target (y): {y.shape}")
    print(f"  Classes: {le_drug.classes_}")
    
    return True

def block_7_split_data():
    """Block 7: Split Data into Train/Test"""
    global X, y, X_train, X_test, y_train, y_test, le_drug
    
    print("\n" + "="*80)
    print("BLOCK 7: Split Data into Train/Test Sets")
    print("="*80)
    
    if X is None or y is None:
        print("✗ Error: Data not prepared. Run Block 6 first.")
        return False
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\n📊 Data Split:")
    print(f"  Training: {X_train.shape}, Target: {y_train.shape}")
    print(f"  Testing: {X_test.shape}, Target: {y_test.shape}")
    print(f"  Ratio: 80/20 with stratification")
    
    print("\n📊 Train set class distribution:")
    unique, counts = np.unique(y_train, return_counts=True)
    for drug_id, count in zip(unique, counts):
        print(f"  {le_drug.inverse_transform([drug_id])[0]}: {count}")
    
    return True

def block_8_train_model():
    """Block 8: Train Decision Tree Classifier"""
    global X_train, y_train, classifier, feature_names, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 8: Train Decision Tree Classifier")
    print("="*80)
    
    if X_train is None or y_train is None:
        print("✗ Error: Data not split. Run Block 7 first.")
        return False
    
    classifier = DecisionTreeClassifier(
        criterion='entropy',
        max_depth=4,
        random_state=42
    )
    classifier.fit(X_train, y_train)
    
    print(f"\n✓ Model trained successfully!")
    print(f"  Criterion: {classifier.criterion}")
    print(f"  Max depth: {classifier.max_depth}")
    print(f"  Leaves: {classifier.get_n_leaves()}")
    print(f"  Tree depth: {classifier.get_depth()}")
    
    # Feature importance
    feature_importance = classifier.feature_importances_
    print("\n📊 Feature Importance:")
    for name, imp in sorted(zip(feature_names, feature_importance), key=lambda x: x[1], reverse=True):
        print(f"  {name:15s}: {imp:.4f}")
    
    # Visualize importance
    plt.figure(figsize=(10, 6))
    sorted_idx = np.argsort(feature_importance)
    plt.barh(range(len(sorted_idx)), feature_importance[sorted_idx], color='skyblue')
    plt.yticks(range(len(sorted_idx)), [feature_names[i] for i in sorted_idx])
    plt.xlabel('Importance', fontsize=12, fontweight='bold')
    plt.ylabel('Features', fontsize=12, fontweight='bold')
    plt.title('Feature Importance', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/feature_importance.png', dpi=300, bbox_inches='tight')
    print(f"\n  ✓ Saved: {viz_dir}/feature_importance.png")
    plt.show()
    
    return True

def block_9_visualize_tree():
    """Block 9: Visualize Decision Tree"""
    global classifier, feature_names, le_drug, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 9: Visualize Decision Tree")
    print("="*80)
    
    if classifier is None:
        print("✗ Error: Model not trained. Run Block 8 first.")
        return False
    
    print("\n📊 Creating decision tree visualization...")
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
    
    return True

def block_10_make_predictions():
    """Block 10: Make Predictions"""
    global classifier, X_test, y_test, y_pred, y_pred_proba, le_drug
    
    print("\n" + "="*80)
    print("BLOCK 10: Make Predictions on Test Set")
    print("="*80)
    
    if classifier is None or X_test is None:
        print("✗ Error: Model not trained or data not split. Run Blocks 7 & 8.")
        return False
    
    y_pred = classifier.predict(X_test)
    y_pred_proba = classifier.predict_proba(X_test)
    
    print("\n✓ Predictions completed!")
    print(f"  Sample predictions (first 5):")
    for i in range(min(5, len(y_pred))):
        actual = le_drug.inverse_transform([y_test[i]])[0]
        predicted = le_drug.inverse_transform([y_pred[i]])[0]
        confidence = y_pred_proba[i][y_pred[i]] * 100
        print(f"    Actual: {actual}  →  Predicted: {predicted} ({confidence:.1f}%)")
    
    return True

def block_11_evaluate_model():
    """Block 11: Evaluate Model"""
    global classifier, X_train, y_train, X_test, y_test, y_pred
    global train_accuracy, test_accuracy, le_drug, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 11: Evaluate Model Performance")
    print("="*80)
    
    if classifier is None or y_pred is None:
        print("✗ Error: Predictions not made. Run Block 10 first.")
        return False
    
    train_accuracy = classifier.score(X_train, y_train)
    test_accuracy = accuracy_score(y_test, y_pred)
    
    print(f"\n📊 Model Performance:")
    print(f"  Training Accuracy: {train_accuracy:.4f} ({train_accuracy*100:.2f}%)")
    print(f"  Testing Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
    
    print(f"\n💡 Performance: ", end="")
    if test_accuracy > 0.9:
        print("Excellent!")
    elif test_accuracy > 0.8:
        print("Good!")
    elif test_accuracy > 0.7:
        print("Moderate")
    else:
        print("Needs improvement")
    
    print("\n📋 Classification Report:")
    print(classification_report(y_test, y_pred, target_names=le_drug.classes_))
    
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
    
    return True

def block_12_save_model():
    """Block 12: Save Model and Encoders"""
    global classifier, le_sex, le_bp, le_chol, le_drug, model_dir
    
    print("\n" + "="*80)
    print("BLOCK 12: Save Model and Encoders")
    print("="*80)
    
    if classifier is None:
        print("✗ Error: Model not trained. Run Block 8 first.")
        return False
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save model
    model_file = f'{model_dir}/drug_classifier_model.pkl'
    with open(model_file, 'wb') as f:
        pickle.dump(classifier, f)
    print(f"✓ Model saved: {model_file}")
    print(f"  Size: {os.path.getsize(model_file)} bytes")
    
    # Save encoders
    encoders = {
        'le_sex': le_sex,
        'le_bp': le_bp,
        'le_chol': le_chol,
        'le_drug': le_drug
    }
    encoders_file = f'{model_dir}/drug_encoders.pkl'
    with open(encoders_file, 'wb') as f:
        pickle.dump(encoders, f)
    print(f"✓ Encoders saved: {encoders_file}")
    print(f"  Size: {os.path.getsize(encoders_file)} bytes")
    
    return True

def block_13_test_predictions():
    """Block 13: Test with Sample Data"""
    global classifier, le_sex, le_bp, le_chol, le_drug, train_accuracy, test_accuracy
    global X_train, y_train, X_test, y_test, y_pred
    
    print("\n" + "="*80)
    print("BLOCK 13: Test Predictions with Sample Data")
    print("="*80)
    
    if classifier is None or le_sex is None:
        print("✗ Error: Model/encoders not ready. Run Blocks 5 & 8.")
        return False
    
    # Display Model Performance Metrics
    print("\n📊 MODEL PERFORMANCE SUMMARY")
    print("="*80)
    
    if train_accuracy is not None and test_accuracy is not None and y_pred is not None:
        # Calculate per-class metrics
        from sklearn.metrics import precision_score, recall_score, f1_score
        
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        print("\n┌─────────────────────────────────────────────────────────────────────────────┐")
        print("│                      CLASSIFICATION METRICS                                 │")
        print("├──────────────────┬──────────────┬──────────────┬──────────────┬────────────┤")
        print("│ Metric           │   Train      │    Test      │  Precision   │   Recall   │")
        print("├──────────────────┼──────────────┼──────────────┼──────────────┼────────────┤")
        print(f"│ Accuracy         │   {train_accuracy:.4f}     │   {test_accuracy:.4f}     │   {precision:.4f}     │  {recall:.4f}  │")
        print(f"│ F1-Score         │      -       │      -       │       -      │  {f1:.4f}  │")
        print("└──────────────────┴──────────────┴──────────────┴──────────────┴────────────┘")
        
        print(f"\n📈 Model Type: Decision Tree Classifier")
        print(f"   Training Accuracy: {train_accuracy*100:.2f}%")
        print(f"   Testing Accuracy:  {test_accuracy*100:.2f}%")
        print(f"   Weighted Precision: {precision*100:.2f}%")
        print(f"   Weighted Recall:    {recall*100:.2f}%")
        print(f"   Weighted F1-Score:  {f1*100:.2f}%")
        
        # Note about regression metrics
        print("\n💡 Note: MAE, MSE, RMSE, and R² are regression metrics.")
        print("   This is a classification task, so we use Accuracy, Precision, Recall, and F1-Score.")
        
    else:
        print("⚠ Model metrics not available. Run Blocks 7-11 first to see complete metrics.")
    
    print("\n" + "="*80)
    print("\n🔮 Sample Predictions:\n")
    
    samples = [
        {'Age': 25, 'Sex': 'F', 'BP': 'HIGH', 'Cholesterol': 'HIGH', 'Na_to_K': 15.5},
        {'Age': 45, 'Sex': 'M', 'BP': 'NORMAL', 'Cholesterol': 'NORMAL', 'Na_to_K': 10.2},
        {'Age': 68, 'Sex': 'F', 'BP': 'LOW', 'Cholesterol': 'HIGH', 'Na_to_K': 28.5}
    ]
    
    for i, sample in enumerate(samples, 1):
        sample_encoded = [
            sample['Age'],
            le_sex.transform([sample['Sex']])[0],
            le_bp.transform([sample['BP']])[0],
            le_chol.transform([sample['Cholesterol']])[0],
            sample['Na_to_K']
        ]
        
        pred = classifier.predict([sample_encoded])
        pred_proba = classifier.predict_proba([sample_encoded])
        predicted_drug = le_drug.inverse_transform(pred)[0]
        confidence = pred_proba[0][pred[0]] * 100
        
        print(f"Sample {i}:")
        print(f"  Patient: Age={sample['Age']}, Sex={sample['Sex']}, BP={sample['BP']}, "
              f"Chol={sample['Cholesterol']}, Na_to_K={sample['Na_to_K']}")
        print(f"  → Predicted: {predicted_drug} ({confidence:.1f}%)")
        print()
    
    return True

# ============================================================================
# MENU SYSTEM
# ============================================================================

BLOCKS = {
    1: ("Import Libraries", block_1_import_libraries),
    2: ("Load Dataset", block_2_load_dataset),
    3: ("Explore Data", block_3_explore_data),
    4: ("Visualize Data", block_4_visualize_data),
    5: ("Preprocess & Encode Data", block_5_preprocess_data),
    6: ("Prepare Features & Target", block_6_prepare_data),
    7: ("Split Train/Test Data", block_7_split_data),
    8: ("Train Decision Tree", block_8_train_model),
    9: ("Visualize Decision Tree", block_9_visualize_tree),
    10: ("Make Predictions", block_10_make_predictions),
    11: ("Evaluate Model", block_11_evaluate_model),
    12: ("Save Model & Encoders", block_12_save_model),
    13: ("Test Sample Predictions", block_13_test_predictions),
}

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*80)
    print("DRUG PREDICTION - BLOCK EXECUTION MENU")
    print("="*80)
    
    for num, (name, _) in BLOCKS.items():
        print(f"{num:2d}. {name}")
    
    print("\n14. Run All Blocks")
    print(" 0. Exit")
    print("="*80)

def parse_input(user_input):
    """Parse user input (e.g., '1', '1,2,3', '1-5')"""
    blocks_to_run = []
    
    parts = user_input.replace(' ', '').split(',')
    
    for part in parts:
        if '-' in part:
            try:
                start, end = map(int, part.split('-'))
                blocks_to_run.extend(range(start, end + 1))
            except:
                print(f"Invalid range: {part}")
        else:
            try:
                blocks_to_run.append(int(part))
            except:
                print(f"Invalid block: {part}")
    
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
            print(f"\n✗ Block {num} failed. Stopping.")
            return False
    
    print("\n" + "="*80)
    print("✓ ALL BLOCKS COMPLETED SUCCESSFULLY!")
    print("="*80)
    return True

def main():
    """Main interactive loop"""
    print("\n" + "="*80)
    print("DRUG PREDICTION - INTERACTIVE MODE")
    print("="*80)
    print("Execute individual blocks or groups of blocks.")
    print("Each block performs a specific task in the ML pipeline.")
    print("="*80)
    
    while True:
        show_menu()
        
        try:
            user_input = input("\nEnter block number(s) (e.g., 1 or 1,2,3 or 1-5): ").strip()
            
            if user_input == '0':
                print("\n✓ Exiting. Goodbye!")
                break
            
            if user_input == '14':
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
                    print(f"✗ Invalid block: {block_num}")
            
            print("\n" + "="*80)
            print("✓ Execution completed!")
            print("="*80)
            
        except KeyboardInterrupt:
            print("\n\n✗ Interrupted. Exiting.")
            break
        except Exception as e:
            print(f"\n✗ Error: {e}")

if __name__ == "__main__":
    main()
