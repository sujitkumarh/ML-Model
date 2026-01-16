"""
Credit Card Fraud Detection - Interactive Block Execution
==========================================================
Execute individual blocks or groups of blocks interactively.
Uses Decision Tree and SVM models for fraud detection.
"""

import os
import sys
import warnings
warnings.filterwarnings('ignore')

# Global variables to store state across blocks
df = None
X = None
y = None
X_train = None
X_test = None
y_train = None
y_test = None
X_train_scaled = None
X_test_scaled = None
scaler = None

# Decision Tree Model variables
dt_classifier = None
y_pred_dt = None
accuracy_dt = None
precision_dt = None
recall_dt = None
f1_dt = None
confusion_dt = None
classification_report_dt = None

# SVM Model variables
svm_classifier = None
y_pred_svm = None
accuracy_svm = None
precision_svm = None
recall_svm = None
f1_svm = None
confusion_svm = None
classification_report_svm = None

viz_dir = "visualizations"
model_dir = "models"

# ============================================================================
# BLOCK FUNCTIONS
# ============================================================================

def block_1_import_libraries():
    """Block 1: Import Libraries"""
    global np, pd, plt, sns
    global train_test_split, StandardScaler
    global DecisionTreeClassifier, SVC
    global accuracy_score, precision_score, recall_score, f1_score
    global confusion_matrix, classification_report
    global pickle
    
    print("\n" + "="*80)
    print("BLOCK 1: Import Libraries")
    print("="*80)
    
    try:
        import numpy as np
        import pandas as pd
        import matplotlib.pyplot as plt
        import seaborn as sns
        from sklearn.model_selection import train_test_split
        from sklearn.preprocessing import StandardScaler
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.svm import SVC
        from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                                     f1_score, confusion_matrix, classification_report)
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
    
    DATA_FILE = "creditcard.csv"
    
    if os.path.exists(DATA_FILE):
        data_path = DATA_FILE
    else:
        data_path = r"C:\Users\shujare\OneDrive - Capgemini\Attachments\SIMS\SEM4\ML\ML_A1\creditcard.csv"
    
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
    
    print("\n📋 Class Distribution:")
    class_counts = df['Class'].value_counts()
    print(class_counts)
    print(f"\n  Normal Transactions (Class 0): {class_counts[0]} ({class_counts[0]/len(df)*100:.2f}%)")
    print(f"  Fraud Transactions (Class 1): {class_counts[1]} ({class_counts[1]/len(df)*100:.2f}%)")
    
    print("\n📋 Sample Data (5 random records):")
    print(df.sample(5))
    
    return True

def block_4_prepare_data():
    """Block 4: Data Preparation"""
    global df, X, y
    
    print("\n" + "="*80)
    print("BLOCK 4: Data Preparation")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    # Separate features and target
    X = df.drop('Class', axis=1)
    y = df['Class']
    
    print(f"✓ Features (X) shape: {X.shape}")
    print(f"✓ Target (y) shape: {y.shape}")
    print(f"\n  Features: {list(X.columns)}")
    print(f"\n  Target distribution:")
    print(f"    Class 0 (Normal): {sum(y == 0)}")
    print(f"    Class 1 (Fraud): {sum(y == 1)}")
    
    return True

def block_5_visualize_data():
    """Block 5: Data Visualization"""
    global df, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 5: Data Visualization")
    print("="*80)
    
    if df is None:
        print("✗ Error: Dataset not loaded. Run Block 2 first.")
        return False
    
    if not os.path.exists(viz_dir):
        os.makedirs(viz_dir)
        print(f"✓ Created directory: {viz_dir}/")
    
    # Class distribution pie chart
    print("\n📊 Creating class distribution visualization...")
    plt.figure(figsize=(10, 6))
    class_counts = df['Class'].value_counts()
    colors = ['#66b3ff', '#ff6666']
    explode = (0.1, 0)
    plt.pie(class_counts, labels=['Normal', 'Fraud'], autopct='%1.2f%%', 
            colors=colors, explode=explode, shadow=True, startangle=90)
    plt.title('Credit Card Transaction Distribution', fontsize=16, fontweight='bold')
    plt.savefig(f'{viz_dir}/class_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/class_distribution.png")
    plt.show()
    
    # Amount distribution
    print("\n📈 Creating transaction amount distribution...")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Normal transactions
    axes[0].hist(df[df['Class'] == 0]['Amount'], bins=50, color='blue', alpha=0.7, edgecolor='black')
    axes[0].set_title('Normal Transactions - Amount Distribution', fontweight='bold')
    axes[0].set_xlabel('Amount')
    axes[0].set_ylabel('Frequency')
    axes[0].grid(True, alpha=0.3)
    
    # Fraud transactions
    axes[1].hist(df[df['Class'] == 1]['Amount'], bins=50, color='red', alpha=0.7, edgecolor='black')
    axes[1].set_title('Fraud Transactions - Amount Distribution', fontweight='bold')
    axes[1].set_xlabel('Amount')
    axes[1].set_ylabel('Frequency')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/amount_distribution.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/amount_distribution.png")
    plt.show()
    
    # Time distribution
    print("\n📉 Creating time distribution visualization...")
    plt.figure(figsize=(12, 6))
    plt.scatter(df[df['Class'] == 0]['Time'], df[df['Class'] == 0]['Amount'], 
                alpha=0.5, s=1, label='Normal', color='blue')
    plt.scatter(df[df['Class'] == 1]['Time'], df[df['Class'] == 1]['Amount'], 
                alpha=0.8, s=3, label='Fraud', color='red')
    plt.title('Transaction Time vs Amount', fontsize=14, fontweight='bold')
    plt.xlabel('Time (seconds)', fontweight='bold')
    plt.ylabel('Amount', fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/time_amount_scatter.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/time_amount_scatter.png")
    plt.show()
    
    # Correlation heatmap (sample of features)
    print("\n🔥 Creating correlation heatmap...")
    plt.figure(figsize=(12, 8))
    correlation_features = ['Time', 'Amount', 'V1', 'V2', 'V3', 'V4', 'V5', 'Class']
    correlation_matrix = df[correlation_features].corr()
    sns.heatmap(correlation_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
                center=0, square=True, linewidths=1)
    plt.title('Feature Correlation Heatmap (Sample)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/correlation_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/correlation_heatmap.png")
    plt.show()
    
    print("\n✓ All visualizations completed!")
    return True

def block_6_split_data():
    """Block 6: Split Data"""
    global X, y, X_train, X_test, y_train, y_test
    
    print("\n" + "="*80)
    print("BLOCK 6: Split Data into Training and Testing Sets")
    print("="*80)
    
    if X is None or y is None:
        print("✗ Error: Data not prepared. Run Block 4 first.")
        return False
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"✓ Data split successfully!")
    print(f"\n  Training set:")
    print(f"    X_train shape: {X_train.shape}")
    print(f"    y_train shape: {y_train.shape}")
    print(f"    Normal: {sum(y_train == 0)}, Fraud: {sum(y_train == 1)}")
    
    print(f"\n  Testing set:")
    print(f"    X_test shape: {X_test.shape}")
    print(f"    y_test shape: {y_test.shape}")
    print(f"    Normal: {sum(y_test == 0)}, Fraud: {sum(y_test == 1)}")
    
    return True

def block_7_scale_features():
    """Block 7: Feature Scaling"""
    global X_train, X_test, X_train_scaled, X_test_scaled, scaler
    
    print("\n" + "="*80)
    print("BLOCK 7: Feature Scaling (for SVM)")
    print("="*80)
    
    if X_train is None or X_test is None:
        print("✗ Error: Data not split. Run Block 6 first.")
        return False
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"✓ Features scaled successfully!")
    print(f"  X_train_scaled shape: {X_train_scaled.shape}")
    print(f"  X_test_scaled shape: {X_test_scaled.shape}")
    print(f"\n  Note: Scaling is important for SVM but not required for Decision Trees")
    
    return True

def block_8_train_decision_tree():
    """Block 8: Train Decision Tree Model"""
    global X_train, y_train, dt_classifier
    
    print("\n" + "="*80)
    print("BLOCK 8: Train Decision Tree Classifier")
    print("="*80)
    
    if X_train is None or y_train is None:
        print("✗ Error: Data not split. Run Block 6 first.")
        return False
    
    print("🌳 Training Decision Tree model...")
    dt_classifier = DecisionTreeClassifier(
        criterion='gini',
        max_depth=10,
        min_samples_split=10,
        min_samples_leaf=5,
        random_state=42
    )
    
    dt_classifier.fit(X_train, y_train)
    
    print(f"\n✓ Decision Tree model trained successfully!")
    print(f"  Max depth: {dt_classifier.max_depth}")
    print(f"  Number of features: {dt_classifier.n_features_in_}")
    print(f"  Number of classes: {dt_classifier.n_classes_}")
    
    return True

def block_9_evaluate_decision_tree():
    """Block 9: Evaluate Decision Tree Model"""
    global dt_classifier, X_test, y_test, y_pred_dt
    global accuracy_dt, precision_dt, recall_dt, f1_dt
    global confusion_dt, classification_report_dt, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 9: Evaluate Decision Tree Model")
    print("="*80)
    
    if dt_classifier is None:
        print("✗ Error: Decision Tree not trained. Run Block 8 first.")
        return False
    
    # Predictions
    y_pred_dt = dt_classifier.predict(X_test)
    
    # Calculate metrics
    accuracy_dt = accuracy_score(y_test, y_pred_dt)
    precision_dt = precision_score(y_test, y_pred_dt)
    recall_dt = recall_score(y_test, y_pred_dt)
    f1_dt = f1_score(y_test, y_pred_dt)
    confusion_dt = confusion_matrix(y_test, y_pred_dt)
    classification_report_dt = classification_report(y_test, y_pred_dt)
    
    print(f"\n📊 Decision Tree Performance:")
    print(f"  Accuracy:  {accuracy_dt:.4f} ({accuracy_dt*100:.2f}%)")
    print(f"  Precision: {precision_dt:.4f}")
    print(f"  Recall:    {recall_dt:.4f}")
    print(f"  F1-Score:  {f1_dt:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(confusion_dt)
    print(f"\n  True Negatives:  {confusion_dt[0][0]}")
    print(f"  False Positives: {confusion_dt[0][1]}")
    print(f"  False Negatives: {confusion_dt[1][0]}")
    print(f"  True Positives:  {confusion_dt[1][1]}")
    
    print(f"\n📄 Classification Report:")
    print(classification_report_dt)
    
    # Visualize confusion matrix
    print("\n📊 Creating confusion matrix visualization...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_dt, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('Decision Tree - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/dt_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/dt_confusion_matrix.png")
    plt.show()
    
    return True

def block_10_save_decision_tree():
    """Block 10: Save Decision Tree Model"""
    global dt_classifier, model_dir
    
    print("\n" + "="*80)
    print("BLOCK 10: Save Decision Tree Model")
    print("="*80)
    
    if dt_classifier is None:
        print("✗ Error: Decision Tree not trained. Run Block 8 first.")
        return False
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    filename = f'{model_dir}/fraud_detection_dt_model.pkl'
    with open(filename, 'wb') as f:
        pickle.dump(dt_classifier, f)
    
    print(f"✓ Decision Tree model saved: {filename}")
    print(f"  Size: {os.path.getsize(filename)} bytes")
    
    return True

def block_11_train_svm():
    """Block 11: Train SVM Model"""
    global X_train_scaled, y_train, svm_classifier
    
    print("\n" + "="*80)
    print("BLOCK 11: Train SVM Classifier")
    print("="*80)
    
    if X_train_scaled is None or y_train is None:
        print("✗ Error: Data not scaled. Run Block 7 first.")
        return False
    
    print("🤖 Training SVM model (this may take a few minutes)...")
    print("  Using RBF kernel with balanced class weights...")
    
    svm_classifier = SVC(
        kernel='rbf',
        C=1.0,
        gamma='scale',
        class_weight='balanced',
        random_state=42
    )
    
    svm_classifier.fit(X_train_scaled, y_train)
    
    print(f"\n✓ SVM model trained successfully!")
    print(f"  Kernel: {svm_classifier.kernel}")
    print(f"  Number of support vectors: {svm_classifier.n_support_}")
    print(f"  Support vectors per class: {list(svm_classifier.n_support_)}")
    
    return True

def block_12_evaluate_svm():
    """Block 12: Evaluate SVM Model"""
    global svm_classifier, X_test_scaled, y_test, y_pred_svm
    global accuracy_svm, precision_svm, recall_svm, f1_svm
    global confusion_svm, classification_report_svm, viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 12: Evaluate SVM Model")
    print("="*80)
    
    if svm_classifier is None:
        print("✗ Error: SVM not trained. Run Block 11 first.")
        return False
    
    # Predictions
    print("🔮 Making predictions...")
    y_pred_svm = svm_classifier.predict(X_test_scaled)
    
    # Calculate metrics
    accuracy_svm = accuracy_score(y_test, y_pred_svm)
    precision_svm = precision_score(y_test, y_pred_svm)
    recall_svm = recall_score(y_test, y_pred_svm)
    f1_svm = f1_score(y_test, y_pred_svm)
    confusion_svm = confusion_matrix(y_test, y_pred_svm)
    classification_report_svm = classification_report(y_test, y_pred_svm)
    
    print(f"\n📊 SVM Performance:")
    print(f"  Accuracy:  {accuracy_svm:.4f} ({accuracy_svm*100:.2f}%)")
    print(f"  Precision: {precision_svm:.4f}")
    print(f"  Recall:    {recall_svm:.4f}")
    print(f"  F1-Score:  {f1_svm:.4f}")
    
    print(f"\n📋 Confusion Matrix:")
    print(confusion_svm)
    print(f"\n  True Negatives:  {confusion_svm[0][0]}")
    print(f"  False Positives: {confusion_svm[0][1]}")
    print(f"  False Negatives: {confusion_svm[1][0]}")
    print(f"  True Positives:  {confusion_svm[1][1]}")
    
    print(f"\n📄 Classification Report:")
    print(classification_report_svm)
    
    # Visualize confusion matrix
    print("\n📊 Creating confusion matrix visualization...")
    plt.figure(figsize=(8, 6))
    sns.heatmap(confusion_svm, annot=True, fmt='d', cmap='Greens', 
                xticklabels=['Normal', 'Fraud'], 
                yticklabels=['Normal', 'Fraud'])
    plt.title('SVM - Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('Actual', fontweight='bold')
    plt.xlabel('Predicted', fontweight='bold')
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/svm_confusion_matrix.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/svm_confusion_matrix.png")
    plt.show()
    
    return True

def block_13_save_svm():
    """Block 13: Save SVM Model and Scaler"""
    global svm_classifier, scaler, model_dir
    
    print("\n" + "="*80)
    print("BLOCK 13: Save SVM Model and Scaler")
    print("="*80)
    
    if svm_classifier is None:
        print("✗ Error: SVM not trained. Run Block 11 first.")
        return False
    
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # Save SVM model
    svm_filename = f'{model_dir}/fraud_detection_svm_model.pkl'
    with open(svm_filename, 'wb') as f:
        pickle.dump(svm_classifier, f)
    
    # Save scaler
    scaler_filename = f'{model_dir}/fraud_detection_scaler.pkl'
    with open(scaler_filename, 'wb') as f:
        pickle.dump(scaler, f)
    
    print(f"✓ SVM model saved: {svm_filename}")
    print(f"  Size: {os.path.getsize(svm_filename)} bytes")
    print(f"\n✓ Scaler saved: {scaler_filename}")
    print(f"  Size: {os.path.getsize(scaler_filename)} bytes")
    
    return True

def block_14_compare_models():
    """Block 14: Compare Models"""
    global accuracy_dt, precision_dt, recall_dt, f1_dt
    global accuracy_svm, precision_svm, recall_svm, f1_svm
    global viz_dir
    
    print("\n" + "="*80)
    print("BLOCK 14: Model Comparison")
    print("="*80)
    
    if accuracy_dt is None or accuracy_svm is None:
        print("✗ Error: Both models must be evaluated first.")
        print("  Run Blocks 9 and 12.")
        return False
    
    comparison_data = {
        'Model': ['Decision Tree', 'SVM'],
        'Accuracy': [accuracy_dt, accuracy_svm],
        'Precision': [precision_dt, precision_svm],
        'Recall': [recall_dt, recall_svm],
        'F1-Score': [f1_dt, f1_svm]
    }
    
    comparison_df = pd.DataFrame(comparison_data)
    print("\n📊 Model Performance Comparison:")
    print(comparison_df.to_string(index=False))
    
    # Determine best model
    best_by_accuracy = 'Decision Tree' if accuracy_dt > accuracy_svm else 'SVM'
    best_by_f1 = 'Decision Tree' if f1_dt > f1_svm else 'SVM'
    best_by_recall = 'Decision Tree' if recall_dt > recall_svm else 'SVM'
    
    print(f"\n🏆 Best Model by Accuracy: {best_by_accuracy} ({max(accuracy_dt, accuracy_svm):.4f})")
    print(f"🏆 Best Model by F1-Score: {best_by_f1} ({max(f1_dt, f1_svm):.4f})")
    print(f"🏆 Best Model by Recall: {best_by_recall} ({max(recall_dt, recall_svm):.4f})")
    
    print(f"\n💡 Note: For fraud detection, Recall is crucial!")
    print(f"   High recall means we catch more fraud cases (fewer false negatives)")
    
    # Visualization
    print("\n📊 Creating comparison bar chart...")
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    dt_scores = [accuracy_dt, precision_dt, recall_dt, f1_dt]
    svm_scores = [accuracy_svm, precision_svm, recall_svm, f1_svm]
    
    x = np.arange(len(metrics))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, dt_scores, width, label='Decision Tree', color='skyblue')
    bars2 = ax.bar(x + width/2, svm_scores, width, label='SVM', color='lightgreen')
    
    ax.set_xlabel('Metrics', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score', fontsize=12, fontweight='bold')
    ax.set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics)
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 1.1)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.3f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(f'{viz_dir}/model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"  ✓ Saved: {viz_dir}/model_comparison.png")
    plt.show()
    
    return True

def block_15_test_predictions():
    """Block 15: Test Sample Predictions"""
    global dt_classifier, svm_classifier, scaler, X_test, y_test
    
    print("\n" + "="*80)
    print("BLOCK 15: Sample Predictions on Test Data")
    print("="*80)
    
    if dt_classifier is None or svm_classifier is None:
        print("✗ Error: Models not trained. Run Blocks 8 and 11 first.")
        return False
    
    # Get 5 random samples from test set
    sample_indices = np.random.choice(X_test.index, 5, replace=False)
    samples = X_test.loc[sample_indices]
    actual_labels = y_test.loc[sample_indices]
    
    print("\n🔮 Sample Predictions:")
    print("="*80)
    
    for i, idx in enumerate(sample_indices, 1):
        sample = samples.loc[idx:idx]
        actual = actual_labels.loc[idx]
        
        # Decision Tree prediction
        dt_pred = dt_classifier.predict(sample)[0]
        
        # SVM prediction (need to scale)
        sample_scaled = scaler.transform(sample)
        svm_pred = svm_classifier.predict(sample_scaled)[0]
        
        print(f"\nSample {i}:")
        print(f"  Actual: {'FRAUD' if actual == 1 else 'Normal'}")
        print(f"  Decision Tree predicts: {'FRAUD' if dt_pred == 1 else 'Normal'} {'✓' if dt_pred == actual else '✗'}")
        print(f"  SVM predicts: {'FRAUD' if svm_pred == 1 else 'Normal'} {'✓' if svm_pred == actual else '✗'}")
    
    return True

# ============================================================================
# MENU SYSTEM
# ============================================================================

BLOCKS = {
    1: ("Import Libraries", block_1_import_libraries),
    2: ("Load Dataset", block_2_load_dataset),
    3: ("Explore Data", block_3_explore_data),
    4: ("Prepare Data", block_4_prepare_data),
    5: ("Visualize Data", block_5_visualize_data),
    6: ("Split Data", block_6_split_data),
    7: ("Scale Features", block_7_scale_features),
    8: ("Train Decision Tree", block_8_train_decision_tree),
    9: ("Evaluate Decision Tree", block_9_evaluate_decision_tree),
    10: ("Save Decision Tree", block_10_save_decision_tree),
    11: ("Train SVM", block_11_train_svm),
    12: ("Evaluate SVM", block_12_evaluate_svm),
    13: ("Save SVM Model", block_13_save_svm),
    14: ("Compare Models", block_14_compare_models),
    15: ("Test Predictions", block_15_test_predictions),
}

def show_menu():
    """Display interactive menu"""
    print("\n" + "="*80)
    print("CREDIT CARD FRAUD DETECTION - BLOCK EXECUTION MENU")
    print("="*80)
    
    for num, (name, _) in BLOCKS.items():
        print(f"{num:2d}. {name}")
    
    print("\n16. Run All Blocks")
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
    print("CREDIT CARD FRAUD DETECTION - INTERACTIVE MODE")
    print("="*80)
    print("Execute individual blocks or groups of blocks.")
    print("Uses Decision Tree and SVM models for fraud detection.")
    print("="*80)
    
    while True:
        show_menu()
        
        try:
            user_input = input("\nEnter block number(s) to execute (e.g., 1 or 1,2,3 or 1-5): ").strip()
            
            if user_input == '0':
                print("\n✓ Exiting. Goodbye!")
                break
            
            if user_input == '16':
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
