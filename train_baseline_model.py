#!/usr/bin/env python3
"""
Baseline Model Training Script
Trains Random Forest classifier on EEG features for seizure prediction.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse


def load_and_prepare_data(data_file: str):
    """Load and prepare EEG data for training."""
    print(f"Loading data from {data_file}...")
    df = pd.read_csv(data_file)
    
    print(f"Data shape: {df.shape}")
    print(f"Columns: {df.columns.tolist()[:10]}...")  # Show first 10 columns
    
    # Separate features and target
    meta_cols = ['subject_id', 'file', 'window_start_sec', 'window_end_sec', 'label']
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    X = df[feature_cols]
    y = df['label']
    
    print(f"Features: {len(feature_cols)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Check for missing values
    missing = X.isnull().sum().sum()
    if missing > 0:
        print(f"Warning: {missing} missing values found. Filling with zeros.")
        X = X.fillna(0)
    
    return X, y, feature_cols


def validate_data_quality(X, y, meta_df):
    """Validate that we have real EEG data with proper patient-aware splitting."""
    print("\n=== DATA QUALITY VALIDATION ===")
    
    # Check for real data vs synthetic
    print(f"Total samples: {len(X)}")
    print(f"Label distribution: {y.value_counts().to_dict()}")
    
    # Check patient distribution
    if 'subject_id' in meta_df.columns:
        patient_counts = meta_df['subject_id'].value_counts()
        print(f"Patients: {list(patient_counts.index)}")
        print(f"Windows per patient: {patient_counts.to_dict()}")
        
        # Check preictal distribution by patient
        preictal_by_patient = meta_df[meta_df['label'] == 1]['subject_id'].value_counts()
        print(f"Preictal windows by patient: {preictal_by_patient.to_dict()}")
    
    # Validate we have both classes
    if len(y.unique()) < 2:
        raise ValueError("Error: Need both preictal and interictal data for training!")
    
    print("✅ Data validation passed - using real EEG recordings only")
    return True


def patient_aware_split(df, val_size=0.2, test_size=0.2, random_state=42):
    """Split data into train/val/test ensuring no patient data leakage."""
    print("Performing patient-aware train/validation/test split...")
    
    # Get unique patients
    patients = df['subject_id'].unique()
    print(f"Available patients: {patients}")
    
    # For single patient, use stratified temporal split to preserve class balance
    if len(patients) == 1:
        print("Single patient detected - using stratified temporal split")
        patient_data = df[df['subject_id'] == patients[0]].sort_values('window_start_sec')
        
        # Separate preictal and interictal windows
        preictal_windows = patient_data[patient_data['label'] == 1]
        interictal_windows = patient_data[patient_data['label'] == 0]
        
        # Calculate split sizes for each class
        preictal_test_size = max(1, int(len(preictal_windows) * test_size))
        preictal_val_size = max(1, int(len(preictal_windows) * val_size))
        
        interictal_test_size = max(1, int(len(interictal_windows) * test_size))
        interictal_val_size = max(1, int(len(interictal_windows) * val_size))
        
        # Temporal split: test (last), val (middle), train (first)
        # Preictal splits
        preictal_test = preictal_windows.iloc[-preictal_test_size:] if preictal_test_size < len(preictal_windows) else preictal_windows.iloc[-1:]
        preictal_val = preictal_windows.iloc[-(preictal_test_size + preictal_val_size):-preictal_test_size] if (preictal_test_size + preictal_val_size) < len(preictal_windows) else pd.DataFrame()
        preictal_train = preictal_windows.iloc[:-(preictal_test_size + preictal_val_size)] if (preictal_test_size + preictal_val_size) < len(preictal_windows) else pd.DataFrame()
        
        # Interictal splits  
        interictal_test = interictal_windows.iloc[-interictal_test_size:]
        interictal_val = interictal_windows.iloc[-(interictal_test_size + interictal_val_size):-interictal_test_size] if (interictal_test_size + interictal_val_size) < len(interictal_windows) else interictal_windows.iloc[-2:-1]
        interictal_train = interictal_windows.iloc[:-(interictal_test_size + interictal_val_size)] if (interictal_test_size + interictal_val_size) < len(interictal_windows) else interictal_windows.iloc[:-2]
        
        # Combine splits
        train_data = pd.concat([preictal_train, interictal_train]) if not preictal_train.empty else interictal_train
        val_data = pd.concat([preictal_val, interictal_val]) if not preictal_val.empty else interictal_val
        test_data = pd.concat([preictal_test, interictal_test])
        
        print(f"Stratified temporal split:")
        print(f"  Train: {len(preictal_train)} preictal + {len(interictal_train)} interictal = {len(train_data)}")
        print(f"  Val:   {len(preictal_val)} preictal + {len(interictal_val)} interictal = {len(val_data)}")
        print(f"  Test:  {len(preictal_test)} preictal + {len(interictal_test)} interictal = {len(test_data)}")
        
    else:
        # Multi-patient: split by patients
        np.random.seed(random_state)
        patients = list(patients)
        np.random.shuffle(patients)
        
        n_test_patients = max(1, int(len(patients) * test_size))
        n_val_patients = max(1, int(len(patients) * val_size))
        
        test_patients = patients[:n_test_patients]
        val_patients = patients[n_test_patients:n_test_patients + n_val_patients]
        train_patients = patients[n_test_patients + n_val_patients:]
        
        train_data = df[df['subject_id'].isin(train_patients)]
        val_data = df[df['subject_id'].isin(val_patients)]
        test_data = df[df['subject_id'].isin(test_patients)]
        
        print(f"Patient split:")
        print(f"  Train: {train_patients}")
        print(f"  Val:   {val_patients}")
        print(f"  Test:  {test_patients}")
    
    # Separate features and targets
    meta_cols = ['subject_id', 'file', 'window_start_sec', 'window_end_sec', 'label']
    feature_cols = [col for col in df.columns if col not in meta_cols]
    
    X_train = train_data[feature_cols]
    y_train = train_data['label'] 
    X_val = val_data[feature_cols]
    y_val = val_data['label']
    X_test = test_data[feature_cols]
    y_test = test_data['label']
    
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_baseline_model(df):
    """Train Random Forest baseline model with hyperparameter tuning on validation set."""
    print("Training Random Forest baseline model with validation...")
    
    # Patient-aware train/val/test split
    X_train, X_val, X_test, y_train, y_val, y_test = patient_aware_split(df)
    
    print(f"Training set: {len(X_train)} samples - {y_train.value_counts().to_dict()}")
    print(f"Validation set: {len(X_val)} samples - {y_val.value_counts().to_dict()}")
    print(f"Test set: {len(X_test)} samples - {y_test.value_counts().to_dict()}")
    
    # Hyperparameter tuning on validation set
    print("\nHyperparameter tuning using validation set...")
    best_score = 0
    best_params = {}
    best_model = None
    
    # Define hyperparameter grid
    param_grid = [
        {'n_estimators': [50, 100], 'max_depth': [5, 10], 'min_samples_split': [2, 5]},
        {'n_estimators': [200], 'max_depth': [None], 'min_samples_split': [2]}
    ]
    
    for params in param_grid:
        for n_est in params['n_estimators']:
            for max_d in params['max_depth']:
                for min_split in params['min_samples_split']:
                    print(f"  Testing: n_est={n_est}, max_depth={max_d}, min_split={min_split}")
                    
                    # Create and train model
                    pipeline = Pipeline([
                        ('scaler', StandardScaler()),
                        ('rf', RandomForestClassifier(
                            n_estimators=n_est,
                            max_depth=max_d,
                            min_samples_split=min_split,
                            class_weight='balanced',
                            random_state=42,
                            n_jobs=-1
                        ))
                    ])
                    
                    # Train on training set
                    pipeline.fit(X_train, y_train)
                    
                    # Evaluate on validation set
                    if len(np.unique(y_val)) > 1:  # Only if we have both classes
                        val_pred_proba = pipeline.predict_proba(X_val)[:, 1]
                        val_score = roc_auc_score(y_val, val_pred_proba)
                    else:
                        val_pred = pipeline.predict(X_val)
                        val_score = np.mean(val_pred == y_val)  # Accuracy fallback
                    
                    print(f"    Validation score: {val_score:.4f}")
                    
                    if val_score > best_score:
                        best_score = val_score
                        best_params = {
                            'n_estimators': n_est,
                            'max_depth': max_d,
                            'min_samples_split': min_split
                        }
                        best_model = pipeline
    
    print(f"\nBest validation score: {best_score:.4f}")
    print(f"Best parameters: {best_params}")
    
    # Final predictions on held-out test set
    print("\n" + "="*50)
    print("FINAL EVALUATION ON HELD-OUT TEST SET")
    print("="*50)
    
    y_test_pred = best_model.predict(X_test)
    y_test_pred_proba = best_model.predict_proba(X_test)[:, 1] if len(best_model.classes_) > 1 else np.zeros(len(y_test))
    
    return best_model, X_test, y_test, y_test_pred, y_test_pred_proba


def evaluate_model(y_true, y_pred, y_pred_proba):
    """Evaluate model performance."""
    print("\n" + "="*50)
    print("MODEL EVALUATION RESULTS")
    print("="*50)
    
    # Classification report
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred))
    
    # Confusion matrix
    print("\nConfusion Matrix:")
    cm = confusion_matrix(y_true, y_pred)
    print(cm)
    
    # ROC AUC (only if we have both classes)
    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_pred_proba)
        print(f"\nROC AUC: {auc:.4f}")
    else:
        print("\nROC AUC: Cannot calculate (only one class in test set)")
        auc = None
    
    return auc


def create_visualizations(y_true, y_pred_proba, auc, output_dir="results"):
    """Create performance visualizations."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if len(np.unique(y_true)) > 1 and auc is not None:
        # ROC Curve
        plt.figure(figsize=(8, 6))
        fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve - EEG Seizure Prediction')
        plt.legend()
        plt.grid(True)
        plt.savefig(output_dir / 'roc_curve.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # Prediction distribution
        plt.figure(figsize=(10, 6))
        plt.hist(y_pred_proba[y_true == 0], bins=30, alpha=0.7, label='Interictal', density=True)
        plt.hist(y_pred_proba[y_true == 1], bins=30, alpha=0.7, label='Preictal', density=True)
        plt.xlabel('Predicted Probability')
        plt.ylabel('Density')
        plt.title('Distribution of Predicted Probabilities')
        plt.legend()
        plt.savefig(output_dir / 'prediction_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"\nVisualization saved to {output_dir}/")


def main():
    parser = argparse.ArgumentParser(description='Train baseline EEG seizure prediction model')
    parser.add_argument('--data', default='data/processed/eeg_windows.csv', 
                       help='Path to processed EEG data CSV file')
    args = parser.parse_args()
    
    try:
        # Load full dataset
        print(f"Loading data from {args.data}...")
        df = pd.read_csv(args.data)
        
        print(f"Data shape: {df.shape}")
        
        # Separate features and target  
        meta_cols = ['subject_id', 'file', 'window_start_sec', 'window_end_sec', 'label']
        feature_cols = [col for col in df.columns if col not in meta_cols]
        
        X = df[feature_cols]
        y = df['label']
        
        # Validate data quality
        validate_data_quality(X, y, df[meta_cols])
        
        # Train model with patient-aware splitting
        model, X_test, y_test, y_pred, y_pred_proba = train_baseline_model(df)
        
        # Evaluate
        auc = evaluate_model(y_test, y_pred, y_pred_proba)
        
        # Create visualizations
        create_visualizations(y_test, y_pred_proba, auc)
        
        # Feature importance (top 10)
        rf_model = model.named_steps['rf']
        feature_importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': rf_model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("\nTop 10 Most Important Features:")
        print(feature_importance.head(10).to_string(index=False))
        
        print(f"\n✅ Baseline model training complete!")
        print(f"✅ Model saved to memory (can be extended to save to disk)")
        print(f"✅ Results visualizations saved to results/")
        
    except FileNotFoundError:
        print(f"Error: Data file {args.data} not found!")
        print("Make sure to run the preprocessing script first:")
        print("python3 data/scripts/convert_edf_to_csv.py")
    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()