#!/usr/bin/env python3
"""
Preictal Window Length Comparison
=================================

Compare model performance across different preictal window lengths
to find optimal seizure prediction timing.
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score, accuracy_score, classification_report
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def load_dataset(window_length_sec):
    """Load dataset for specific window length"""
    file_path = f'data/processed/seizure_data_{window_length_sec}s.csv'
    try:
        df = pd.read_csv(file_path)
        print(f'✅ Loaded {window_length_sec}s dataset: {len(df)} windows')
        return df
    except FileNotFoundError:
        print(f'❌ Dataset not found: {file_path}')
        return None

def create_model_pipelines():
    """Create pipelines for different ML algorithms"""
    models = {
        'Random_Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('classifier', RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                class_weight='balanced',
                random_state=42
            ))
        ]),
        'Logistic_Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('classifier', LogisticRegression(
                C=0.01,
                class_weight='balanced',
                max_iter=1000,
                random_state=42
            ))
        ]),
        'SVM': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            ))
        ])
    }
    return models

def evaluate_window_length(window_length_sec, models):
    """Evaluate all models on dataset with specific window length"""
    print(f'\n🔬 Evaluating {window_length_sec}s preictal window')
    print('='*50)
    
    # Load data
    df = load_dataset(window_length_sec)
    if df is None:
        return None
    
    # Prepare features and labels
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    groups = df['subject_id'].values
    
    # Cross-validation setup
    logo = LeaveOneGroupOut()
    results = {}
    
    # Test each model
    for model_name, pipeline in models.items():
        print(f'  Testing {model_name}...')
        
        auc_scores = []
        acc_scores = []
        
        # Leave-one-patient-out cross-validation
        for train_idx, test_idx in logo.split(X, y, groups):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train model
            pipeline.fit(X_train, y_train)
            
            # Predict
            y_prob = pipeline.predict_proba(X_test)[:, 1]
            y_pred = pipeline.predict(X_test)
            
            # Calculate metrics
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            
            auc_scores.append(auc)
            acc_scores.append(acc)
        
        # Store results
        results[model_name] = {
            'auc_mean': np.mean(auc_scores),
            'auc_std': np.std(auc_scores),
            'auc_scores': auc_scores,
            'acc_mean': np.mean(acc_scores),
            'acc_std': np.std(acc_scores)
        }
        
        print(f'    AUC: {np.mean(auc_scores):.3f} ± {np.std(auc_scores):.3f}')
        print(f'    Accuracy: {np.mean(acc_scores):.3f} ± {np.std(acc_scores):.3f}')
    
    return results

def run_comparison():
    """Run comparison across all window lengths"""
    print('🎯 PREICTAL WINDOW LENGTH COMPARISON')
    print('====================================')
    
    # Window lengths to test
    window_lengths = [30, 60, 120, 300]
    models = create_model_pipelines()
    
    all_results = {}
    
    # Test each window length
    for window_sec in window_lengths:
        results = evaluate_window_length(window_sec, models)
        if results:
            all_results[f'{window_sec}s'] = results
    
    # Save results
    output_dir = Path('results/preictal_analysis')
    output_dir.mkdir(parents=True, exist_ok=True)
    
    with open(output_dir / 'window_comparison_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    
    # Generate summary table
    summary_data = []
    for window, window_results in all_results.items():
        for model, metrics in window_results.items():
            summary_data.append({
                'Window_Length': window,
                'Model': model,
                'AUC_Mean': metrics['auc_mean'],
                'AUC_Std': metrics['auc_std'],
                'Accuracy_Mean': metrics['acc_mean'],
                'Accuracy_Std': metrics['acc_std']
            })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'window_comparison_summary.csv', index=False)
    
    # Print summary
    print(f'\n📊 SUMMARY RESULTS')
    print('='*60)
    print(f"{'Window':<8} {'Model':<17} {'AUC':<12} {'Accuracy':<12}")
    print('-'*60)
    
    for _, row in summary_df.iterrows():
        print(f"{row['Window_Length']:<8} {row['Model']:<17} "
              f"{row['AUC_Mean']:.3f}±{row['AUC_Std']:.3f}  "
              f"{row['Accuracy_Mean']:.3f}±{row['Accuracy_Std']:.3f}")
    
    # Find best performers
    print(f'\n🏆 BEST PERFORMERS')
    print('='*30)
    
    best_auc = summary_df.loc[summary_df['AUC_Mean'].idxmax()]
    print(f"Best AUC: {best_auc['Model']} with {best_auc['Window_Length']} "
          f"({best_auc['AUC_Mean']:.3f}±{best_auc['AUC_Std']:.3f})")
    
    best_acc = summary_df.loc[summary_df['Accuracy_Mean'].idxmax()]
    print(f"Best Accuracy: {best_acc['Model']} with {best_acc['Window_Length']} "
          f"({best_acc['Accuracy_Mean']:.3f}±{best_acc['Accuracy_Std']:.3f})")
    
    print(f'\n✅ Results saved to {output_dir}')
    return all_results, summary_df

if __name__ == '__main__':
    results, summary = run_comparison()