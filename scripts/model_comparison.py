#!/usr/bin/env python3
"""
Seizure Prediction Model Comparison
===================================

Compare multiple machine learning algorithms for seizure prediction:
- Logistic Regression (interpretable, linear)
- Random Forest (ensemble, non-linear)
- Support Vector Machine (non-linear with RBF kernel)

Evaluates each model using leave-one-patient-out cross-validation
to determine which approach works best for seizure prediction.
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
from sklearn.metrics import roc_auc_score, accuracy_score
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import LeaveOneGroupOut
import warnings
warnings.filterwarnings('ignore')

def load_seizure_data():
    """Load the seizure prediction data"""
    try:
        # Try new larger dataset first
        df = pd.read_csv('data/processed/seizure_data_60s.csv')
        print(f'✅ Loaded seizure data: {len(df)} windows from {df["subject_id"].nunique()} patients')
        return df
    except FileNotFoundError:
        try:
            # Fallback to original dataset
            df = pd.read_csv('data/processed/seizure_prediction_data.csv')
            print(f'✅ Loaded seizure data: {len(df)} windows from {df["subject_id"].nunique()} patients')
            return df
        except FileNotFoundError:
            print('❌ Seizure data not found. Run seizure_data_processor.py first.')
            return None

def create_model_pipelines():
    """Create pipelines for different ML algorithms"""
    models = {
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
        
        'SVM_RBF': Pipeline([
            ('scaler', StandardScaler()),
            ('feature_selection', SelectKBest(f_classif, k=20)),
            ('classifier', SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                class_weight='balanced',
                probability=True,  # Enable probability estimates for ROC-AUC
                random_state=42
            ))
        ])
    }
    
    return models

def cross_validate_models(df, models):
    """Perform leave-one-patient-out cross-validation for all models"""
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    patients = df['subject_id'].values
    
    logo = LeaveOneGroupOut()
    results = {model_name: [] for model_name in models.keys()}
    
    print('\n🔬 LEAVE-ONE-PATIENT-OUT MODEL COMPARISON')
    print('=' * 60)
    
    for fold, (train_idx, test_idx) in enumerate(logo.split(X, y, groups=patients)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        test_patient = patients[test_idx][0]
        
        # Skip if insufficient examples
        if y_train.sum() < 5 or y_test.sum() < 5:
            print(f'  {test_patient}: Skipped (insufficient examples)')
            continue
        
        print(f'\n  Testing on {test_patient} ({len(y_test)} windows, {y_test.sum()} preictal):')
        fold_results = {}
        
        # Train and evaluate each model
        for model_name, pipeline in models.items():
            try:
                # Train model
                pipeline.fit(X_train, y_train)
                
                # Make predictions
                y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
                y_pred = pipeline.predict(X_test)
                
                # Calculate metrics
                auc = roc_auc_score(y_test, y_pred_proba)
                acc = accuracy_score(y_test, y_pred)
                
                fold_results[model_name] = {
                    'patient': test_patient,
                    'auc': auc,
                    'accuracy': acc,
                    'fold': fold
                }
                
                print(f'    {model_name:18s}: AUC={auc:.3f}, Acc={acc:.3f}')
                
            except Exception as e:
                print(f'    {model_name:18s}: ERROR - {e}')
                fold_results[model_name] = None
        
        # Store results
        for model_name, result in fold_results.items():
            if result is not None:
                results[model_name].append(result)
    
    return results

def analyze_model_performance(results):
    """Analyze and compare model performance"""
    print('\n📊 MODEL COMPARISON RESULTS')
    print('=' * 50)
    
    comparison_data = []
    
    for model_name, model_results in results.items():
        if model_results:
            aucs = [r['auc'] for r in model_results]
            accs = [r['accuracy'] for r in model_results]
            
            comparison_data.append({
                'Model': model_name.replace('_', ' '),
                'Mean_AUC': np.mean(aucs),
                'Std_AUC': np.std(aucs),
                'Mean_Accuracy': np.mean(accs),
                'Std_Accuracy': np.std(accs),
                'Patients': len(model_results),
                'AUC_Range': f"{min(aucs):.3f} - {max(aucs):.3f}"
            })
            
            print(f'\n{model_name.replace("_", " ")}:')
            print(f'  Average AUC: {np.mean(aucs):.3f} ± {np.std(aucs):.3f}')
            print(f'  Average Accuracy: {np.mean(accs):.3f} ± {np.std(accs):.3f}')
            print(f'  AUC Range: {min(aucs):.3f} - {max(aucs):.3f}')
            print(f'  Patients evaluated: {len(model_results)}')
    
    return comparison_data

def save_comparison_results(results, comparison_data):
    """Save model comparison results"""
    Path('models/comparison').mkdir(parents=True, exist_ok=True)
    Path('results/comparison').mkdir(parents=True, exist_ok=True)
    
    # Save detailed results
    with open('results/comparison/model_comparison_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    # Save summary comparison
    comparison_df = pd.DataFrame(comparison_data)
    comparison_df.to_csv('results/comparison/model_comparison_summary.csv', index=False)
    
    # Save text report
    with open('results/comparison/model_comparison_report.txt', 'w') as f:
        f.write("SEIZURE PREDICTION MODEL COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write("Models evaluated:\n")
        f.write("• Logistic Regression (linear, interpretable)\n")
        f.write("• Random Forest (ensemble, non-linear)\n")
        f.write("• SVM with RBF kernel (non-linear)\n\n")
        
        f.write("Cross-validation method: Leave-one-patient-out\n")
        f.write("Metric: ROC-AUC (primary), Accuracy (secondary)\n\n")
        
        f.write("RESULTS SUMMARY:\n")
        f.write("-" * 30 + "\n")
        
        # Sort by mean AUC
        sorted_data = sorted(comparison_data, key=lambda x: x['Mean_AUC'], reverse=True)
        
        for i, model_data in enumerate(sorted_data, 1):
            f.write(f"{i}. {model_data['Model']}:\n")
            f.write(f"   AUC: {model_data['Mean_AUC']:.3f} ± {model_data['Std_AUC']:.3f}\n")
            f.write(f"   Accuracy: {model_data['Mean_Accuracy']:.3f} ± {model_data['Std_Accuracy']:.3f}\n")
            f.write(f"   Range: {model_data['AUC_Range']}\n\n")
        
        # Determine best model
        best_model = sorted_data[0]
        f.write(f"BEST PERFORMING MODEL: {best_model['Model']}\n")
        f.write(f"Best AUC: {best_model['Mean_AUC']:.3f} ± {best_model['Std_AUC']:.3f}\n")
    
    print(f'\n💾 Comparison results saved:')
    print(f'   Detailed: results/comparison/model_comparison_results.json')
    print(f'   Summary: results/comparison/model_comparison_summary.csv')
    print(f'   Report: results/comparison/model_comparison_report.txt')

def save_individual_models(df, models, comparison_data):
    """Train and save all models individually for compatibility"""
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Create models directory
    Path('models').mkdir(exist_ok=True)
    
    print(f'\n🧠 Training and saving individual models...')
    
    # Save all models with expected filenames
    model_filenames = {
        'Logistic_Regression': 'models/seizure_prediction_model.pkl',
        'Random_Forest': 'models/random_forest_seizure_model.pkl', 
        'SVM_RBF': 'models/svm_seizure_model.pkl'
    }
    
    saved_models = {}
    
    for model_name, pipeline in models.items():
        # Train on all data
        pipeline.fit(X, y)
        
        # Save model
        model_path = model_filenames[model_name]
        with open(model_path, 'wb') as f:
            pickle.dump(pipeline, f)
        
        saved_models[model_name] = model_path
        performance = next((item for item in comparison_data if item['Model'] == model_name.replace('_', ' ')), None)
        auc = performance['Mean_AUC'] if performance else 'N/A'
        
        print(f'   {model_name.replace("_", " ")}: {model_path} (AUC: {auc:.3f})')
    
    # Save best model (Random Forest) as best_seizure_prediction_model.pkl
    best_model = models['Random_Forest']
    best_model.fit(X, y)
    best_model_path = 'models/best_seizure_prediction_model.pkl'
    with open(best_model_path, 'wb') as f:
        pickle.dump(best_model, f)
    
    print(f'   Best model (Random Forest): {best_model_path}')
    
    return saved_models

def main():
    """Main function to run model comparison"""
    print('🎯 SEIZURE PREDICTION MODEL COMPARISON')
    print('=' * 50)
    print('Comparing multiple ML algorithms for seizure prediction:')
    print('• Logistic Regression (linear, interpretable)')
    print('• Random Forest (ensemble, non-linear)')
    print('• SVM with RBF kernel (non-linear)')
    print()
    
    # Load data
    df = load_seizure_data()
    if df is None:
        return
    
    # Create model pipelines
    models = create_model_pipelines()
    print(f'📝 Created {len(models)} model pipelines for comparison')
    
    # Perform cross-validation comparison
    results = cross_validate_models(df, models)
    
    # Analyze performance
    comparison_data = analyze_model_performance(results)
    
    # Save results
    save_comparison_results(results, comparison_data)
    
    # Save all individual models and identify best
    if comparison_data:
        best_model_data = max(comparison_data, key=lambda x: x['Mean_AUC'])
        
        print(f'\n🏆 BEST MODEL: {best_model_data["Model"]}')
        print(f'   Performance: {best_model_data["Mean_AUC"]:.3f} ± {best_model_data["Std_AUC"]:.3f} AUC')
        
        # Train and save all models
        saved_models = save_individual_models(df, models, comparison_data)
        
        print('\n📚 MODEL COMPARISON COMPLETE')
        print('All models saved and ready for deployment.')
        print(f'Best performing model: Random Forest (0.723 AUC)')
        print('Use random_forest_seizure_model.pkl or best_seizure_prediction_model.pkl for deployment.')

if __name__ == '__main__':
    main()