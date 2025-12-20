#!/usr/bin/env python3

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import joblib
from sklearn.metrics import roc_curve, auc, confusion_matrix, classification_report
from sklearn.model_selection import cross_val_predict, StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('default')
sns.set_palette("husl")

def load_data_and_results():
    """Load the data and model comparison results."""
    
    # Load processed data - try new larger dataset first
    data_path = "data/processed/seizure_data_60s.csv"
    if not Path(data_path).exists():
        data_path = "data/processed/real_seizure_targeted_data.csv"
        if not Path(data_path).exists():
            data_path = "data/processed/seizure_prediction_data.csv"
    
    df = pd.read_csv(data_path)
    
    # Load model comparison results
    results_path = "results/comparison/model_comparison_results.json"
    with open(results_path, 'r') as f:
        results = json.load(f)
    
    return df, results

def create_model_comparison_figure(results, output_dir):
    """Create simple bar chart comparing model performance."""
    
    print("📊 Creating model comparison figure...")
    
    # Extract performance data
    models = ['Logistic Regression', 'Random Forest', 'SVM RBF']
    auc_means = []
    auc_stds = []
    
    for model_key in ['Logistic_Regression', 'Random_Forest', 'SVM_RBF']:
        model_results = results[model_key]
        auc_scores = [fold['auc'] for fold in model_results]
        auc_means.append(np.mean(auc_scores))
        auc_stds.append(np.std(auc_scores))
    
    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    bars = ax.bar(x, auc_means, yerr=auc_stds, capsize=5, 
                  color=colors, alpha=0.8)
    
    # Add value labels on bars
    for i, (mean, std) in enumerate(zip(auc_means, auc_stds)):
        ax.text(i, mean + std + 0.01, f'{mean:.3f}', 
                ha='center', va='bottom', fontweight='bold')
    
    ax.set_xlabel('Model')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Model Performance Comparison\n(Leave-One-Patient-Out Cross-Validation)')
    ax.set_xticks(x)
    ax.set_xticklabels(models)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_dataset_distribution_figure(df, output_dir):
    """Create figure showing dataset characteristics."""
    
    print("📊 Creating dataset distribution figure...")
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # 1. Class distribution
    class_counts = df['label'].value_counts()
    axes[0,0].pie([class_counts[0], class_counts[1]], 
                  labels=['Interictal', 'Preictal'],
                  colors=['lightblue', 'orange'],
                  autopct='%1.1f%%')
    axes[0,0].set_title('Class Distribution')
    
    # 2. Patient distribution
    patient_counts = df['subject_id'].value_counts().sort_index()
    axes[0,1].bar(patient_counts.index, patient_counts.values, color='skyblue')
    axes[0,1].set_xlabel('Patient ID')
    axes[0,1].set_ylabel('Number of Windows')
    axes[0,1].set_title('Windows per Patient')
    
    # 3. Preictal windows per patient
    preictal_by_patient = df[df['label'] == 1].groupby('subject_id').size()
    axes[1,0].bar(preictal_by_patient.index, preictal_by_patient.values, color='orange')
    axes[1,0].set_xlabel('Patient ID')
    axes[1,0].set_ylabel('Preictal Windows')
    axes[1,0].set_title('Preictal Windows per Patient')
    
    # 4. Class distribution by patient
    class_by_patient = df.groupby(['subject_id', 'label']).size().unstack()
    class_by_patient.plot(kind='bar', stacked=True, ax=axes[1,1], 
                         color=['lightblue', 'orange'], 
                         legend=True)
    axes[1,1].set_xlabel('Patient ID')
    axes[1,1].set_ylabel('Number of Windows')
    axes[1,1].set_title('Class Distribution by Patient')
    axes[1,1].legend(['Interictal', 'Preictal'])
    axes[1,1].tick_params(axis='x', rotation=0)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_cross_patient_performance_figure(results, output_dir):
    """Create figure showing performance across individual patients."""
    
    print("📊 Creating cross-patient performance figure...")
    
    patients = ['PN00', 'PN03', 'PN05', 'PN06']
    models = ['Logistic_Regression', 'Random_Forest', 'SVM_RBF']
    model_names = ['Logistic Regression', 'Random Forest', 'SVM RBF']
    
    # Extract patient-specific results
    patient_aucs = {model: [] for model in models}
    
    for model in models:
        for fold in results[model]:
            patient_aucs[model].append(fold['auc'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(patients))
    width = 0.25
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, (model, name) in enumerate(zip(models, model_names)):
        offset = (i - 1) * width
        bars = ax.bar(x + offset, patient_aucs[model], width, 
                     label=name, color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, v in enumerate(patient_aucs[model]):
            ax.text(x[j] + offset, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=9)
    
    ax.set_xlabel('Test Patient')
    ax.set_ylabel('ROC-AUC Score')
    ax.set_title('Cross-Patient Performance\n(Leave-One-Patient-Out Results)')
    ax.set_xticks(x)
    ax.set_xticklabels(patients)
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color='red', linestyle='--', alpha=0.5, label='Random Chance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'cross_patient_performance.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_roc_curves_figure(df, output_dir):
    """Create ROC curves - both simple and cross-validated versions."""
    
    print("📊 Creating ROC curves...")
    
    # Load trained models
    lr_model = joblib.load('models/seizure_prediction_model.pkl')
    rf_model = joblib.load('models/random_forest_seizure_model.pkl')
    svm_model = joblib.load('models/svm_seizure_model.pkl')
    
    # Prepare data
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Create figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    models = [
        ('Logistic Regression', lr_model, '#1f77b4'),
        ('Random Forest', rf_model, '#2ca02c'),
        ('SVM RBF', svm_model, '#ff7f0e')
    ]
    
    # First subplot: Simple ROC curves (may show overfitting)
    for name, model, color in models:
        # Get prediction probabilities
        y_pred_proba = model.predict_proba(X)[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax1.plot(fpr, tpr, color=color, lw=2, 
               label=f'{name} (AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax1.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.5)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])
    ax1.set_xlabel('False Positive Rate')
    ax1.set_ylabel('True Positive Rate')
    ax1.set_title('ROC Curves - Training Set Performance\n(May show overfitting)')
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.3)
    
    # Second subplot: Cross-validated ROC curves (unbiased)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model, color in models:
        # Get cross-validated predictions
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, _ = roc_curve(y, y_pred_proba)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        ax2.plot(fpr, tpr, color=color, lw=2, 
               label=f'{name} (CV AUC = {roc_auc:.3f})')
    
    # Plot diagonal line
    ax2.plot([0, 1], [0, 1], color='red', lw=2, linestyle='--', alpha=0.5)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])
    ax2.set_xlabel('False Positive Rate')
    ax2.set_ylabel('True Positive Rate')
    ax2.set_title('ROC Curves - Cross-Validated Performance\n(Avoiding Overfitting)')
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'roc_curves_cv.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_confusion_matrices_figure(df, output_dir):
    """Create confusion matrices for all models."""
    
    print("📊 Creating confusion matrices...")
    
    # Load trained models
    models = [
        ('Logistic Regression', joblib.load('models/seizure_prediction_model.pkl')),
        ('Random Forest', joblib.load('models/random_forest_seizure_model.pkl')),
        ('SVM RBF', joblib.load('models/svm_seizure_model.pkl'))
    ]
    
    # Prepare data
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Create figure with subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Use cross-validation to get unbiased predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (name, model) in enumerate(models):
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)
        
        # Calculate confusion matrix
        cm = confusion_matrix(y, y_pred)
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['Interictal', 'Preictal'],
                   yticklabels=['Interictal', 'Preictal'],
                   ax=axes[i])
        
        axes[i].set_title(f'{name}\nConfusion Matrix')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_metrics_figure(df, output_dir):
    """Create comprehensive performance metrics comparison."""
    
    print("📊 Creating performance metrics comparison...")
    
    # Load trained models
    models = [
        ('Logistic Regression', joblib.load('models/seizure_prediction_model.pkl')),
        ('Random Forest', joblib.load('models/random_forest_seizure_model.pkl')),
        ('SVM RBF', joblib.load('models/svm_seizure_model.pkl'))
    ]
    
    # Prepare data
    feature_cols = [col for col in df.columns if col.startswith('feature_')]
    X = df[feature_cols].values
    y = df['label'].values
    
    # Use cross-validation to get unbiased predictions
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    metrics_data = []
    
    for name, model in models:
        # Get cross-validated predictions
        y_pred = cross_val_predict(model, X, y, cv=cv)
        y_pred_proba = cross_val_predict(model, X, y, cv=cv, method='predict_proba')[:, 1]
        
        # Calculate metrics
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()
        
        metrics_data.append({
            'Model': name,
            'Accuracy': accuracy_score(y, y_pred),
            'Precision': precision_score(y, y_pred),
            'Recall': recall_score(y, y_pred),
            'F1-Score': f1_score(y, y_pred),
            'Specificity': tn / (tn + fp),
            'AUC': auc(*roc_curve(y, y_pred_proba)[:2])
        })
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 6))
    
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score', 'Specificity', 'AUC']
    x = np.arange(len(metrics_to_plot))
    width = 0.25
    
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e']
    
    for i, model_name in enumerate(['Logistic Regression', 'Random Forest', 'SVM RBF']):
        values = [metrics_df[metrics_df['Model'] == model_name][metric].iloc[0] 
                 for metric in metrics_to_plot]
        offset = (i - 1) * width
        ax.bar(x + offset, values, width, label=model_name, 
               color=colors[i], alpha=0.8)
        
        # Add value labels
        for j, v in enumerate(values):
            ax.text(x[j] + offset, v + 0.01, f'{v:.3f}', 
                   ha='center', va='bottom', fontsize=8, rotation=45)
    
    ax.set_xlabel('Metrics')
    ax.set_ylabel('Score')
    ax.set_title('Comprehensive Performance Metrics Comparison\n(Cross-Validated)')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_to_plot)
    ax.set_ylim(0, 1.1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / 'performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()

def main():
    """Generate all figures."""
    
    print("📊 Generating All Analysis Figures")
    print("=" * 50)
    
    # Create output directory
    output_dir = Path("results/figures")
    output_dir.mkdir(exist_ok=True)
    
    # Load data
    df, results = load_data_and_results()
    print(f"📈 Loaded data: {len(df)} samples, {len(df.columns)} features")
    
    # Create all figures
    create_model_comparison_figure(results, output_dir)
    create_dataset_distribution_figure(df, output_dir)
    create_cross_patient_performance_figure(results, output_dir)
    create_roc_curves_figure(df, output_dir)
    create_confusion_matrices_figure(df, output_dir)
    create_performance_metrics_figure(df, output_dir)
    
    print("\n✅ All figures generated successfully!")
    print(f"📁 Saved in: {output_dir}")
    print("\nGenerated figures:")
    print("  • model_comparison.png - Performance comparison bar chart")
    print("  • dataset_distribution.png - Dataset characteristics")
    print("  • cross_patient_performance.png - Patient-specific results")
    print("  • roc_curves_cv.png - ROC curves with proper cross-validation")
    print("  • confusion_matrices.png - Confusion matrices for all models")
    print("  • performance_metrics.png - Comprehensive metrics comparison")
    
    print("\n🔍 ROC Curve Analysis:")
    print("  • Previous ROC showing AUC=1.0 was due to overfitting")
    print("  • New ROC curves use proper cross-validation")
    print("  • Cross-validated AUC scores are more realistic and unbiased")

if __name__ == "__main__":
    main()