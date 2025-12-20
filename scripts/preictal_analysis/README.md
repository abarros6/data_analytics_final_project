# Preictal Window Analysis Scripts

This folder contains scripts for analyzing the impact of different preictal window lengths on seizure prediction performance.

## Research Question
**How does varying preictal window length affect seizure prediction performance?**

## Scripts Overview

### 1. `variable_preictal_processor.py`
**Purpose**: Generate datasets with different preictal window lengths
**Usage**: 
```bash
python scripts/preictal_analysis/variable_preictal_processor.py
```
**Outputs**: 
- `data/processed/seizure_data_30s.csv`
- `data/processed/seizure_data_60s.csv` 
- `data/processed/seizure_data_120s.csv`
- `data/processed/seizure_data_300s.csv`

### 2. `preictal_window_comparison.py`
**Purpose**: Compare model performance across all window lengths
**Usage**:
```bash
python scripts/preictal_analysis/preictal_window_comparison.py
```
**Outputs**:
- `results/preictal_analysis/window_comparison_results.json`
- `results/preictal_analysis/window_comparison_summary.csv`

### 3. `generate_preictal_figures.py`
**Purpose**: Generate publication-ready figures for IEEE paper
**Usage**:
```bash
python scripts/preictal_analysis/generate_preictal_figures.py
```
**Outputs**:
- `results/preictal_analysis/preictal_performance_heatmap.png`
- `results/preictal_analysis/window_optimization_curves.png`
- `results/preictal_analysis/clinical_tradeoff_analysis.png`
- `results/preictal_analysis/dataset_distribution_analysis.png`

## Complete Pipeline
Run all scripts in sequence:
```bash
# 1. Generate datasets with different window lengths
python scripts/preictal_analysis/variable_preictal_processor.py

# 2. Compare model performance
python scripts/preictal_analysis/preictal_window_comparison.py

# 3. Generate figures
python scripts/preictal_analysis/generate_preictal_figures.py
```

## Key Findings
- **Optimal window**: 120 seconds (2 minutes)
- **Best performance**: Random Forest with 0.626 AUC
- **Trade-off**: Longer windows provide more warning time but may decrease specificity
- **Clinical relevance**: 120s window balances prediction accuracy with practical intervention time

## Dependencies
- pandas, numpy, scikit-learn
- matplotlib, seaborn
- mne (for EEG processing)
- All dependencies from main project