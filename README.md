# EEG Data Analytics Final Project

A machine learning project analyzing EEG data for seizure prediction using the Siena Scalp EEG Database.

## Team Members

- Dylan Abbinett
- Kelsey Kloosterman  
- Anthony Barros

## Project Overview

This project applies machine learning techniques to EEG (electroencephalography) data for **seizure prediction**. We aim to predict preictal (pre-seizure) states from scalp EEG recordings, enabling the development of wearable monitoring and intervention devices for epilepsy patients.

The core problem is a **binary classification task**: given a window of multi-channel EEG data, predict whether the patient is in a preictal state (leading to seizure) or an interictal state (non-seizure).

### Dataset

**Siena Scalp EEG Database v1.0.0**
- **Source**: https://physionet.org/content/siena-scalp-eeg/1.0.0/
- **Size**: ~20 GB (128 hours of continuous recording)
- **Subjects**: 14 epilepsy patients (9 male, 5 female; ages 20-71)
- **Seizures**: 47 documented seizure events across all recordings
- **Sampling Rate**: 512 Hz
- **Electrode Setup**: International 10-20 System (16-19 EEG channels per subject)
- **Reference**: Detti et al. (2020). EEG Synchronization Analysis for Seizure Prediction. *Processes*, 8(7), 846. https://doi.org/10.3390/pr8070846

The dataset includes expert clinical validation of seizure events and ILAE (International League Against Epilepsy) seizure classification, making it well-suited for developing and validating ML-based seizure prediction algorithms.

## Getting Started

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Key dependencies:
- **mne**: EEG signal processing and EDF file handling
- **scikit-learn**: Traditional ML models (Random Forest, SVM)
- **torch**: Neural network implementations
- **pandas, numpy, scipy**: Data manipulation and analysis

### 2. Download and Process Data

**Download the Dataset:**
```bash
python data/scripts/download_data.py --setup-dirs
```

**Test Data Loading (test this first):**
```bash
python data/scripts/convert_edf_to_csv.py --test-loading
```

**Convert EDF to ML-Ready CSV:**
```bash
python data/scripts/convert_edf_to_csv.py
```

This creates `data/processed/eeg_windows.csv` containing preprocessed, labeled EEG windows.

### 3. Run Analysis

```bash
jupyter notebook notebooks/
```

---

## Methodology

### Data Preprocessing Pipeline

Our `convert_edf_to_csv.py` script implements comprehensive preprocessing with the following steps:

#### 1. **Data Loading & Channel Selection**
- **EDF File Loading**: Uses MNE library for robust EDF parsing
- **Channel Filtering**: Automatically excludes non-EEG channels (EKG, annotations)
  - Keeps 29 EEG channels per subject (Fp1, F3, C3, P3, O1, F7, T3, T5, etc.)
  - Excludes EKG channels 33-34 to prevent model from learning cardiac patterns
- **Sampling Rate**: Preserves original 512 Hz sampling rate
- **Error Handling**: Graceful handling of corrupted or missing files

#### 2. **Signal Preprocessing**
- **Bandpass Filtering**: 0.5-40 Hz using MNE's FIR filter
  - **High-pass (0.5 Hz)**: Removes DC drift and very slow artifacts
  - **Low-pass (40 Hz)**: Removes muscle noise, line noise, and high-frequency artifacts
  - **Rationale**: Clinically relevant seizure activity occurs in 0.5-40 Hz range
  - **Implementation**: `raw.filter(0.5, 40.0, fir_design='firwin')`

#### 3. **Temporal Windowing**
- **Fixed Windows**: 30-second segments (configurable: `--window-size`)
- **Non-overlapping**: Default stride = window size (configurable: `--stride`)
- **Window Trade-offs**:
  - Shorter windows: Lower latency, less context, more samples
  - Longer windows: Higher latency, more context, fewer samples
- **Time Alignment**: Each window tagged with start/end timestamps

#### 4. **Seizure-Based Labeling**
- **Label Classes**:
  - `0`: **Interictal** (normal, non-seizure periods)
  - `1`: **Preictal** (5 minutes before seizure onset, configurable: `--preictal-window`)
- **Exclusion Strategy**:
  - **Ictal periods**: During seizure events (completely excluded)
  - **Early postictal**: Immediately after seizures (excluded to avoid confounding)
- **Time Parsing**: Robust parsing of `h.m.s` format from seizure annotation files
- **Multi-seizure Handling**: Processes multiple seizures per subject/file

#### 5. **Normalization & Feature Engineering**
- **Z-score Normalization**: Applied per channel, per window
  - Formula: `(x - mean(x)) / std(x)` for each channel in each window
  - **Prevents**: Channel-specific amplitude biases
  - **Ensures**: Model learns patterns, not absolute amplitudes
- **Feature Format**: Each time sample becomes a feature column
  - Column naming: `{channel}_t{timepoint}` (e.g., `Fp1_t0`, `Fp1_t1`, ...)
  - For 30s @ 512Hz: 15,360 samples × 29 channels = 445,440 features per window

#### 6. **Quality Control & Validation**
- **Missing Data**: Handles subjects with missing EDF or seizure files
- **Progress Tracking**: Real-time processing status per subject
- **Statistics**: Reports total windows, class distribution, processing errors
- **Memory Management**: Efficient processing of large datasets
- **Reproducibility**: Consistent preprocessing across all subjects

#### 7. **Output Format**
- **CSV Structure**:
  ```
  subject_id | file | window_start_sec | window_end_sec | Fp1_t0 | Fp1_t1 | ... | label
  PN00      | PN00-1.edf | 0.0 | 30.0 | -0.23 | 0.41 | ... | 0
  ```
- **Metadata Columns**: Subject ID, source file, temporal boundaries
- **Feature Columns**: Normalized EEG samples (445,440 columns for 30s windows)
- **Target Column**: Binary label for classification

#### 8. **Configurable Parameters**
All preprocessing steps are configurable via command-line arguments:

```bash
python data/scripts/convert_edf_to_csv.py \
    --window-size 30 \          # Window length in seconds
    --preictal-window 300 \     # Preictal period (5 min)
    --filter-low 0.5 \          # High-pass filter cutoff
    --filter-high 40.0 \        # Low-pass filter cutoff
    --stride 30 \               # Window stride (overlap control)
    --data-dir data/raw/siena-scalp-eeg \
    --output data/processed/eeg_windows.csv
```

#### 9. **Preprocessing Validation**
- **Test Mode**: `--test-loading` validates preprocessing without saving
- **Sample Output**: Shows channel count, feature dimensions, class distribution
- **Memory Estimation**: Reports DataFrame size and memory usage
- **Format Verification**: Confirms column structure and data types

#### 10. **Class Imbalance Considerations**
The preprocessing automatically handles the natural class imbalance:
- **Expected Distribution**: ~95-98% interictal, ~2-5% preictal windows
- **Preservation**: Natural imbalance preserved for realistic evaluation
- **Downstream Handling**: Imbalance addressed in model training phase via:
  - Weighted loss functions
  - SMOTE oversampling
  - Stratified train/val/test splits
  - Threshold optimization

### Class Imbalance Considerations

**The Problem**: 
The dataset is highly imbalanced—interictal windows vastly outnumber preictal windows. With 47 seizures over 128 hours of recording, preictal windows (assuming 5-15 min before seizure) constitute only ~2-5% of the dataset.

**Strategies to Handle Imbalance**:

1. **Weighted Loss Functions** (Recommended for neural networks)
   - Assign higher loss weight to minority (preictal) class
   - PyTorch example: `torch.nn.BCEWithLogitsLoss(pos_weight=weight)`
   - Weight ratio ≈ (n_interictal / n_preictal)

2. **Class Weights in Tree Models** (For Random Forest/SVM)
   - scikit-learn: `class_weight='balanced'` automatically adjusts for imbalance
   - Penalizes minority class misclassification more heavily

3. **SMOTE** (Synthetic Minority Over-sampling)
   - Generate synthetic preictal samples via interpolation
   - Apply only to training set; validate/test on real data
   - Caution: Can lead to overfitting if not carefully applied

4. **Threshold Adjustment**
   - Default classification threshold is 0.5; can adjust based on desired sensitivity/specificity
   - For seizure prediction, high sensitivity (catch all seizures) may be prioritized

5. **Stratified Train/Test Split**
   - Ensure train/val/test sets maintain original class distribution
   - Prevents training set with too few positive examples


## Train Phase

### Models to Evaluate

#### 1. Baseline: Hand-Crafted Features + Traditional ML

**Feature Extraction** (from raw EEG):
- **Spectral Power**: Power in frequency bands (delta: 0.5-4 Hz, theta: 4-8 Hz, alpha: 8-12 Hz, beta: 12-30 Hz, gamma: 30-40 Hz)
- **Functional Connectivity**: Cross-frequency coupling, phase synchronization between channels
- **Hjorth Parameters**: Activity, mobility, complexity of signal
- **Entropy Measures**: Approximate Entropy, Sample Entropy
- **Statistical**: Mean, variance, skewness, kurtosis per channel

**Models**:
- **Random Forest** (100-500 trees, max_depth tunable)
  - Advantages: Fast, interpretable feature importance, handles imbalance with class weights
  - Baseline for comparison; often outperforms more complex models on tabular data
  
- **Support Vector Machine** (RBF kernel)
  - Advantages: Good generalization, naturally handles imbalance with `class_weight='balanced'`
  - Disadvantages: Slower to train, less interpretable
  
- **Logistic Regression** (with L2 regularization)
  - Advantages: Provides probability estimates, simple baseline
  - Disadvantages: Assumes linear decision boundary

#### 2. Temporal Neural Networks

**LSTM/GRU** (Recurrent Networks):
- **Architecture**: Multi-layer LSTM processing 16-19 channels sequentially
- **Rationale**: Captures temporal dependencies in EEG; "remembers" relevant patterns from prior timesteps
- **Hyperparameters**: Hidden size (64-512), num layers (1-3), dropout (0.2-0.5)
- **Input**: (batch, seq_len, n_channels) where seq_len is number of timesteps


### Training Strategy

1. **Start Simple**: Baseline → Traditional ML → LSTM/TCN
2. **Patient-Specific Models**: Train separate models per subject (as literature recommends)
   - Seizure patterns vary significantly across patients
   - Better performance but less generalizable
   
3. **Loss Functions**:
   - Classification: Binary cross-entropy with class weights
   - Focal loss (if standard BCE doesn't converge): `FL(pt) = -α(1-pt)^γ log(pt)`
   - Addresses class imbalance more explicitly than weighting

4. **Optimization**:
   - Adam optimizer with learning rate 1e-3 to 1e-4
   - Early stopping based on validation metric (AUC-ROC preferred over accuracy)

---

## Validation Phase

### Hyperparameter Tuning Strategy

Validate on a held-out validation set (e.g., 20% of data for each patient) **before** final test evaluation.

#### Hyperparameters by Model Type

**Random Forest**:
- `n_estimators`: [100, 300, 500]
- `max_depth`: [10, 20, None]
- `min_samples_split`: [2, 5, 10]
- `min_samples_leaf`: [1, 2, 4]
- Method: GridSearchCV with stratified 5-fold cross-validation

**SVM (RBF Kernel)**:
- `C`: [0.1, 1, 10, 100]
- `gamma`: [0.001, 0.01, 0.1, 'scale']
- Method: GridSearchCV with stratified 5-fold CV
- Note: Scale features before SVM

**LSTM/GRU**:
- `hidden_size`: [64, 128, 256, 512]
- `num_layers`: [1, 2, 3]
- `dropout`: [0.1, 0.3, 0.5]
- `learning_rate`: [1e-2, 1e-3, 1e-4]
- `batch_size`: [16, 32, 64]
- Method: Random search (more efficient than grid) or Optuna for Bayesian optimization

**Temporal CNN**:
- `kernel_size`: [3, 5, 7, 15]
- `num_filters`: [32, 64, 128, 256]
- `dilation`: [1, 2, 4]
- `learning_rate`: [1e-2, 1e-3, 1e-4]
- Method: Similar to LSTM; use multiple kernel sizes in parallel

### Early Stopping

For neural networks:
```python
# Stop training if validation AUC doesn't improve for N epochs
early_stopping = EarlyStopping(metric='auc', patience=15, mode='max')
```

Monitor validation loss/metrics every epoch; save best model weights.

---

## Test Phase

### Test Set Evaluation

**Setup**:
- Hold out a completely independent test set (e.g., last subject or specific time period per patient)
- No information from test set used in preprocessing, feature engineering, or hyperparameter tuning
- Test on raw, unseen EEG data with same preprocessing as train/val

**Final Metrics** (report on test set):
- ROC curve and AUC
- Precision-recall curve and AUC
- Confusion matrix
- Classification report (precision, recall, F1 per class)
- Sensitivity, specificity, and balanced accuracy
- Prediction distribution (histogram of predicted probabilities)

### Model Comparison -> example of how we intend to compare models. 

Create a comparison table:

| Model | ROC-AUC | F1-Score | Sensitivity | Specificity | Train Time | Inference Time |
|-------|---------|----------|-------------|-------------|-----------|-----------------|
| Random Forest | 0.85 | 0.72 | 0.80 | 0.84 | 5s | 0.01s |
| SVM | 0.87 | 0.74 | 0.82 | 0.86 | 120s | 0.05s |
| LSTM | 0.90 | 0.78 | 0.85 | 0.89 | 300s | 0.02s |
| Temporal CNN | 0.91 | 0.79 | 0.87 | 0.90 | 250s | 0.01s |

### Error Analysis -> this depends on how we handle the class imbalances by the time we have chosen our learning approach

- **False Positives**: When does model incorrectly predict seizure? (Patient burden)
- **False Negatives**: When does model miss real seizures? (Safety risk)
- **Per-Patient Performance**: Does one model work well for all patients or are patient-specific models needed?
- **Temporal Patterns**: Do errors cluster at specific times (e.g., sleep vs. awake)?

## Running the Project

### Quick Start

```bash
# 1. Download dataset (test mode first)
python data/scripts/download_data.py --test
python data/scripts/download_data.py --setup-dirs  # Full download

# 2. Test preprocessing pipeline
python data/scripts/convert_edf_to_csv.py --test-loading

# 3. Convert EDF to ML-ready CSV
python data/scripts/convert_edf_to_csv.py

# 4. Run analysis
jupyter notebook notebooks/01_exploratory_analysis.ipynb
```

## References

### Dataset
- Detti, P. (2020). Siena Scalp EEG Database (version 1.0.0). *PhysioNet*. https://doi.org/10.13026/5d4a-j060
- Original paper: Detti et al. (2020). EEG Synchronization Analysis for Seizure Prediction. *Processes*, 8(7), 846.

### EEG Signal Processing
- MNE-Python documentation: https://mne.tools/
- Teplan, M. (2002). Fundamentals of EEG measurement. *Measurement Science Review*, 2(2), 1-11.

### Seizure Prediction Literature
- Mormann et al. (2007). Seizure prediction: the long and winding road. *Brain*, 130(2), 314-326.
- Khan & Gotman (2003). Comprehensive seizure detection hardware implementations. *IEEE Engineering in Medicine and Biology Magazine*, 22(1), 75-89.


