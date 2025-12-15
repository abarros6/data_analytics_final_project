# EEG Seizure Prediction - 2-Day Execution Plan

## Project Overview
- **Goal**: Binary classification to predict preictal (pre-seizure) vs interictal states
- **Dataset**: Siena Scalp EEG Database - 20GB, 14 patients, 47 seizures, 512Hz sampling
- **Main Challenge**: Massive feature space (445,440 features/window) + severe class imbalance + hardware constraints
- **Timeline**: 2 days maximum

## Hardware Constraints Considerations
- **Memory Limitation**: Avoid loading full raw dataset (20GB) into memory
- **Processing Strategy**: Patient-by-patient processing, chunked data loading
- **Feature Reduction**: Use engineered features instead of raw samples (99.9% reduction)
- **Model Selection**: Focus on lightweight, efficient models

---

## Sequential Action Items (Commit-Based)

### ✅ COMMIT 1: Environment Setup & Data Validation
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Tasks**:
- [ ] Install project dependencies (`pip install -r requirements.txt`)
- [ ] Test data loading pipeline on 1-2 subjects only
- [ ] Verify EDF files are accessible and readable  
- [ ] Document any missing files or data issues
- [ ] Estimate memory usage for single patient processing

**Success Criteria**: 
- Dependencies installed without errors
- Can successfully load and process at least 1 patient's EDF data
- Memory usage documented and manageable

**Expected Output**: 
- Working environment
- Data validation report
- Memory usage baseline

---

### ✅ COMMIT 2: Memory-Efficient Feature Engineering
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Tasks**:
- [ ] Implement spectral power features (5 frequency bands × ~17 channels = ~85 features)
  - Delta (0.5-4 Hz), Theta (4-8 Hz), Alpha (8-12 Hz), Beta (12-30 Hz), Gamma (30-40 Hz)
- [ ] Add statistical features (mean, std, skew, kurtosis × channels = ~68 features)
- [ ] Replace raw sample processing with engineered features in preprocessing pipeline
- [ ] Test memory usage reduction vs original approach
- [ ] Verify feature extraction on 2-3 patients

**Success Criteria**:
- Feature count reduced from 445,440 to ~150-200 per window
- Memory usage dramatically reduced (target: <2GB for single patient)
- Features properly normalized and formatted

**Expected Output**:
- Modified `convert_edf_to_csv.py` with feature engineering
- Processed feature dataset for 2-3 patients
- Memory usage comparison report

---

### ✅ COMMIT 3: Baseline Model Implementation  
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Tasks**:
- [ ] Implement Random Forest classifier with `class_weight='balanced'`
- [ ] Add proper train/validation split logic (stratified, patient-aware)
- [ ] Handle severe class imbalance (~2-5% preictal samples)
- [ ] Generate initial performance metrics (ROC-AUC, precision, recall, F1)
- [ ] Test on 3-4 patients to validate approach

**Success Criteria**:
- Random Forest trains successfully on imbalanced data
- ROC-AUC > 0.7 (reasonable baseline for this problem)
- Model handles class imbalance appropriately

**Expected Output**:
- Working baseline model with evaluation metrics
- Training script for Random Forest
- Initial performance results

---

### ✅ COMMIT 4: Neural Network Comparison Model
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete  

**Tasks**:
- [ ] Implement lightweight 1D CNN using engineered features as input
- [ ] Alternative: Simple 2-layer LSTM if CNN doesn't work well
- [ ] Use same train/validation splits as Random Forest
- [ ] Compare performance against Random Forest baseline
- [ ] Optimize for memory efficiency (small batch sizes)

**Success Criteria**:
- Neural network trains without memory errors
- Performance comparison completed vs Random Forest
- Model inference time reasonable for real-time application

**Expected Output**:
- Neural network implementation 
- Performance comparison table
- Training time and inference time benchmarks

---

### ✅ COMMIT 5: Evaluation Framework & Results
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Tasks**:
- [ ] Implement comprehensive evaluation metrics (ROC-AUC, PR-AUC, confusion matrix)
- [ ] Generate model comparison table with key metrics
- [ ] Create visualizations: ROC curves, precision-recall curves, confusion matrices  
- [ ] Analyze per-patient performance (patient-specific vs general model)
- [ ] Error analysis: characterize false positives/negatives

**Success Criteria**:
- Clear model comparison with statistical significance
- Professional visualizations of results
- Insights into model strengths/weaknesses

**Expected Output**:
- Evaluation notebook with all metrics and plots
- Model comparison summary
- Error analysis findings

---

### ✅ COMMIT 6: Final Report & Documentation
**Status**: [ ] Not Started | [ ] In Progress | [ ] Complete

**Tasks**:
- [ ] Write executive summary of results and findings
- [ ] Document methodology and key decisions made
- [ ] Discuss limitations and hardware constraint impacts
- [ ] Provide recommendations for future work/improvements
- [ ] Create final presentation-ready materials

**Success Criteria**:
- Complete project documentation
- Clear presentation of results and conclusions
- Professional quality deliverable

**Expected Output**:
- Final report document
- Key findings summary
- Presentation materials

---

## Risk Mitigation Strategies

### Memory/Hardware Risks:
- **Fallback 1**: If full feature extraction fails, use only spectral features (~85 features)
- **Fallback 2**: If neural networks too memory-intensive, focus on Random Forest optimization
- **Fallback 3**: If processing all patients impossible, demonstrate on subset (5-6 patients)

### Time Constraints:
- **Priority 1**: Get baseline working (Commits 1-3) - Day 1 
- **Priority 2**: Neural network comparison (Commit 4) - Day 2 morning
- **Priority 3**: Results and documentation (Commits 5-6) - Day 2 afternoon

### Model Performance Risks:
- **Expectation**: ROC-AUC 0.7-0.85 is reasonable for this challenging problem
- **Class imbalance**: Accept that high precision may be challenging due to 95%+ majority class
- **Focus**: Emphasize recall (sensitivity) for seizure prediction safety

---

## Current Status: Ready to Begin
**Next Action**: Proceed with COMMIT 1 - Environment Setup & Data Validation