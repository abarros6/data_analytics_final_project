# Development Decisions Log
**Project**: EEG Seizure Prediction  
**Timeline**: 2-Day Sprint  
**Date**: December 15, 2025

## Critical Decisions Made

### 1. Hardware Constraint Management
**Decision**: Use subset of patients (2-3 instead of 14)  
**Reason**: 
- Full dataset: 20GB (~6.3GB EDF files + processing overhead)
- Each patient has ~5 EDF files at 90MB each
- Raw feature space: 445,440 features per 30s window
- Memory requirements exceed available hardware for 2-day sprint

**Impact**: Reduced scope but allows proof-of-concept completion
**Team Discussion Needed**: Scale-up strategy for full dataset

### 2. Feature Engineering Priority
**Decision**: Implement engineered features instead of raw samples  
**Reason**: 
- Raw approach: 445,440 features per window
- Engineered approach: ~150 features per window (99.9% reduction)
- Spectral power (5 bands × ~17 channels = 85 features)
- Statistical features (4 stats × ~17 channels = 68 features)

**Impact**: Massive memory reduction, faster training
**Team Discussion Needed**: Validate feature selection maintains predictive power

### 3. Patient Selection Strategy
**Decision**: Start with PN00, PN01, PN03 (patients with confirmed seizures)  
**Reason**:
- PN00: 4 seizures documented
- PN01: 1 seizure documented  
- PN03: 3 seizures documented
- Provides sufficient positive examples for model training

**Impact**: Balanced dataset for initial model development
**Team Discussion Needed**: Patient selection criteria for full implementation

### 4. Technical Stack Decisions
**Decision**: Python 3.12 with virtual environment  
**Dependencies**: 
- MNE (v1.11.0) for EEG processing
- scikit-learn (v1.8.0) for baseline models
- PyTorch (v2.2.2) for neural networks
- imbalanced-learn for SMOTE handling

**Impact**: Consistent development environment
**Team Discussion Needed**: Production deployment requirements

### 5. Model Development Strategy
**Decision**: Baseline-first approach  
**Priority Order**:
1. Random Forest (fast, interpretable, handles imbalance well)
2. Lightweight 1D CNN (comparison model)
3. Skip complex architectures (LSTM, Transformers) due to time constraint

**Impact**: Focus on working solution over advanced techniques
**Team Discussion Needed**: Advanced model requirements for final submission

### 6. Evaluation Approach
**Decision**: Patient-aware splits with imbalance handling  
**Strategy**:
- Stratified splits maintaining ~95% interictal, ~5% preictal ratio
- Focus on ROC-AUC and precision-recall metrics
- Emphasize recall (sensitivity) for seizure safety

**Impact**: Realistic evaluation mimicking clinical deployment
**Team Discussion Needed**: Performance thresholds for acceptance

## Data Pipeline Modifications

### Original Plan vs. Implemented
| Aspect | Original Plan | 2-Day Implementation | Rationale |
|--------|---------------|---------------------|-----------|
| Dataset Size | Full 14 patients | 3 patients | Hardware constraints |
| Features | Raw samples (445K) | Engineered (~150) | Memory efficiency |
| Window Size | 30 seconds | 30 seconds | Maintained |
| Models | RF + SVM + LSTM + CNN | RF + 1D CNN | Time constraints |
| Validation | Cross-validation | Train/Val split | Simplified for sprint |

## Risk Mitigation Strategies

### Identified Risks & Solutions
1. **Memory Issues**: Feature engineering reduces memory 1000x
2. **Class Imbalance**: Weighted loss functions, SMOTE if needed
3. **Time Constraints**: Baseline-first, skip complex architectures
4. **Data Quality**: Start with well-documented patients (PN00-03)
5. **Reproducibility**: Virtual environment + detailed documentation

## File Structure Changes

### Added Files
- `setup_venv.sh`: Virtual environment setup script
- `PROJECT_EXECUTION_PLAN.md`: Sprint planning document
- `DEVELOPMENT_DECISIONS.md`: This decision log

### Modified Pipeline
- `convert_edf_to_csv.py`: Will be modified for feature engineering
- Data directory: Focus on `PN00/`, `PN01/`, `PN03/` subdirectories

## Next Steps for Team Review

### Immediate Actions (Today)
1. Review hardware constraint decisions
2. Approve patient subset selection
3. Validate feature engineering approach

### Future Considerations
1. **Full Dataset Strategy**: How to scale to all 14 patients
2. **Advanced Models**: Timeline for LSTM/Transformer implementation
3. **Production Deployment**: Infrastructure requirements
4. **Performance Targets**: Minimum acceptable ROC-AUC thresholds

## Questions for Team Discussion

1. **Scope**: Is 3-patient proof-of-concept sufficient for project requirements?
2. **Features**: Should we validate engineered features against literature?
3. **Models**: Is Random Forest + CNN sufficient, or do we need LSTM?
4. **Timeline**: Can we extend beyond 2 days if needed?
5. **Hardware**: Do we have access to higher-capacity machines for full dataset?

---

**Documentation maintained by**: Claude Code Assistant  
**Last updated**: December 15, 2025  
**Status**: Active development - decisions subject to team review