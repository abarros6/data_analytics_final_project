#!/usr/bin/env python3
"""
Variable Preictal Window Processor
==================================

Generate datasets with different preictal window lengths to test 
the impact on seizure prediction performance.

Window lengths tested: 30s, 60s, 120s, 300s
"""

import sys
import os
sys.path.append('.')
from scripts.seizure_data_processor import parse_patient_seizures, process_seizure_file
import pandas as pd
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

def process_with_window_length(preictal_window_sec):
    """Process seizure data with specific preictal window length"""
    print(f'🎯 Processing with {preictal_window_sec}s preictal window')
    print('='*50)
    
    data_dir = Path('data/raw/physionet.org/files/siena-scalp-eeg/1.0.0')
    patients = ['PN00', 'PN01', 'PN03', 'PN05', 'PN06']
    
    all_windows = []
    total_seizures = 0
    
    for patient in patients:
        print(f'📁 Processing {patient}...')
        patient_dir = data_dir / patient
        
        # Parse seizures for this patient
        seizures = parse_patient_seizures(patient_dir, patient)
        total_seizures += len(seizures)
        
        if not seizures:
            print(f"  No seizures found for {patient}")
            continue
        
        # Process each seizure-containing file with custom window
        for seizure_info in seizures:
            windows = process_seizure_file(patient_dir, patient, seizure_info, preictal_window_sec)
            all_windows.extend(windows)
    
    if all_windows:
        df = pd.DataFrame(all_windows)
        output_file = f'data/processed/seizure_data_{preictal_window_sec}s.csv'
        df.to_csv(output_file, index=False)
        
        # Label analysis
        preictal_count = df['label'].sum()
        interictal_count = len(df) - preictal_count
        preictal_pct = (preictal_count / len(df) * 100) if len(df) > 0 else 0
        
        print(f'\n✅ PROCESSING COMPLETE: {preictal_window_sec}s window')
        print(f'Output: {output_file}')
        print(f'Total windows: {len(df)}')
        print(f'Preictal: {preictal_count} ({preictal_pct:.1f}%)')
        print(f'Interictal: {interictal_count} ({100-preictal_pct:.1f}%)')
        print()
        
        return output_file, len(df), preictal_count, interictal_count
    else:
        print(f'❌ No windows extracted for {preictal_window_sec}s')
        return None, 0, 0, 0

def main():
    """Generate datasets for multiple preictal window lengths"""
    print('🔬 VARIABLE PREICTAL WINDOW ANALYSIS')
    print('====================================')
    print('Testing different preictal window lengths for seizure prediction')
    print()
    
    # Test different window lengths
    window_lengths = [30, 60, 120, 300]  # 30s, 1min, 2min, 5min
    
    results = []
    
    for window_sec in window_lengths:
        output_file, total_windows, preictal, interictal = process_with_window_length(window_sec)
        if output_file:
            results.append({
                'window_length_sec': window_sec,
                'output_file': output_file,
                'total_windows': total_windows,
                'preictal_windows': preictal,
                'interictal_windows': interictal,
                'preictal_percentage': (preictal / total_windows * 100) if total_windows > 0 else 0
            })
    
    # Summary
    print('📊 SUMMARY OF GENERATED DATASETS')
    print('=================================')
    for result in results:
        print(f"Window: {result['window_length_sec']:3d}s | "
              f"Total: {result['total_windows']:3d} | "
              f"Preictal: {result['preictal_windows']:3d} ({result['preictal_percentage']:4.1f}%) | "
              f"File: {result['output_file']}")
    
    print(f'\n✅ Generated {len(results)} datasets for preictal window analysis')
    print('Ready for model comparison across different window lengths!')

if __name__ == '__main__':
    main()