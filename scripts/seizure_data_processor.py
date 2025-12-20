#!/usr/bin/env python3
"""
Real Seizure Targeted Processor - Process ACTUAL seizure periods only
====================================================================

This script processes ONLY the EDF segments that contain documented seizures
to capture real preictal periods. No fabricated data, no random labels.

Strategy:
1. Parse seizure timing from PhysioNet annotations
2. Process ONLY the specific EDF files containing seizures  
3. Extract windows around actual seizure times (preictal + interictal)
4. Label based on real seizure occurrence (60 sec before = preictal)
"""

import mne
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import skew, kurtosis
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

def parse_time_to_seconds(time_str):
    """Convert time string to seconds from midnight"""
    time_str = time_str.replace(':', '.').strip()
    parts = time_str.split('.')
    if len(parts) == 3:
        h, m, s = map(int, parts)
        return h * 3600 + m * 60 + s
    return None

def parse_patient_seizures(patient_dir, patient_id):
    """Parse all seizures for a patient with file mapping"""
    seizure_file = patient_dir / f"Seizures-list-{patient_id}.txt"
    seizures = []
    
    if not seizure_file.exists():
        print(f"  No seizure file for {patient_id}")
        return seizures
    
    with open(seizure_file, 'r') as f:
        lines = f.readlines()
    
    current_seizure = {}
    for line in lines:
        line = line.strip()
        if 'File name:' in line:
            current_seizure['edf_file'] = line.split(':')[1].strip()
        elif 'Registration start time:' in line:
            time_str = line.split(':', 1)[1].strip()
            current_seizure['reg_start'] = parse_time_to_seconds(time_str)
        elif 'Seizure start time:' in line or 'Start time:' in line:
            time_str = line.split(':', 1)[1].strip()
            seizure_start = parse_time_to_seconds(time_str)
            if seizure_start and current_seizure.get('reg_start'):
                # Calculate relative to recording start  
                relative_time = seizure_start - current_seizure['reg_start']
                if relative_time < 0:  # Handle day rollover
                    relative_time += 24 * 3600
                current_seizure['seizure_start_sec'] = relative_time
        elif 'Seizure end time:' in line or 'End time:' in line:
            time_str = line.split(':', 1)[1].strip()
            seizure_end = parse_time_to_seconds(time_str)
            if seizure_end and current_seizure.get('reg_start'):
                relative_time = seizure_end - current_seizure['reg_start']
                if relative_time < 0:  # Handle day rollover
                    relative_time += 24 * 3600
                current_seizure['seizure_end_sec'] = relative_time
                
                # Complete seizure entry
                if all(k in current_seizure for k in ['edf_file', 'seizure_start_sec', 'seizure_end_sec']):
                    seizures.append({
                        'edf_file': current_seizure['edf_file'],
                        'start_sec': current_seizure['seizure_start_sec'],
                        'end_sec': current_seizure['seizure_end_sec']
                    })
                    print(f"  → Seizure in {current_seizure['edf_file']}: {current_seizure['seizure_start_sec']:.0f}-{current_seizure['seizure_end_sec']:.0f}s")
                current_seizure = {}
    
    return seizures

def extract_features(data, sfreq):
    """Extract features from EEG window"""
    features = []
    for ch_data in data:
        if np.std(ch_data) > 1e-10:
            features.extend([
                np.mean(ch_data),
                np.std(ch_data), 
                skew(ch_data),
                kurtosis(ch_data)
            ])
        else:
            features.extend([0, 0, 0, 0])
    return features

def process_seizure_file(patient_dir, patient_id, seizure_info, preictal_window_sec=60):
    """Process specific EDF file containing seizure"""
    edf_path = patient_dir / seizure_info['edf_file']
    if not edf_path.exists():
        print(f"    EDF file not found: {seizure_info['edf_file']}")
        return []
    
    print(f"    Processing {seizure_info['edf_file']} (seizure at {seizure_info['start_sec']:.0f}s)")
    
    try:
        # Load EDF file
        raw = mne.io.read_raw_edf(str(edf_path), preload=False, verbose=False)
        
        # Get EEG channels
        eeg_channels = [ch for ch in raw.ch_names if 'EEG' in ch or ch in ['Fp1', 'F3', 'C3', 'P3', 'O1', 'F7', 'T3', 'T5']][:10]
        if len(eeg_channels) == 0:
            print(f"    No EEG channels found in {seizure_info['edf_file']}")
            return []
        
        raw.pick_channels(eeg_channels)
        
        # Define time window around seizure (10 minutes before to 5 minutes after)
        seizure_start = seizure_info['start_sec']
        window_start = max(0, seizure_start - 600)  # 10 min before
        window_end = min(raw.times[-1], seizure_start + 300)   # 5 min after
        
        # Crop to target window
        raw.crop(tmin=window_start, tmax=window_end)
        raw.load_data()
        
        # Basic filtering
        if raw.info['sfreq'] != 256:
            raw.resample(256)
        raw.filter(1, 30, verbose=False)
        
        # Extract 4-second windows
        window_samples = int(4 * 256)
        windows = []
        
        for start_sample in range(0, len(raw.times) - window_samples, window_samples):
            window_start_time = raw.times[start_sample] + window_start  # Absolute time in recording
            window_end_time = raw.times[start_sample + window_samples] + window_start
            window_center = (window_start_time + window_end_time) / 2
            
            # Extract features
            window_data = raw.get_data()[:, start_sample:start_sample+window_samples]
            features = extract_features(window_data, 256)
            
            # Real seizure labeling
            # Preictal: variable seconds before seizure start
            preictal_start = seizure_start - preictal_window_sec
            preictal_end = seizure_start
            
            if preictal_start <= window_center < preictal_end:
                label = 1  # Preictal
            else:
                label = 0  # Interictal
                
            windows.append({
                'subject_id': patient_id,
                'edf_file': seizure_info['edf_file'],
                'window_start_sec': window_start_time,
                'seizure_start_sec': seizure_start,
                'label': label,
                **{f'feature_{i}': f for i, f in enumerate(features)}
            })
        
        print(f"    Extracted {len(windows)} windows, seizure at {seizure_start:.0f}s")
        return windows
        
    except Exception as e:
        print(f"    Error processing {seizure_info['edf_file']}: {e}")
        return []

def main():
    print('🎯 REAL SEIZURE TARGETED PROCESSOR')
    print('==================================')
    print('Processing ACTUAL seizure periods only - no fabricated data')
    print()
    
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
        
        # Process each seizure-containing file
        for seizure_info in seizures:
            windows = process_seizure_file(patient_dir, patient, seizure_info, preictal_window_sec=60)
            all_windows.extend(windows)
    
    if all_windows:
        df = pd.DataFrame(all_windows)
        output_file = 'data/processed/real_seizure_targeted_data.csv'
        df.to_csv(output_file, index=False)
        
        print(f'\n✅ REAL SEIZURE PROCESSING COMPLETE')
        print(f'═════════════════════════════════════')
        print(f'Output: {output_file}')
        print(f'Total seizures processed: {total_seizures}')
        print(f'Total windows extracted: {len(df)}')
        print(f'Features per window: {len([c for c in df.columns if c.startswith("feature_")])}')
        print()
        
        # Label analysis
        preictal_count = df['label'].sum()
        interictal_count = len(df) - preictal_count
        preictal_pct = (preictal_count / len(df) * 100) if len(df) > 0 else 0
        
        print(f'🏷️  REAL SEIZURE LABELING RESULTS:')
        print(f'   Preictal windows: {preictal_count} ({preictal_pct:.1f}%)')
        print(f'   Interictal windows: {interictal_count} ({100-preictal_pct:.1f}%)')
        print()
        
        # Per-patient breakdown
        print(f'📊 PER-PATIENT BREAKDOWN:')
        for patient in patients:
            patient_data = df[df['subject_id'] == patient]
            if len(patient_data) > 0:
                p_preictal = patient_data['label'].sum()
                p_total = len(patient_data)
                p_pct = (p_preictal / p_total * 100) if p_total > 0 else 0
                print(f'   {patient}: {p_total} windows, {p_preictal} preictal ({p_pct:.1f}%)')
        
        print()
        print('✅ SUCCESS: Real seizure prediction data ready')
        print('   • Uses actual PhysioNet seizure timing')
        print('   • Contains real preictal periods (60 sec before seizure)')
        print('   • No fabricated or random labels')
        print('   • Ready for genuine seizure prediction modeling')
        
    else:
        print('❌ No seizure windows extracted - check seizure timing and EDF files')

if __name__ == '__main__':
    main()