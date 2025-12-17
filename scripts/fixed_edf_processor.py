#!/usr/bin/env python3
"""
Fixed EDF to CSV Converter with Robust Seizure Detection
========================================================

Fixes critical bugs in seizure detection and implements proper data validation.

Key Fixes:
1. Proper day rollover handling for seizure times
2. Comprehensive data validation
3. All available patients processing
4. Quality checks and error reporting
"""

import os
import sys
import csv
from pathlib import Path
import time
from datetime import datetime, timedelta
import argparse
import mne
import numpy as np
import pandas as pd
from scipy import stats
from scipy.signal import welch
from scipy.stats import skew, kurtosis
from typing import List, Dict, Tuple, Optional

class FixedEEGProcessor:
    def __init__(self, config=None):
        self.config = config or {
            'input_dir': 'data/raw/physionet.org/files/siena-scalp-eeg/1.0.0',
            'output_dir': 'data/processed',
            'window_size_sec': 4,
            'overlap': 0.5,  # 50% overlap
            'preictal_window_sec': 60,  # 1 minute before seizure
            'filter_low_freq': 0.5,
            'filter_high_freq': 40,
            'target_sfreq': 256
        }
        self.stats = {
            'patients_processed': 0,
            'total_windows': 0,
            'preictal_windows': 0,
            'interictal_windows': 0,
            'seizures_detected': 0,
            'errors': [],
            'processing_times': {}
        }
        
    def log(self, message, level='INFO'):
        """Enhanced logging with timestamps"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_entry = f"[{timestamp}] {level}: {message}"
        print(log_entry)
        
    def find_all_patients(self) -> List[str]:
        """Find all available patients in dataset"""
        dataset_dir = Path(self.config['input_dir'])
        
        if not dataset_dir.exists():
            self.log(f"Dataset directory not found: {dataset_dir}", 'ERROR')
            return []
        
        patients = []
        for item in dataset_dir.iterdir():
            if item.is_dir() and item.name.startswith('PN'):
                # Check if has EDF files
                edf_files = list(item.glob('*.edf'))
                if edf_files:
                    patients.append(item.name)
                else:
                    self.log(f"Skipping {item.name}: No EDF files")
        
        patients.sort()
        self.log(f"Found {len(patients)} patients with EDF files: {patients}")
        return patients
    
    def parse_time_with_date_handling(self, time_str: str, base_date: datetime = None) -> datetime:
        """
        Parse time string with proper date handling for day rollover
        
        Args:
            time_str: Time in format "HH.MM.SS" or "HH:MM:SS"
            base_date: Base date for calculations
            
        Returns:
            datetime object
        """
        # Normalize separators
        time_str = time_str.replace(':', '.')
        
        try:
            parts = time_str.split('.')
            if len(parts) == 3:
                h, m, s = map(int, parts)
                
                # Create time on base date
                if base_date is None:
                    base_date = datetime(2000, 1, 1)  # Arbitrary base date
                
                time_obj = base_date.replace(hour=h, minute=m, second=s, microsecond=0)
                return time_obj
                
        except Exception as e:
            self.log(f"Error parsing time '{time_str}': {e}", 'ERROR')
            
        return None
    
    def parse_seizure_times_fixed(self, seizure_file: Path) -> Dict[str, List[Tuple[float, float]]]:
        """
        Fixed seizure time parsing with proper day rollover handling
        
        Returns:
            Dictionary mapping EDF filenames to list of (start_sec, end_sec) tuples
        """
        file_seizures = {}
        
        if not seizure_file.exists():
            self.log(f"Seizure file not found: {seizure_file}", 'WARNING')
            return file_seizures
        
        try:
            with open(seizure_file, 'r') as f:
                lines = f.readlines()
            
            current_file = None
            reg_start_time = None
            reg_end_time = None
            base_date = datetime(2000, 1, 1)  # Arbitrary base date
            
            # Parse file info first
            for line in lines:
                line = line.strip()
                
                if "File name:" in line:
                    current_file = line.split("File name:")[1].strip()
                    if current_file not in file_seizures:
                        file_seizures[current_file] = []
                        
                elif "Registration start time:" in line:
                    reg_start_str = line.split("Registration start time:")[1].strip()
                    reg_start_time = self.parse_time_with_date_handling(reg_start_str, base_date)
                    
                elif "Registration end time:" in line:
                    reg_end_str = line.split("Registration end time:")[1].strip()
                    reg_end_time = self.parse_time_with_date_handling(reg_end_str, base_date)
                    
                    # Handle day rollover - if end time < start time, add 1 day
                    if reg_end_time and reg_start_time and reg_end_time < reg_start_time:
                        reg_end_time += timedelta(days=1)
                        self.log(f"Detected day rollover for {current_file}")
            
            # Parse seizures
            seizure_start_time = None
            seizure_num = 0
            
            for line in lines:
                line = line.strip()
                
                if "Seizure n" in line or "Start time:" in line:
                    if "Start time:" in line:
                        seizure_start_str = line.split("Start time:")[1].strip()
                        seizure_start_time = self.parse_time_with_date_handling(seizure_start_str, base_date)
                        seizure_num += 1
                        
                elif ("End time:" in line or "Seizure end time:" in line) and seizure_start_time and current_file:
                    if "End time:" in line:
                        seizure_end_str = line.split("End time:")[1].strip()
                    else:
                        seizure_end_str = line.split("Seizure end time:")[1].strip()
                        
                    seizure_end_time = self.parse_time_with_date_handling(seizure_end_str, base_date)
                    
                    if seizure_end_time and reg_start_time:
                        # Handle day rollover for seizure times
                        if seizure_start_time < reg_start_time:
                            seizure_start_time += timedelta(days=1)
                        if seizure_end_time < seizure_start_time:
                            seizure_end_time += timedelta(days=1)
                        
                        # Convert to seconds relative to recording start
                        start_rel = (seizure_start_time - reg_start_time).total_seconds()
                        end_rel = (seizure_end_time - reg_start_time).total_seconds()
                        
                        if start_rel >= 0 and end_rel > start_rel:
                            file_seizures[current_file].append((start_rel, end_rel))
                            self.log(f"Seizure {seizure_num}: {start_rel:.1f}s to {end_rel:.1f}s")
                            self.stats['seizures_detected'] += 1
                        else:
                            self.log(f"Invalid seizure timing: {start_rel:.1f}s to {end_rel:.1f}s", 'WARNING')
                    
                    seizure_start_time = None  # Reset for next seizure
                    
        except Exception as e:
            error_msg = f"Error parsing seizure file {seizure_file}: {e}"
            self.log(error_msg, 'ERROR')
            self.stats['errors'].append(error_msg)
        
        return file_seizures
    
    def validate_eeg_data(self, raw: mne.io.Raw) -> Dict[str, any]:
        """
        Comprehensive EEG data quality validation
        
        Returns:
            Dictionary with validation results
        """
        validation = {
            'valid': True,
            'issues': [],
            'channel_stats': {},
            'sampling_rate': raw.info['sfreq'],
            'duration': len(raw.times) / raw.info['sfreq'],
            'n_channels': len(raw.ch_names)
        }
        
        # Check sampling rate
        if raw.info['sfreq'] < 100:
            validation['issues'].append(f"Low sampling rate: {raw.info['sfreq']} Hz")
        
        # Check duration
        if validation['duration'] < 60:  # Less than 1 minute
            validation['issues'].append(f"Very short recording: {validation['duration']:.1f}s")
        
        # Check each channel
        data = raw.get_data()
        for i, ch_name in enumerate(raw.ch_names):
            ch_data = data[i, :]
            
            ch_stats = {
                'mean': np.mean(ch_data),
                'std': np.std(ch_data),
                'min': np.min(ch_data),
                'max': np.max(ch_data),
                'zero_variance': np.std(ch_data) == 0,
                'constant_value': len(np.unique(ch_data)) == 1,
                'extreme_values': np.any(np.abs(ch_data) > 1000)  # 1000 µV threshold
            }
            
            validation['channel_stats'][ch_name] = ch_stats
            
            # Check for problematic channels
            if ch_stats['zero_variance']:
                validation['issues'].append(f"Channel {ch_name}: Zero variance (constant)")
            elif ch_stats['extreme_values']:
                validation['issues'].append(f"Channel {ch_name}: Extreme values detected")
            elif ch_stats['std'] < 0.1:
                validation['issues'].append(f"Channel {ch_name}: Very low variance ({ch_stats['std']:.3f})")
        
        if validation['issues']:
            validation['valid'] = False
            
        return validation
    
    def extract_window_features(self, window_data: np.ndarray, sfreq: float, ch_names: List[str]) -> Dict[str, float]:
        """
        Extract comprehensive features from EEG window
        """
        features = {}
        n_channels, n_timepoints = window_data.shape
        
        # Frequency bands
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 40)
        }
        
        for ch_idx, ch_name in enumerate(ch_names):
            if ch_idx >= n_channels:
                continue
                
            channel_data = window_data[ch_idx, :]
            
            # Skip if channel has issues
            if np.std(channel_data) == 0:
                # Set default values for constant channels
                for feature_type in ['mean', 'std', 'skew', 'kurtosis', 'rms', 'peak_to_peak', 'zero_crossings']:
                    features[f'{ch_name}_{feature_type}'] = 0.0
                for band in bands:
                    features[f'{ch_name}_{band}_power'] = 0.0
                    features[f'{ch_name}_{band}_rel'] = 0.0
                continue
            
            # Time domain features
            features[f'{ch_name}_mean'] = float(np.mean(channel_data))
            features[f'{ch_name}_std'] = float(np.std(channel_data))
            features[f'{ch_name}_skew'] = float(skew(channel_data))
            features[f'{ch_name}_kurtosis'] = float(kurtosis(channel_data))
            features[f'{ch_name}_rms'] = float(np.sqrt(np.mean(channel_data**2)))
            features[f'{ch_name}_peak_to_peak'] = float(np.ptp(channel_data))
            features[f'{ch_name}_zero_crossings'] = float(len(np.where(np.diff(np.signbit(channel_data)))[0]))
            
            # Frequency domain features
            try:
                # Use appropriate window size for PSD
                nperseg = min(256, n_timepoints // 4)
                if nperseg < 16:  # Too short for reliable PSD
                    nperseg = min(16, n_timepoints)
                
                freqs, psd = welch(channel_data, fs=sfreq, nperseg=nperseg)
                
                # Power in frequency bands
                band_powers = {}
                for band_name, (low_freq, high_freq) in bands.items():
                    mask = (freqs >= low_freq) & (freqs < high_freq)
                    if np.any(mask):
                        band_power = np.trapz(psd[mask], freqs[mask])
                    else:
                        band_power = 0.0
                    
                    band_powers[band_name] = float(band_power)
                    features[f'{ch_name}_{band_name}_power'] = band_power
                
                # Relative power
                total_power = sum(band_powers.values())
                if total_power > 0:
                    for band_name, power in band_powers.items():
                        features[f'{ch_name}_{band_name}_rel'] = float(power / total_power)
                else:
                    for band_name in bands:
                        features[f'{ch_name}_{band_name}_rel'] = 0.0
                
                # Spectral features
                if np.sum(psd) > 0:
                    features[f'{ch_name}_spectral_centroid'] = float(np.sum(freqs * psd) / np.sum(psd))
                    centroid = features[f'{ch_name}_spectral_centroid']
                    features[f'{ch_name}_spectral_bandwidth'] = float(
                        np.sqrt(np.sum(((freqs - centroid) ** 2) * psd) / np.sum(psd))
                    )
                else:
                    features[f'{ch_name}_spectral_centroid'] = 0.0
                    features[f'{ch_name}_spectral_bandwidth'] = 0.0
                    
            except Exception as e:
                self.log(f"Error computing frequency features for {ch_name}: {e}", 'WARNING')
                # Set default values
                for band in bands:
                    features[f'{ch_name}_{band}_power'] = 0.0
                    features[f'{ch_name}_{band}_rel'] = 0.0
                features[f'{ch_name}_spectral_centroid'] = 0.0
                features[f'{ch_name}_spectral_bandwidth'] = 0.0
        
        return features
    
    def label_windows(self, window_times: List[Tuple[float, float]], seizure_times: List[Tuple[float, float]]) -> List[int]:
        """
        Label windows as preictal (1) or interictal (0)
        
        Args:
            window_times: List of (start, end) times for each window
            seizure_times: List of (start, end) times for seizures
            
        Returns:
            List of labels (0 or 1)
        """
        labels = []
        preictal_sec = self.config['preictal_window_sec']
        
        for win_start, win_end in window_times:
            is_preictal = False
            
            for seizure_start, seizure_end in seizure_times:
                # Check if window overlaps with preictal period
                preictal_start = seizure_start - preictal_sec
                preictal_end = seizure_start
                
                # Window overlaps with preictal period
                if (win_start < preictal_end and win_end > preictal_start):
                    is_preictal = True
                    break
                
                # Window overlaps with seizure itself
                if (win_start < seizure_end and win_end > seizure_start):
                    is_preictal = True
                    break
            
            labels.append(1 if is_preictal else 0)
            
        return labels
    
    def process_patient(self, patient: str) -> Optional[pd.DataFrame]:
        """Process a single patient's EEG data"""
        start_time = time.time()
        patient_dir = Path(self.config['input_dir']) / patient
        
        self.log(f"Processing patient {patient}...")
        
        # Find EDF files
        edf_files = list(patient_dir.glob('*.edf'))
        if not edf_files:
            self.log(f"No EDF files found for {patient}", 'ERROR')
            return None
        
        # Load seizure times
        seizure_file = patient_dir / f'Seizures-list-{patient}.txt'
        seizure_times = self.parse_seizure_times_fixed(seizure_file)
        
        patient_data = []
        
        for edf_file in edf_files:
            self.log(f"Processing {edf_file.name}...")
            
            try:
                # Load EDF file
                raw = mne.io.read_raw_edf(str(edf_file), preload=True, verbose=False)
                
                # Validate data quality
                validation = self.validate_eeg_data(raw)
                if not validation['valid']:
                    self.log(f"Data quality issues in {edf_file.name}: {validation['issues']}", 'WARNING')
                
                # Get EEG channels only
                eeg_channels = [ch for ch in raw.ch_names if ch.startswith('EEG') or ch in ['Fp1', 'Fp2', 'F3', 'F4', 'C3', 'C4', 'P3', 'P4', 'O1', 'O2', 'F7', 'F8', 'T3', 'T4', 'T5', 'T6', 'Fz', 'Cz', 'Pz']]
                
                if not eeg_channels:
                    # Fallback: use all non-EKG channels
                    eeg_channels = [ch for ch in raw.ch_names if 'EKG' not in ch and 'ECG' not in ch]
                
                if eeg_channels:
                    raw.pick_channels(eeg_channels)
                else:
                    self.log(f"No suitable EEG channels found in {edf_file.name}", 'WARNING')
                    continue
                
                # Resample if needed
                if raw.info['sfreq'] != self.config['target_sfreq']:
                    raw.resample(self.config['target_sfreq'])
                
                # Apply filtering
                raw.filter(
                    l_freq=self.config['filter_low_freq'],
                    h_freq=self.config['filter_high_freq'],
                    verbose=False
                )
                
                # Extract windows
                window_size_samples = int(self.config['window_size_sec'] * raw.info['sfreq'])
                overlap_samples = int(window_size_samples * self.config['overlap'])
                step_samples = window_size_samples - overlap_samples
                
                window_times = []
                windows_data = []
                
                for start_sample in range(0, len(raw.times) - window_size_samples, step_samples):
                    end_sample = start_sample + window_size_samples
                    
                    window_start_sec = start_sample / raw.info['sfreq']
                    window_end_sec = end_sample / raw.info['sfreq']
                    
                    window_data = raw.get_data()[:, start_sample:end_sample]
                    
                    window_times.append((window_start_sec, window_end_sec))
                    windows_data.append(window_data)
                
                # Get seizure times for this file
                file_seizures = seizure_times.get(edf_file.name, [])
                self.log(f"Found {len(file_seizures)} seizures in {edf_file.name}")
                
                # Label windows
                labels = self.label_windows(window_times, file_seizures)
                
                # Extract features for each window
                for i, (window_data, label) in enumerate(zip(windows_data, labels)):
                    win_start, win_end = window_times[i]
                    
                    try:
                        features = self.extract_window_features(
                            window_data, raw.info['sfreq'], raw.ch_names
                        )
                        
                        # Add metadata
                        features.update({
                            'subject_id': patient,
                            'file': edf_file.name,
                            'window_start_sec': win_start,
                            'window_end_sec': win_end,
                            'label': label
                        })
                        
                        patient_data.append(features)
                        
                        if label == 1:
                            self.stats['preictal_windows'] += 1
                        else:
                            self.stats['interictal_windows'] += 1
                        
                        self.stats['total_windows'] += 1
                        
                    except Exception as e:
                        error_msg = f"Error extracting features for window {i} in {edf_file.name}: {e}"
                        self.log(error_msg, 'ERROR')
                        self.stats['errors'].append(error_msg)
                        
            except Exception as e:
                error_msg = f"Error processing {edf_file.name}: {e}"
                self.log(error_msg, 'ERROR')
                self.stats['errors'].append(error_msg)
                continue
        
        processing_time = time.time() - start_time
        self.stats['processing_times'][patient] = processing_time
        self.log(f"Completed {patient} in {processing_time:.1f}s")
        
        if patient_data:
            return pd.DataFrame(patient_data)
        else:
            self.log(f"No valid data extracted for {patient}", 'WARNING')
            return None
    
    def run_complete_processing(self):
        """Process all available patients"""
        start_time = time.time()
        
        # Create output directory
        Path(self.config['output_dir']).mkdir(parents=True, exist_ok=True)
        
        # Find all patients
        patients = self.find_all_patients()
        
        if not patients:
            self.log("No patients found to process!", 'ERROR')
            return None
        
        all_data = []
        
        for patient in patients:
            patient_df = self.process_patient(patient)
            if patient_df is not None:
                all_data.append(patient_df)
                self.stats['patients_processed'] += 1
            else:
                self.log(f"Failed to process {patient}", 'ERROR')
        
        if not all_data:
            self.log("No valid data extracted from any patient!", 'ERROR')
            return None
        
        # Combine all patient data
        combined_df = pd.concat(all_data, ignore_index=True)
        
        # Save results
        output_file = Path(self.config['output_dir']) / 'eeg_windows_fixed.csv'
        combined_df.to_csv(output_file, index=False)
        
        # Print summary
        total_time = time.time() - start_time
        
        self.log("\n" + "="*60)
        self.log("PROCESSING SUMMARY")
        self.log("="*60)
        self.log(f"Total patients processed: {self.stats['patients_processed']}")
        self.log(f"Total windows extracted: {self.stats['total_windows']}")
        self.log(f"Preictal windows: {self.stats['preictal_windows']}")
        self.log(f"Interictal windows: {self.stats['interictal_windows']}")
        self.log(f"Total seizures detected: {self.stats['seizures_detected']}")
        
        if self.stats['preictal_windows'] > 0:
            ratio = self.stats['interictal_windows'] / self.stats['preictal_windows']
            self.log(f"Class imbalance ratio: {ratio:.1f}:1")
        
        self.log(f"Processing time: {total_time:.1f}s")
        self.log(f"Output saved to: {output_file}")
        
        if self.stats['errors']:
            self.log(f"Errors encountered: {len(self.stats['errors'])}")
            for error in self.stats['errors'][-5:]:  # Show last 5 errors
                self.log(f"  - {error}", 'ERROR')
        
        # Data quality summary
        self.log("\nDATA QUALITY SUMMARY")
        self.log("-" * 30)
        unique_patients = combined_df['subject_id'].nunique()
        unique_files = combined_df['file'].nunique()
        
        self.log(f"Unique patients in output: {unique_patients}")
        self.log(f"Unique files processed: {unique_files}")
        
        # Per-patient breakdown
        for patient in combined_df['subject_id'].unique():
            patient_data = combined_df[combined_df['subject_id'] == patient]
            preictal_count = (patient_data['label'] == 1).sum()
            interictal_count = (patient_data['label'] == 0).sum()
            self.log(f"{patient}: {preictal_count} preictal, {interictal_count} interictal")
        
        return combined_df

def main():
    parser = argparse.ArgumentParser(description='Fixed EEG Data Processing Pipeline')
    parser.add_argument('--input-dir', default='data/raw/physionet.org/files/siena-scalp-eeg/1.0.0',
                       help='Input directory containing patient folders')
    parser.add_argument('--output-dir', default='data/processed',
                       help='Output directory for processed data')
    parser.add_argument('--window-size', type=int, default=4,
                       help='Window size in seconds')
    parser.add_argument('--overlap', type=float, default=0.5,
                       help='Window overlap fraction')
    parser.add_argument('--preictal-window', type=int, default=60,
                       help='Preictal window duration in seconds')
    
    args = parser.parse_args()
    
    config = {
        'input_dir': args.input_dir,
        'output_dir': args.output_dir,
        'window_size_sec': args.window_size,
        'overlap': args.overlap,
        'preictal_window_sec': args.preictal_window,
        'filter_low_freq': 0.5,
        'filter_high_freq': 40,
        'target_sfreq': 256
    }
    
    processor = FixedEEGProcessor(config)
    result = processor.run_complete_processing()
    
    if result is not None:
        print(f"\nProcessing completed successfully!")
        print(f"Output shape: {result.shape}")
    else:
        print("Processing failed!")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main())