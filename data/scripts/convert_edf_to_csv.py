#!/usr/bin/env python3
"""
Siena Scalp EEG Dataset to CSV Converter

This script converts the Siena Scalp EEG dataset from EDF format to CSV format
suitable for machine learning training. It processes continuous EEG recordings
and creates labeled training windows.

Author: Data Analytics Final Project
"""

import os
import re
import pandas as pd
import numpy as np
import mne
from pathlib import Path
from typing import List, Tuple, Dict, Optional
import argparse
from scipy.stats import zscore, skew, kurtosis
from scipy.signal import welch


class EEGDataConverter:
    """Class to handle EDF to CSV conversion for EEG data."""
    
    def __init__(self, 
                 data_dir: str = "data/raw/siena-scalp-eeg",
                 output_file: str = "data/processed/eeg_windows.csv",
                 window_size_sec: int = 30,
                 preictal_window_sec: int = 300,
                 filter_low_freq: float = 0.5,
                 filter_high_freq: float = 40.0,
                 stride_sec: int = 30):
        """
        Initialize the EEG data converter.
        
        Args:
            data_dir: Path to the directory containing subject folders
            output_file: Path to output CSV file
            window_size_sec: Length of each window in seconds
            preictal_window_sec: Time before seizure considered preictal (seconds)
            filter_low_freq: Low cutoff frequency for bandpass filter (Hz)
            filter_high_freq: High cutoff frequency for bandpass filter (Hz)
            stride_sec: Window stride in seconds
        """
        self.data_dir = Path(data_dir)
        self.output_file = Path(output_file)
        self.window_size_sec = window_size_sec
        self.preictal_window_sec = preictal_window_sec
        self.filter_low_freq = filter_low_freq
        self.filter_high_freq = filter_high_freq
        self.stride_sec = stride_sec
        
        # Statistics tracking
        self.stats = {
            'total_windows': 0,
            'preictal_windows': 0,
            'interictal_windows': 0,
            'subjects_processed': 0,
            'subjects_skipped': 0,
            'errors': []
        }
        
        # Ensure output directory exists
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
    
    def parse_seizure_times(self, seizure_file: Path) -> Dict[str, List[Tuple[float, float]]]:
        """
        Parse seizure times from the Seizures-list-PNxx.txt file.
        
        Args:
            seizure_file: Path to the seizure list file
            
        Returns:
            Dictionary mapping EDF filenames to list of (start_sec, end_sec) tuples
        """
        file_seizures = {}
        
        try:
            with open(seizure_file, 'r') as f:
                lines = f.readlines()
            
            current_file = None
            reg_start_sec = None
            seizure_start_abs = None
            
            for line in lines:
                line = line.strip()
                
                if "File name:" in line:
                    current_file = line.split("File name:")[1].strip()
                    if current_file not in file_seizures:
                        file_seizures[current_file] = []
                        
                elif "Registration start time:" in line:
                    reg_start_time = line.split("Registration start time:")[1].strip()
                    reg_start_sec = self.time_to_seconds(reg_start_time)
                    
                elif "Seizure start time:" in line:
                    seizure_start_time = line.split("Seizure start time:")[1].strip()
                    seizure_start_abs = self.time_to_seconds(seizure_start_time)
                    
                elif "Seizure end time:" in line and current_file and reg_start_sec is not None:
                    seizure_end_time = line.split("Seizure end time:")[1].strip()
                    seizure_end_abs = self.time_to_seconds(seizure_end_time)
                    
                    # Convert to relative time from start of recording
                    start_rel = seizure_start_abs - reg_start_sec
                    end_rel = seizure_end_abs - reg_start_sec
                    
                    if start_rel >= 0 and end_rel > start_rel:
                        file_seizures[current_file].append((start_rel, end_rel))
        
        except Exception as e:
            self.stats['errors'].append(f"Error parsing seizure file {seizure_file}: {e}")
        
        return file_seizures
    
    def time_to_seconds(self, time_str: str) -> float:
        """Convert time string in format h.m.s to seconds."""
        parts = time_str.split('.')
        if len(parts) == 3:
            h, m, s = map(int, parts)
            return h * 3600 + m * 60 + s
        return 0.0
    
    def extract_features(self, window_data: np.ndarray, sfreq: float, ch_names: List[str]) -> Dict[str, float]:
        """
        Extract engineered features from EEG window data.
        
        Args:
            window_data: EEG data array (n_channels, n_timepoints)
            sfreq: Sampling frequency
            ch_names: Channel names
            
        Returns:
            Dictionary of engineered features
        """
        features = {}
        n_channels, n_timepoints = window_data.shape
        
        # Frequency bands (Hz)
        bands = {
            'delta': (0.5, 4),
            'theta': (4, 8), 
            'alpha': (8, 12),
            'beta': (12, 30),
            'gamma': (30, 40)
        }
        
        for ch_idx, ch_name in enumerate(ch_names):
            channel_data = window_data[ch_idx, :]
            
            # Statistical features
            features[f'{ch_name}_mean'] = np.mean(channel_data)
            features[f'{ch_name}_std'] = np.std(channel_data)
            features[f'{ch_name}_skew'] = skew(channel_data)
            features[f'{ch_name}_kurtosis'] = kurtosis(channel_data)
            
            # Power spectral density
            try:
                freqs, psd = welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)//4))
                
                # Extract power in each frequency band
                for band_name, (low_freq, high_freq) in bands.items():
                    mask = (freqs >= low_freq) & (freqs <= high_freq)
                    band_power = np.trapz(psd[mask], freqs[mask]) if np.any(mask) else 0
                    features[f'{ch_name}_{band_name}_power'] = band_power
                    
            except Exception as e:
                # Fallback: set band powers to 0 if PSD calculation fails
                for band_name in bands.keys():
                    features[f'{ch_name}_{band_name}_power'] = 0
                    
        return features
    
    def get_window_label(self, window_start: float, window_end: float, 
                        seizures: List[Tuple[float, float]]) -> int:
        """
        Determine the label for a window based on seizure times.
        
        Args:
            window_start: Start time of window in seconds
            window_end: End time of window in seconds
            seizures: List of (start_time, end_time) tuples for seizures
            
        Returns:
            0 for interictal, 1 for preictal
        """
        for seizure_start, seizure_end in seizures:
            # Check if window overlaps with seizure (ictal) - skip these
            if (window_start < seizure_end and window_end > seizure_start):
                return -1  # Mark for exclusion
            
            # Check if window is in preictal period (before seizure)
            preictal_start = seizure_start - self.preictal_window_sec
            if preictal_start <= window_start < seizure_start:
                return 1  # Preictal
        
        return 0  # Interictal
    
    def load_and_preprocess_edf(self, edf_file: Path) -> Optional[mne.io.Raw]:
        """
        Load EDF file and apply preprocessing.
        
        Args:
            edf_file: Path to EDF file
            
        Returns:
            Preprocessed MNE Raw object or None if error
        """
        try:
            # Load EDF file
            raw = mne.io.read_raw_edf(edf_file, preload=True, verbose=False)
            
            # Get only EEG channels (exclude EKG, etc.)
            eeg_channels = [ch for ch in raw.ch_names if not ch.upper().startswith('EKG')]
            if eeg_channels:
                raw.pick_channels(eeg_channels)
            
            # Apply bandpass filter
            raw.filter(self.filter_low_freq, self.filter_high_freq, 
                      fir_design='firwin', verbose=False)
            
            return raw
            
        except Exception as e:
            self.stats['errors'].append(f"Error processing {edf_file}: {e}")
            return None
    
    def create_windows(self, raw: mne.io.Raw, seizures: List[Tuple[float, float]], 
                      subject_id: str, file_name: str) -> List[Dict]:
        """
        Create fixed-length windows from continuous EEG data.
        
        Args:
            raw: MNE Raw object with EEG data
            seizures: List of seizure times
            subject_id: Subject identifier
            file_name: Original EDF file name
            
        Returns:
            List of dictionaries containing window data and metadata
        """
        windows = []
        sfreq = raw.info['sfreq']
        n_samples_window = int(self.window_size_sec * sfreq)
        stride_samples = int(self.stride_sec * sfreq)
        
        # Get data
        data, times = raw.get_data(return_times=True)
        n_channels, n_timepoints = data.shape
        
        # Create windows
        for start_idx in range(0, n_timepoints - n_samples_window + 1, stride_samples):
            end_idx = start_idx + n_samples_window
            
            # Get time bounds
            window_start_time = times[start_idx]
            window_end_time = times[end_idx - 1]
            
            # Get label
            label = self.get_window_label(window_start_time, window_end_time, seizures)
            
            # Skip ictal windows
            if label == -1:
                continue
            
            # Extract window data
            window_data = data[:, start_idx:end_idx]
            
            # Normalize each channel (z-score) 
            window_data_norm = np.zeros_like(window_data)
            for ch_idx in range(n_channels):
                channel_data = window_data[ch_idx, :]
                if np.std(channel_data) > 0:
                    window_data_norm[ch_idx, :] = zscore(channel_data)
                else:
                    window_data_norm[ch_idx, :] = channel_data
            
            # Create feature dictionary with metadata
            window_dict = {
                'subject_id': subject_id,
                'file': file_name,
                'window_start_sec': window_start_time,
                'window_end_sec': window_end_time,
                'label': label
            }
            
            # Extract engineered features instead of raw samples
            features = self.extract_features(window_data_norm, sfreq, raw.ch_names)
            window_dict.update(features)
            
            windows.append(window_dict)
            
            # Update statistics
            if label == 1:
                self.stats['preictal_windows'] += 1
            else:
                self.stats['interictal_windows'] += 1
            
            self.stats['total_windows'] += 1
        
        return windows
    
    def process_subject(self, subject_dir: Path) -> List[Dict]:
        """
        Process all EDF files for a single subject.
        
        Args:
            subject_dir: Path to subject directory (e.g., PN00)
            
        Returns:
            List of windows for this subject
        """
        subject_id = subject_dir.name
        print(f"Processing subject {subject_id}...")
        
        # Find seizure file
        seizure_files = list(subject_dir.glob(f"Seizures-list-{subject_id}.txt"))
        if not seizure_files:
            self.stats['errors'].append(f"No seizure file found for {subject_id}")
            self.stats['subjects_skipped'] += 1
            return []
        
        seizure_file = seizure_files[0]
        file_seizures = self.parse_seizure_times(seizure_file)
        
        # Find EDF files
        edf_files = list(subject_dir.glob("*.edf"))
        if not edf_files:
            self.stats['errors'].append(f"No EDF files found for {subject_id}")
            self.stats['subjects_skipped'] += 1
            return []
        
        subject_windows = []
        
        for edf_file in edf_files:
            print(f"  Processing {edf_file.name}...")
            
            # Load and preprocess
            raw = self.load_and_preprocess_edf(edf_file)
            if raw is None:
                continue
            
            # Get seizures for this specific file
            seizures = file_seizures.get(edf_file.name, [])
            
            # Create windows
            windows = self.create_windows(raw, seizures, subject_id, edf_file.name)
            subject_windows.extend(windows)
        
        self.stats['subjects_processed'] += 1
        return subject_windows
    
    def convert_dataset(self, test_loading: bool = False) -> None:
        """Convert the entire dataset from EDF to CSV format."""
        mode_text = "TEST LOADING" if test_loading else "CONVERSION"
        print(f"Starting EDF {mode_text}...")
        print(f"Data directory: {self.data_dir}")
        if not test_loading:
            print(f"Output file: {self.output_file}")
        print(f"Window size: {self.window_size_sec} seconds")
        print(f"Preictal window: {self.preictal_window_sec} seconds")
        print(f"Filter range: {self.filter_low_freq}-{self.filter_high_freq} Hz")
        if test_loading:
            print("TEST MODE: Only loading and validating data structure")
        print("-" * 50)
        
        # Find all subject directories
        subject_dirs = [d for d in self.data_dir.glob("PN*") if d.is_dir()]
        subject_dirs.sort()
        
        if not subject_dirs:
            print(f"No subject directories found in {self.data_dir}")
            return
        
        print(f"Found {len(subject_dirs)} subject directories")
        
        all_windows = []
        
        # Process each subject (limit to first 2 in test mode)
        test_limit = 2 if test_loading else len(subject_dirs)
        for i, subject_dir in enumerate(subject_dirs[:test_limit]):
            try:
                windows = self.process_subject(subject_dir)
                all_windows.extend(windows)
                
                if test_loading and i == 0:  # Show details for first subject in test mode
                    print(f"\nSample data from {subject_dir.name}:")
                    if windows:
                        sample_window = windows[0]
                        print(f"  - Channels: {len([k for k in sample_window.keys() if '_t' in k])}")
                        print(f"  - Window samples: {len([k for k in sample_window.keys() if k.startswith(list(sample_window.keys())[5])])}")
                        print(f"  - Labels: {set([w['label'] for w in windows[:10]])}")
                        print(f"  - Total windows for this subject: {len(windows)}")
                
            except Exception as e:
                self.stats['errors'].append(f"Error processing {subject_dir}: {e}")
                self.stats['subjects_skipped'] += 1
        
        # Convert to DataFrame and save
        if all_windows:
            print(f"\nCreating DataFrame with {len(all_windows)} windows...")
            df = pd.DataFrame(all_windows)
            
            if test_loading:
                print("\n=== TEST LOADING RESULTS ===")
                print(f"DataFrame shape: {df.shape}")
                print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
                print(f"Columns: {len(df.columns)} total")
                feature_cols = [col for col in df.columns if '_t' in col]
                print(f"Feature columns: {len(feature_cols)}")
                print(f"Data types: {df.dtypes.value_counts().to_dict()}")
                print("✅ Data loading validation successful!")
                return  # Don't save in test mode
            
            # Save to CSV
            print(f"Saving to {self.output_file}...")
            df.to_csv(self.output_file, index=False)
            
            # Get file size
            file_size_mb = self.output_file.stat().st_size / (1024 * 1024)
            
            print("\n" + "=" * 50)
            print("CONVERSION COMPLETE")
            print("=" * 50)
            print(f"Total windows created: {self.stats['total_windows']}")
            print(f"Preictal windows: {self.stats['preictal_windows']} "
                  f"({100 * self.stats['preictal_windows'] / max(1, self.stats['total_windows']):.1f}%)")
            print(f"Interictal windows: {self.stats['interictal_windows']} "
                  f"({100 * self.stats['interictal_windows'] / max(1, self.stats['total_windows']):.1f}%)")
            print(f"Subjects processed: {self.stats['subjects_processed']}")
            print(f"Subjects skipped: {self.stats['subjects_skipped']}")
            print(f"Output file: {self.output_file}")
            print(f"File size: {file_size_mb:.1f} MB")
            
            if self.stats['errors']:
                print(f"\nErrors encountered: {len(self.stats['errors'])}")
                for error in self.stats['errors'][:5]:  # Show first 5 errors
                    print(f"  - {error}")
                if len(self.stats['errors']) > 5:
                    print(f"  ... and {len(self.stats['errors']) - 5} more errors")
        else:
            print("No windows created. Check for errors above.")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Convert Siena EEG dataset from EDF to CSV')
    parser.add_argument('--data-dir', default='data/raw/siena-scalp-eeg',
                       help='Path to directory containing subject folders')
    parser.add_argument('--output', default='data/processed/eeg_windows.csv',
                       help='Output CSV file path')
    parser.add_argument('--window-size', type=int, default=30,
                       help='Window size in seconds (default: 30)')
    parser.add_argument('--preictal-window', type=int, default=300,
                       help='Preictal window duration in seconds (default: 300)')
    parser.add_argument('--filter-low', type=float, default=0.5,
                       help='Low cutoff frequency for bandpass filter (default: 0.5)')
    parser.add_argument('--filter-high', type=float, default=40.0,
                       help='High cutoff frequency for bandpass filter (default: 40.0)')
    parser.add_argument('--stride', type=int, default=None,
                       help='Window stride in seconds (default: same as window-size)')
    parser.add_argument('--test-loading', action='store_true',
                       help='Test mode: only load and validate 2 subjects without saving')
    
    args = parser.parse_args()
    
    # Set stride to window size if not specified (non-overlapping)
    stride = args.stride if args.stride is not None else args.window_size
    
    # Create converter and run
    converter = EEGDataConverter(
        data_dir=args.data_dir,
        output_file=args.output,
        window_size_sec=args.window_size,
        preictal_window_sec=args.preictal_window,
        filter_low_freq=args.filter_low,
        filter_high_freq=args.filter_high,
        stride_sec=stride
    )
    
    converter.convert_dataset(test_loading=args.test_loading)


if __name__ == "__main__":
    main()