#!/usr/bin/env python3
"""
EEG Seizure Prediction on New Data
=================================

Uses the trained generalized model to make predictions on new EEG data.
Supports both single file predictions and batch processing.

Usage:
    # Predict on new EDF file
    python scripts/predict_new_data.py --input path/to/new_file.edf --model models/generalized_seizure_model.pkl
    
    # Batch predict on directory
    python scripts/predict_new_data.py --input-dir path/to/edf_files/ --model models/generalized_seizure_model.pkl
    
    # Predict on pre-processed CSV
    python scripts/predict_new_data.py --csv path/to/features.csv --model models/generalized_seizure_model.pkl
"""

import os
import sys
import argparse
from pathlib import Path
import time
from datetime import datetime
from typing import List, Dict, Optional

import numpy as np
import pandas as pd
import mne
from scipy import stats
from scipy.signal import welch
from scipy.stats import skew, kurtosis, zscore
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent / 'src'))
from data_preprocessing import EEGPreprocessor

class SeizurePredictionService:
    def __init__(self, model_path: str):
        self.model_path = Path(model_path)
        self.load_model()
        
    def load_model(self):
        """Load the trained model."""
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found: {self.model_path}")
        
        print(f"Loading model from {self.model_path}")
        model_data = joblib.load(self.model_path)
        
        self.pipeline = model_data['pipeline']
        self.model_info = {
            'timestamp': model_data.get('timestamp'),
            'model_type': model_data.get('model_type'),
            'random_state': model_data.get('random_state')
        }
        
        print(f"Loaded {self.model_info['model_type']} model")
        print(f"Trained: {self.model_info['timestamp']}")
    
    def extract_features_from_edf(self, edf_path: str, window_size: int = 4, overlap: float = 0.5) -> pd.DataFrame:
        """Extract features from an EDF file using the same preprocessing as training."""
        print(f"Processing EDF file: {edf_path}")
        
        # Use the same preprocessing as training
        preprocessor = EEGPreprocessor({
            'window_size': window_size,
            'overlap': overlap,
            'target_freq': 256,
            'filter_low': 0.5,
            'filter_high': 40
        })
        
        # Load and preprocess the EDF file
        raw = mne.io.read_raw_edf(edf_path, preload=True, verbose=False)
        
        # Extract windows and features
        windows_data = []
        
        # Calculate window parameters
        window_samples = int(window_size * raw.info['sfreq'])
        step_samples = int(window_samples * (1 - overlap))
        
        for start_sample in range(0, len(raw.times) - window_samples, step_samples):
            end_sample = start_sample + window_samples
            window_data = raw.get_data()[:, start_sample:end_sample]
            
            # Extract features for this window
            features = self.extract_window_features(window_data, raw.info['sfreq'])
            features['window_start'] = start_sample / raw.info['sfreq']
            features['file_name'] = Path(edf_path).name
            
            windows_data.append(features)
        
        df = pd.DataFrame(windows_data)
        print(f"Extracted {len(df)} windows from {edf_path}")
        
        return df
    
    def extract_window_features(self, window_data: np.ndarray, sfreq: float) -> Dict:
        """Extract features from a single EEG window."""
        features = {}
        
        for ch_idx, channel_data in enumerate(window_data):
            ch_name = f"ch_{ch_idx}"
            
            # Time domain features
            features[f'{ch_name}_mean'] = np.mean(channel_data)
            features[f'{ch_name}_std'] = np.std(channel_data)
            features[f'{ch_name}_var'] = np.var(channel_data)
            features[f'{ch_name}_skew'] = skew(channel_data)
            features[f'{ch_name}_kurtosis'] = kurtosis(channel_data)
            features[f'{ch_name}_rms'] = np.sqrt(np.mean(channel_data**2))
            features[f'{ch_name}_peak_to_peak'] = np.ptp(channel_data)
            features[f'{ch_name}_zero_crossings'] = len(np.where(np.diff(np.signbit(channel_data)))[0])
            
            # Frequency domain features
            try:
                freqs, psd = welch(channel_data, fs=sfreq, nperseg=min(256, len(channel_data)//4))
                
                # Power in different frequency bands
                delta_power = np.trapz(psd[(freqs >= 0.5) & (freqs < 4)], freqs[(freqs >= 0.5) & (freqs < 4)])
                theta_power = np.trapz(psd[(freqs >= 4) & (freqs < 8)], freqs[(freqs >= 4) & (freqs < 8)])
                alpha_power = np.trapz(psd[(freqs >= 8) & (freqs < 13)], freqs[(freqs >= 8) & (freqs < 13)])
                beta_power = np.trapz(psd[(freqs >= 13) & (freqs < 30)], freqs[(freqs >= 13) & (freqs < 30)])
                gamma_power = np.trapz(psd[(freqs >= 30) & (freqs < 40)], freqs[(freqs >= 30) & (freqs < 40)])
                
                total_power = delta_power + theta_power + alpha_power + beta_power + gamma_power
                
                if total_power > 0:
                    features[f'{ch_name}_delta_rel'] = delta_power / total_power
                    features[f'{ch_name}_theta_rel'] = theta_power / total_power
                    features[f'{ch_name}_alpha_rel'] = alpha_power / total_power
                    features[f'{ch_name}_beta_rel'] = beta_power / total_power
                    features[f'{ch_name}_gamma_rel'] = gamma_power / total_power
                else:
                    features[f'{ch_name}_delta_rel'] = 0
                    features[f'{ch_name}_theta_rel'] = 0
                    features[f'{ch_name}_alpha_rel'] = 0
                    features[f'{ch_name}_beta_rel'] = 0
                    features[f'{ch_name}_gamma_rel'] = 0
                
                features[f'{ch_name}_spectral_centroid'] = np.sum(freqs * psd) / np.sum(psd)
                features[f'{ch_name}_spectral_bandwidth'] = np.sqrt(np.sum(((freqs - features[f'{ch_name}_spectral_centroid']) ** 2) * psd) / np.sum(psd))
                
            except Exception as e:
                print(f"Warning: Could not compute frequency features for channel {ch_idx}: {e}")
                # Set default values
                for feat in ['delta_rel', 'theta_rel', 'alpha_rel', 'beta_rel', 'gamma_rel', 'spectral_centroid', 'spectral_bandwidth']:
                    features[f'{ch_name}_{feat}'] = 0
        
        return features
    
    def predict_from_features(self, features_df: pd.DataFrame) -> Dict:
        """Make predictions from extracted features."""
        # Remove metadata columns
        feature_cols = [col for col in features_df.columns if col not in ['window_start', 'file_name', 'label', 'patient_id']]
        X = features_df[feature_cols].values
        
        # Make predictions
        predictions = self.pipeline.predict(X)
        probabilities = self.pipeline.predict_proba(X)[:, 1]  # Probability of seizure (class 1)
        
        results = {
            'predictions': predictions,
            'probabilities': probabilities,
            'n_windows': len(features_df),
            'seizure_windows': int(np.sum(predictions)),
            'seizure_probability_mean': float(np.mean(probabilities)),
            'seizure_probability_max': float(np.max(probabilities)),
            'seizure_probability_std': float(np.std(probabilities))
        }
        
        return results
    
    def predict_edf_file(self, edf_path: str, output_dir: str = None) -> Dict:
        """Complete prediction pipeline for a single EDF file."""
        start_time = time.time()
        
        # Extract features
        features_df = self.extract_features_from_edf(edf_path)
        
        # Make predictions
        results = self.predict_from_features(features_df)
        
        # Add file info
        results['file_path'] = edf_path
        results['file_name'] = Path(edf_path).name
        results['processing_time'] = time.time() - start_time
        
        # Create detailed results DataFrame
        feature_cols = [col for col in features_df.columns if col not in ['window_start', 'file_name']]
        results_df = features_df[['window_start', 'file_name']].copy()
        results_df['prediction'] = results['predictions']
        results_df['seizure_probability'] = results['probabilities']
        
        results['detailed_results'] = results_df
        
        # Save results if output directory specified
        if output_dir:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            # Save detailed results
            results_file = output_path / f"{Path(edf_path).stem}_predictions.csv"
            results_df.to_csv(results_file, index=False)
            
            # Create visualization
            self.create_prediction_visualization(results_df, output_path / f"{Path(edf_path).stem}_predictions.png")
            
            results['output_files'] = {
                'csv': str(results_file),
                'plot': str(output_path / f"{Path(edf_path).stem}_predictions.png")
            }
        
        return results
    
    def create_prediction_visualization(self, results_df: pd.DataFrame, output_path: str):
        """Create visualization of predictions over time."""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 8))
        
        # Plot seizure probability over time
        ax1.plot(results_df['window_start'], results_df['seizure_probability'], 'b-', alpha=0.7)
        ax1.axhline(y=0.5, color='r', linestyle='--', alpha=0.7, label='Decision Threshold')
        ax1.set_ylabel('Seizure Probability')
        ax1.set_title(f'Seizure Prediction Timeline - {results_df["file_name"].iloc[0]}')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot binary predictions
        ax2.plot(results_df['window_start'], results_df['prediction'], 'ro-', alpha=0.7, markersize=3)
        ax2.set_ylabel('Predicted Class')
        ax2.set_xlabel('Time (seconds)')
        ax2.set_title('Binary Predictions (0=Interictal, 1=Preictal)')
        ax2.set_yticks([0, 1])
        ax2.set_yticklabels(['Interictal', 'Preictal'])
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
    
    def print_prediction_summary(self, results: Dict):
        """Print a summary of prediction results."""
        print(f"\nPREDICTION SUMMARY")
        print(f"=" * 50)
        print(f"File: {results['file_name']}")
        print(f"Processing time: {results['processing_time']:.1f} seconds")
        print(f"Total windows analyzed: {results['n_windows']}")
        print(f"Windows predicted as preictal: {results['seizure_windows']} ({results['seizure_windows']/results['n_windows']*100:.1f}%)")
        print(f"Average seizure probability: {results['seizure_probability_mean']:.3f}")
        print(f"Maximum seizure probability: {results['seizure_probability_max']:.3f}")
        print(f"Probability standard deviation: {results['seizure_probability_std']:.3f}")
        
        if 'output_files' in results:
            print(f"\nOutput files:")
            print(f"  Detailed results: {results['output_files']['csv']}")
            print(f"  Visualization: {results['output_files']['plot']}")

def main():
    parser = argparse.ArgumentParser(description='Predict seizures on new EEG data')
    parser.add_argument('--model', required=True, help='Path to trained model file')
    parser.add_argument('--input', help='Input EDF file')
    parser.add_argument('--input-dir', help='Directory containing EDF files')
    parser.add_argument('--csv', help='Pre-processed CSV file with features')
    parser.add_argument('--output-dir', default='predictions', help='Output directory for results')
    
    args = parser.parse_args()
    
    if not any([args.input, args.input_dir, args.csv]):
        parser.error("Must specify one of --input, --input-dir, or --csv")
    
    # Initialize prediction service
    service = SeizurePredictionService(args.model)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)
    
    if args.csv:
        # Predict from pre-processed CSV
        print(f"Loading features from {args.csv}")
        features_df = pd.read_csv(args.csv)
        results = service.predict_from_features(features_df)
        
        print(f"\nPrediction Results:")
        print(f"Total windows: {results['n_windows']}")
        print(f"Predicted seizure windows: {results['seizure_windows']}")
        print(f"Mean seizure probability: {results['seizure_probability_mean']:.3f}")
        
    elif args.input:
        # Predict single file
        results = service.predict_edf_file(args.input, str(output_dir))
        service.print_prediction_summary(results)
        
    elif args.input_dir:
        # Batch predict directory
        input_dir = Path(args.input_dir)
        edf_files = list(input_dir.glob("*.edf"))
        
        if not edf_files:
            print(f"No EDF files found in {input_dir}")
            return
        
        print(f"Found {len(edf_files)} EDF files to process")
        
        all_results = []
        for edf_file in edf_files:
            print(f"\nProcessing {edf_file.name}...")
            try:
                results = service.predict_edf_file(str(edf_file), str(output_dir))
                all_results.append(results)
                service.print_prediction_summary(results)
            except Exception as e:
                print(f"Error processing {edf_file}: {e}")
        
        # Create summary report
        if all_results:
            summary_df = pd.DataFrame([
                {
                    'file_name': r['file_name'],
                    'n_windows': r['n_windows'],
                    'seizure_windows': r['seizure_windows'],
                    'seizure_percentage': r['seizure_windows'] / r['n_windows'] * 100,
                    'mean_probability': r['seizure_probability_mean'],
                    'max_probability': r['seizure_probability_max']
                }
                for r in all_results
            ])
            
            summary_path = output_dir / 'batch_prediction_summary.csv'
            summary_df.to_csv(summary_path, index=False)
            print(f"\nBatch summary saved to {summary_path}")

if __name__ == "__main__":
    main()