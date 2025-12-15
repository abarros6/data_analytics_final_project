#!/bin/bash
# Virtual Environment Setup for EEG Seizure Prediction Project
# Run this script to create a consistent development environment

echo "Setting up virtual environment for EEG Seizure Prediction Project..."

# Create virtual environment
python3 -m venv eeg_venv

# Activate virtual environment
source eeg_venv/bin/activate

# Upgrade pip to latest version
pip install --upgrade pip

# Install project dependencies
pip install -r requirements.txt

echo "Virtual environment setup complete!"
echo ""
echo "To activate the environment in the future, run:"
echo "source eeg_venv/bin/activate"
echo ""
echo "To deactivate, run:"
echo "deactivate"