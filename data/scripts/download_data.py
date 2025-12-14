#!/usr/bin/env python3
"""
Data download script for Siena Scalp EEG dataset.
Downloads the dataset from PhysioNet using wget.
"""

import os
import subprocess
import sys
from pathlib import Path
import argparse


def download_siena_eeg_dataset(data_dir: str = "data/raw", verbose: bool = True, test_mode: bool = False):
    """
    Download the Siena Scalp EEG dataset from PhysioNet.
    
    Args:
        data_dir: Directory where the dataset will be downloaded
        verbose: Whether to show detailed output
        test_mode: If True, only download a small subset for testing
    """
    # Create data directory if it doesn't exist
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)
    
    # Change to data directory for download
    original_dir = os.getcwd()
    os.chdir(data_path)
    
    # PhysioNet URL for Siena Scalp EEG dataset
    url = "https://physionet.org/files/siena-scalp-eeg/1.0.0/"
    
    # wget command with flags:
    # -r: recursive download
    # -N: only download newer files (timestamp checking)
    # -c: continue partial downloads
    # -np: don't ascend to parent directory
    wget_cmd = ["wget", "-r", "-N", "-c", "-np"]
    
    if test_mode:
        # In test mode, limit download size and depth
        wget_cmd.extend([
            "--level=2",  # Only go 2 levels deep
            "--limit-rate=200k",  # Limit to 200KB/s to control download
            "--quota=50m",  # Stop after 50MB total
            r"--reject-regex=.*\.edf$"  # Skip the large EDF files in test mode
        ])
        if verbose:
            print("TEST MODE: Limited download (no EDF files, 50MB max)")
    
    wget_cmd.append(url)
    
    if verbose:
        print("=" * 60)
        print("Downloading Siena Scalp EEG Dataset from PhysioNet")
        print("=" * 60)
        print(f"URL: {url}")
        print(f"Target directory: {data_path.absolute()}")
        print(f"Command: {' '.join(wget_cmd)}")
        print("-" * 60)
    
    try:
        # Check if wget is available
        subprocess.run(["wget", "--version"], capture_output=True, check=True)
        
        if verbose:
            print("Starting download... This may take a while depending on your connection.")
            print("You can press Ctrl+C to cancel if needed.\n")
        
        # Run wget command
        if verbose:
            # Show output in real-time
            process = subprocess.Popen(wget_cmd, stdout=subprocess.PIPE, 
                                     stderr=subprocess.STDOUT, universal_newlines=True)
            
            for line in process.stdout:
                print(line, end='')
            
            process.wait()
            return_code = process.returncode
        else:
            # Run silently
            result = subprocess.run(wget_cmd, capture_output=True, text=True)
            return_code = result.returncode
        
        if return_code == 0:
            if verbose:
                print("\n" + "=" * 60)
                print("Download completed successfully!")
                
                # Show downloaded structure
                physionet_dir = data_path / "physionet.org" / "files" / "siena-scalp-eeg" / "1.0.0"
                if physionet_dir.exists():
                    print(f"Dataset downloaded to: {physionet_dir}")
                    
                    # List subject directories
                    subject_dirs = [d for d in physionet_dir.iterdir() if d.is_dir() and d.name.startswith("PN")]
                    if subject_dirs:
                        print(f"Found {len(subject_dirs)} subject directories:")
                        for subj_dir in sorted(subject_dirs)[:5]:  # Show first 5
                            print(f"  - {subj_dir.name}")
                        if len(subject_dirs) > 5:
                            print(f"  ... and {len(subject_dirs) - 5} more")
                    
                    # Suggest moving files
                    print(f"\nTo use with the conversion script, you may want to move/copy the subject")
                    print(f"directories to: {data_path / 'siena-scalp-eeg'}")
                    print(f"Example command:")
                    print(f"  mv {physionet_dir}/* {data_path}/siena-scalp-eeg/")
                
                print("=" * 60)
        else:
            print(f"\nDownload failed with return code: {return_code}")
            return False
            
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("Error: wget is not installed or not available in PATH")
        print("\nTo install wget:")
        print("  macOS: brew install wget")
        print("  Ubuntu/Debian: sudo apt-get install wget")
        print("  Windows: Download from https://www.gnu.org/software/wget/")
        print("\nAlternative: You can use curl instead:")
        print(f"  curl -L -o siena-eeg.zip {url}")
        return False
    
    except KeyboardInterrupt:
        print("\n\nDownload cancelled by user.")
        return False
    
    except Exception as e:
        print(f"\nUnexpected error: {e}")
        return False
    
    finally:
        # Return to original directory
        os.chdir(original_dir)
    
    return True


def setup_directory_structure(data_dir: str = "data/raw"):
    """
    Set up the expected directory structure for the conversion script.
    
    Args:
        data_dir: Base data directory
    """
    data_path = Path(data_dir)
    physionet_path = data_path / "physionet.org" / "files" / "siena-scalp-eeg" / "1.0.0"
    target_path = data_path / "siena-scalp-eeg"
    
    if physionet_path.exists() and not target_path.exists():
        print(f"\nSetting up directory structure...")
        print(f"Moving files from {physionet_path} to {target_path}")
        
        try:
            # Create target directory
            target_path.mkdir(parents=True, exist_ok=True)
            
            # Move all subject directories
            for item in physionet_path.iterdir():
                if item.is_dir() and item.name.startswith("PN"):
                    target_item = target_path / item.name
                    if not target_item.exists():
                        item.rename(target_item)
                        print(f"  Moved {item.name}")
            
            # Copy other important files
            for item in physionet_path.iterdir():
                if item.is_file():
                    target_item = target_path / item.name
                    if not target_item.exists():
                        import shutil
                        shutil.copy2(item, target_item)
                        print(f"  Copied {item.name}")
            
            print("Directory structure setup complete!")
            
        except Exception as e:
            print(f"Error setting up directory structure: {e}")
            print("You may need to manually move the files.")


def main():
    """Main function with command-line argument parsing."""
    parser = argparse.ArgumentParser(description='Download Siena Scalp EEG dataset')
    parser.add_argument('--data-dir', default='data/raw',
                       help='Directory to download data to (default: data/raw)')
    parser.add_argument('--quiet', action='store_true',
                       help='Run with minimal output')
    parser.add_argument('--setup-dirs', action='store_true',
                       help='Also setup directory structure for conversion script')
    parser.add_argument('--test', action='store_true',
                       help='Test mode: limited download (no EDF files, 50MB max)')
    
    args = parser.parse_args()
    
    # Download dataset
    success = download_siena_eeg_dataset(args.data_dir, verbose=not args.quiet, test_mode=args.test)
    
    if success and args.setup_dirs:
        setup_directory_structure(args.data_dir)
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())