#!/usr/bin/env python3
"""
Kaggle Dataset Download and Setup
Handles authentication and automatic dataset download
"""

import os
import json
import subprocess
import sys
from pathlib import Path

def check_kaggle_installed():
    """Check if kaggle CLI is installed"""
    try:
        import kaggle
        return True
    except ImportError:
        return False

def check_credentials():
    """Check if Kaggle credentials are configured"""
    kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
    return kaggle_json.exists()

def download_dataset():
    """Download the chest X-ray pneumonia dataset"""
    data_dir = Path("/Users/mashrafi/dev/HC701/assignment2/pneumonia_data")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "="*80)
    print("DOWNLOADING KAGGLE DATASET")
    print("="*80)
    
    try:
        cmd = [
            "kaggle", "datasets", "download",
            "-d", "paultimothymooney/chest-xray-pneumonia",
            "-p", str(data_dir)
        ]
        print(f"\nRunning: {' '.join(cmd)}\n")
        result = subprocess.run(cmd, cwd="/Users/mashrafi/dev/HC701/assignment2")
        
        if result.returncode == 0:
            print("\n✓ Dataset downloaded successfully")
            
            # Extract the zip file
            zip_file = data_dir / "chest-xray-pneumonia.zip"
            if zip_file.exists():
                print(f"\nExtracting: {zip_file}")
                extract_cmd = ["unzip", "-q", str(zip_file), "-d", str(data_dir)]
                subprocess.run(extract_cmd, check=True)
                print("✓ Dataset extracted successfully")
                
                # Remove the zip file
                zip_file.unlink()
                print("✓ Cleaned up zip file")
            
            return True
        else:
            print("\n✗ Download failed")
            return False
            
    except FileNotFoundError as e:
        print(f"\n✗ Error: {e}")
        print("  Ensure kaggle CLI is installed: pip install kaggle")
        return False

def main():
    print("\n" + "="*80)
    print("KAGGLE SETUP STATUS CHECK")
    print("="*80)
    
    # Check 1: Kaggle CLI installed
    print("\n[1] Checking Kaggle CLI installation...", end=" ")
    if check_kaggle_installed():
        print("✓")
    else:
        print("✗ (fixed: installing now...)")
        subprocess.run(["conda", "run", "-n", "CV8501", "pip", "install", "kaggle", "-q"])
    
    # Check 2: Credentials
    print("[2] Checking Kaggle credentials...", end=" ")
    if check_credentials():
        print("✓")
    else:
        print("✗")
        print("\n" + "!"*80)
        print("KAGGLE CREDENTIALS NOT FOUND")
        print("!"*80)
        print("""
To set up Kaggle credentials:

1. Go to: https://www.kaggle.com/account
2. Scroll to "API" section
3. Click "Create New API Token"
4. This downloads kaggle.json to ~/Downloads/
5. Run these commands in terminal:

   mkdir -p ~/.kaggle
   cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
   chmod 600 ~/.kaggle/kaggle.json

6. Then run this script again

Once configured, the dataset will be automatically downloaded.
Dataset size: ~1.2 GB (may take 5-10 minutes)
""")
        return False
    
    # Download
    print("[3] Downloading dataset...")
    download_dataset()
    
    # Verify
    print("\n[4] Verifying download...", end=" ")
    chest_xray_dir = Path("/Users/mashrafi/dev/HC701/assignment2/pneumonia_data/chest_xray")
    if chest_xray_dir.exists():
        print("✓")
        
        # Count files
        try:
            train_normal = len(list((chest_xray_dir / "train" / "NORMAL").glob("*.jpeg")))
            train_pneumonia = len(list((chest_xray_dir / "train" / "PNEUMONIA").glob("*.jpeg")))
            
            print(f"\n✓ Dataset ready:")
            print(f"  Training set: {train_normal + train_pneumonia} images")
            print(f"    - NORMAL: {train_normal}")
            print(f"    - PNEUMONIA: {train_pneumonia}")
            
            return True
        except:
            print("✓ (directory exists)")
            return True
    else:
        print("✗")
        return False

if __name__ == "__main__":
    success = main()
    
    print("\n" + "="*80)
    if success:
        print("✓ SETUP COMPLETE - Ready to proceed with Task 2.1b")
        print("\nNext command:")
        print("  python task2_dataset_prep.py")
    else:
        print("✗ SETUP INCOMPLETE - Please configure Kaggle credentials")
    print("="*80 + "\n")
    
    sys.exit(0 if success else 1)
