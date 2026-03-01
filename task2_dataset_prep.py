"""
Task 2: X-ray Pneumonia Classification
This script downloads and prepares the Kaggle Pneumonia X-ray dataset
"""

import os
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil

print("=" * 80)
print("TASK 2.1: X-RAY PNEUMONIA DATASET PREPARATION")
print("=" * 80)

# Configuration - use relative paths for cross-platform compatibility
script_dir = os.path.dirname(os.path.abspath(__file__))
data_folder = os.path.join(script_dir, "pneumonia_data")
output_folder = os.path.join(script_dir, "pneumonia_results")

os.makedirs(data_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

print("""
IMPORTANT: Download Instructions
To download the Kaggle Pneumonia X-ray dataset, follow these steps:

1. Set up Kaggle API credentials:
   - Go to https://www.kaggle.com/settings/account
   - Click "Create new API token"
   - Save the kaggle.json file to ~/.kaggle/kaggle.json
   - Run: chmod 600 ~/.kaggle/kaggle.json

2. Install kaggle CLI (if not already installed):
   pip install kaggle

3. Download the dataset using command:
   kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {data_folder}

4. Extract the downloaded file:
   unzip chest-xray-pneumonia.zip -d {data_folder}
   
The dataset structure should be:
{data_folder}/chest_xray/train/
  ├── NORMAL/
  │   ├── *.jpeg files
  └── PNEUMONIA/
      ├── *.jpeg files
{data_folder}/chest_xray/val/
{data_folder}/chest_xray/test/
""".format(data_folder=data_folder))

print("\n" + "=" * 80)
print("CHECKING FOR EXISTING DATASET...")
print("=" * 80)

train_path = os.path.join(data_folder, "chest_xray", "train")
val_path = os.path.join(data_folder, "chest_xray", "val")
test_path = os.path.join(data_folder, "chest_xray", "test")

if os.path.exists(train_path):
    print(f"\n✓ Found training data at: {train_path}")
    
    # List classes
    classes = []
    for class_name in os.listdir(train_path):
        class_path = os.path.join(train_path, class_name)
        if os.path.isdir(class_path):
            num_images = len([f for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))])
            classes.append((class_name, num_images))
            print(f"  - {class_name}: {num_images} images")
    
    # Now split the training data
    print("\n" + "=" * 80)
    print("TASK 2.1b: SPLITTING TRAINING DATA")
    print("=" * 80)
    
    # Create splits
    all_files = []
    all_labels = []
    
    for class_name, _ in classes:
        class_path = os.path.join(train_path, class_name)
        files = [f for f in os.listdir(class_path) if f.endswith(('.jpeg', '.jpg', '.png'))]
        for f in files:
            all_files.append(f)
            all_labels.append(class_name)
    
    # Split: 80% train, 10% val, 10% test (from the original training set)
    train_indices, remaining_indices = train_test_split(
        range(len(all_files)), 
        test_size=0.2, 
        random_state=42, 
        stratify=all_labels
    )
    
    remaining_files = [all_files[i] for i in remaining_indices]
    remaining_labels = [all_labels[i] for i in remaining_indices]
    
    val_indices, test_indices = train_test_split(
        range(len(remaining_files)), 
        test_size=0.5, 
        random_state=42,
        stratify=remaining_labels
    )
    
    train_files = [all_files[i] for i in train_indices]
    train_labels = [all_labels[i] for i in train_indices]
    
    val_files = [remaining_files[i] for i in val_indices]
    val_labels = [remaining_labels[i] for i in val_indices]
    
    test_files = [remaining_files[i] for i in test_indices]
    test_labels = [remaining_labels[i] for i in test_indices]
    
    print(f"\nOriginal training set: {len(all_files)} images")
    print(f"New training set: {len(train_files)} images ({len(train_files)/len(all_files)*100:.1f}%)")
    print(f"New validation set: {len(val_files)} images ({len(val_files)/len(all_files)*100:.1f}%)")
    print(f"New test set: {len(test_files)} images ({len(test_files)/len(all_files)*100:.1f}%)")
    
    # Create DataFrames
    train_df = pd.DataFrame({
        'filename': train_files,
        'label': train_labels,
        'filepath': [os.path.join(train_path, label, f) for f, label in zip(train_files, train_labels)]
    })
    
    val_df = pd.DataFrame({
        'filename': val_files,
        'label': val_labels,
        'filepath': [os.path.join(train_path, label, f) for f, label in zip(val_files, val_labels)]
    })
    
    test_df = pd.DataFrame({
        'filename': test_files,
        'label': test_labels,
        'filepath': [os.path.join(train_path, label, f) for f, label in zip(test_files, test_labels)]
    })
    
    # Save to CSV
    csv_file = os.path.join(output_folder, "data_split.csv")
    
    # Combine into single CSV with split column
    train_df['split'] = 'train'
    val_df['split'] = 'val'
    test_df['split'] = 'test'
    
    combined_df = pd.concat([train_df, val_df, test_df], ignore_index=True)
    combined_df.to_csv(csv_file, index=False)
    
    print(f"\n✓ Saved split information to: {csv_file}")
    
    # Print split summary
    print("\n" + "=" * 80)
    print("TASK 2.1c: DATA SPLIT REPORT")
    print("=" * 80)
    
    for split_name, df in [('Training', train_df), ('Validation', val_df), ('Testing', test_df)]:
        print(f"\n{split_name} Set:")
        for label in df['label'].unique():
            label_count = len(df[df['label'] == label])
            files = df[df['label'] == label]['filename'].tolist()
            print(f"  {label}:")
            print(f"    Count: {label_count}")
            print(f"    Files: {files[0]} ... {files[-1]}")
    
    # Display CSV preview
    print("\n" + "-" * 80)
    print("CSV File Preview:")
    print(combined_df.head(10))
    print(f"\nTotal rows: {len(combined_df)}")
    
    print("\n" + "=" * 80)
    print("✓ DATA PREPARATION COMPLETE")
    print("=" * 80)
    
else:
    print(f"\n✗ Training data not found at: {train_path}")
    print(f"\nPlease download the dataset following the instructions above.")
    print(f"\nQuick Kaggle download command:")
    print(f"  kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p {data_folder}")
    print(f"  unzip {os.path.join(data_folder, 'chest-xray-pneumonia.zip')} -d {data_folder}")

print("\nNext steps:")
print("1. Verify the CSV file contains all splits")
print("2. Proceed to Task 2.2 (ML Model Development)")
