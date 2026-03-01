"""
Task 1.1a: Load and inspect CT DICOM files
This script loads DICOM files, inspects their properties, and organizes by Instance Number
"""

import os
import pydicom as dicom
from pathlib import Path
import numpy as np

# Configuration
data_folder = "/Users/mashrafi/dev/HC701/assignment2/Task1 Data"
dicom_files = []
slice_info = {}

print("=" * 80)
print("TASK 1.1a: DICOM INSPECTION")
print("=" * 80)

# Step 1: List all DICOM files
print("\n[Step 1] Scanning DICOM files...")
dcm_path = Path(data_folder)
dicom_files = sorted(list(dcm_path.glob("*.dcm")))
print(f"Found {len(dicom_files)} DICOM files")

if len(dicom_files) == 0:
    print("ERROR: No DICOM files found!")
    exit(1)

# Step 2: Inspect first file to understand structure
print("\n[Step 2] Inspecting first DICOM file structure...")
first_dcm = dicom.dcmread(str(dicom_files[0]))
print(f"\nFirst file: {dicom_files[0].name}")
print(f"Patient Name: {first_dcm.get('PatientName', 'N/A')}")
print(f"Modality: {first_dcm.get('Modality', 'N/A')}")
print(f"Image Shape: {first_dcm.pixel_array.shape}")
print(f"Slice Location: {first_dcm.get('SliceLocation', 'N/A')}")
print(f"Instance Number (0020,0013): {first_dcm.get('InstanceNumber', 'N/A')}")

# Step 3: Read all DICOM files and extract Instance Number
print("\n[Step 3] Reading all DICOM files and extracting Instance Numbers...")
instance_to_file = {}

for dcm_file in dicom_files:
    try:
        dcm_data = dicom.dcmread(str(dcm_file))
        instance_num = dcm_data.get('InstanceNumber', None)
        
        if instance_num is not None:
            instance_to_file[instance_num] = dcm_file
    except Exception as e:
        print(f"Error reading {dcm_file.name}: {e}")

print(f"Successfully read {len(instance_to_file)} DICOM files with Instance Numbers")

# Step 4: Find Instance Numbers 71-110
print("\n[Step 4] Finding slices 71-110...")
target_instances = list(range(71, 111))
found_instances = sorted([i for i in target_instances if i in instance_to_file.keys()])
missing_instances = [i for i in target_instances if i not in instance_to_file.keys()]

print(f"Target range: 71-110 ({len(target_instances)} slices)")
print(f"Found: {len(found_instances)} slices")
print(f"Missing: {len(missing_instances)} slices")

if missing_instances:
    print(f"Missing Instance Numbers: {missing_instances}")

# Step 5: Check Image Properties
print("\n[Step 5] Checking image properties...")
sample_instances = sorted(list(instance_to_file.keys()))[:5]
print(f"\nFirst 5 Instance Numbers: {sample_instances}")

for inst in sample_instances:
    dcm_data = dicom.dcmread(str(instance_to_file[inst]))
    print(f"Instance {inst}: shape={dcm_data.pixel_array.shape}, "
          f"pixel spacing={dcm_data.get('PixelSpacing', ['N/A', 'N/A'])}, "
          f"slice thickness={dcm_data.get('SliceThickness', 'N/A')}")

# Step 6: Summary
print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Total DICOM files: {len(dicom_files)}")
print(f"Total unique Instance Numbers: {len(instance_to_file)}")
print(f"Instance Number range: {min(instance_to_file.keys()) if instance_to_file else 'N/A'} - {max(instance_to_file.keys()) if instance_to_file else 'N/A'}")
print(f"Target slices (71-110): {len(found_instances)} found, {len(missing_instances)} missing")
print("\n✓ DICOM files are ready for extraction!")
