"""
Task 1.1b & 1.1c: Extract slices 71-110 and save as .nii.gz
This script extracts the target slices and converts them to NIFTI format
"""

import os
import pydicom
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

# Configuration
data_folder = "/Users/mashrafi/dev/HC701/assignment2/Task1 Data"
output_folder = "/Users/mashrafi/dev/HC701/assignment2"
output_filename = "ct_lungs_slices_71_110.nii.gz"

print("=" * 80)
print("TASK 1.1b & 1.1c: EXTRACT SLICES AND SAVE AS NIFTI")
print("=" * 80)

# Step 1: Load all DICOM files and create mapping
print("\n[Step 1] Loading all DICOM files...")
dcm_path = Path(data_folder)
dicom_files = sorted(list(dcm_path.glob("*.dcm")))

# Create mapping of Instance Number to file path
instance_to_file = {}
for dcm_file in dicom_files:
    try:
        dcm_data = pydicom.dcmread(str(dcm_file))
        instance_num = dcm_data.get('InstanceNumber', None)
        if instance_num is not None:
            instance_to_file[instance_num] = dcm_file
    except Exception as e:
        print(f"Warning: Could not read {dcm_file.name}: {e}")

print(f"Loaded {len(instance_to_file)} DICOM files")

# Step 2: Extract slices 71-110
print("\n[Step 2] Extracting slices 71-110...")
target_instances = list(range(71, 111))
volume_data = []

for inst in tqdm(target_instances, desc="Extracting slices"):
    if inst in instance_to_file:
        dcm_data = pydicom.dcmread(str(instance_to_file[inst]))
        # Get pixel array and convert to float
        pixel_array = dcm_data.pixel_array.astype(np.float32)
        volume_data.append(pixel_array)

print(f"Successfully extracted {len(volume_data)} slices")

# Step 3: Stack slices to create 3D volume
print("\n[Step 3] Creating 3D volume...")
# Stack slices - they will be in order (depth, height, width)
volume_3d = np.stack(volume_data, axis=2)
print(f"3D Volume shape: {volume_3d.shape}")

# Get pixel spacing and slice thickness from first matching DICOM
first_matching_dcm = pydicom.dcmread(str(instance_to_file[71]))
pixel_spacing = list(map(float, first_matching_dcm.get('PixelSpacing', [1, 1])))
slice_thickness = float(first_matching_dcm.get('SliceThickness', 1.25))

print(f"Pixel spacing (mm): {pixel_spacing}")
print(f"Slice thickness (mm): {slice_thickness}")

# Step 4: Create affine matrix
print("\n[Step 4] Creating affine transformation matrix...")
# Create a simple diagonal affine matrix with pixel spacing and slice thickness
affine = np.eye(4)
affine[0, 0] = pixel_spacing[0]  # x spacing
affine[1, 1] = pixel_spacing[1]  # y spacing
affine[2, 2] = slice_thickness    # z spacing

print(f"Affine matrix:\n{affine}")

# Step 5: Create NIfTI image and save
print("\n[Step 5] Creating and saving NIfTI file...")
nifti_img = nib.Nifti1Image(volume_3d, affine)

# Update header with relevant information
nifti_img.header.set_data_dtype(np.float32)

output_path = os.path.join(output_folder, output_filename)
nib.save(nifti_img, output_path)

print(f"✓ Successfully saved: {output_path}")
print(f"  File size: {os.path.getsize(output_path) / (1024*1024):.2f} MB")

# Step 6: Verify the saved file
print("\n[Step 6] Verifying saved file...")
loaded_img = nib.load(output_path)
print(f"Loaded shape: {loaded_img.shape}")
print(f"Loaded data type: {loaded_img.get_data_dtype()}")
print(f"Loaded affine:\n{loaded_img.affine}")

print("\n" + "=" * 80)
print("✓ EXTRACTION COMPLETE!")
print("=" * 80)
print(f"\nNext steps:")
print(f"1. Open {output_filename} in 3D Slicer")
print(f"2. Take a screenshot of the 3D volume")
print(f"3. Continue with Task 1.2 (intensity windowing)")
