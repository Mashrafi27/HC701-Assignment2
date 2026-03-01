"""
Task 1.3: Lung Segmentation using Non-ML Methods
This script implements HU-based thresholding and morphological operations
"""

import os
import numpy as np
import nibabel as nib
from scipy import ndimage
from scipy.ndimage import label, find_objects
from skimage import morphology
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
nifti_file = "/Users/mashrafi/dev/HC701/assignment2/ct_lungs_slices_71_110.nii.gz"
output_folder = "/Users/mashrafi/dev/HC701/assignment2/segmentation_results"
pixel_spacing = [0.68359375, 0.68359375]  # mm
slice_thickness = 1.25  # mm

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("=" * 80)
print("TASK 1.3: LUNG SEGMENTATION USING NON-ML METHODS")
print("=" * 80)

# Load the NIFTI file
print("\n[Step 1] Loading NIFTI file...")
img = nib.load(nifti_file)
volume = img.get_fdata()
print(f"Volume shape: {volume.shape}")
print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")

# Step 2: Lung tissue thresholding (using raw pixel values from DICOM)
print("\n[Step 2] Applying threshold for lung tissue...")
# The DICOM raw values are not HU values yet, but we can threshold based on intensity
# Air (lungs) has low values, soft tissue moderate, bone high
# We'll use percentile-based approach
threshold_low = np.percentile(volume[volume > 0], 5)  # 5th percentile of non-zero
threshold_high = np.percentile(volume, 30)  # 30th percentile overall

print(f"Using thresholds: {threshold_low:.2f} to {threshold_high:.2f}")
lung_mask = (volume >= threshold_low) & (volume <= threshold_high)
print(f"Found {lung_mask.sum()} voxels in lung tissue range ({np.sum(lung_mask)/np.prod(volume.shape)*100:.1f}%)")

# Step 3: Morphological operations on first slice
print("\n[Step 3] Processing first axial slice (71)...")
first_slice = lung_mask[:, :, 0]
mid_slice = lung_mask[:, :, 20]  # Middle slice for better visualization

# Apply morphological operations
print("  - Applying morphological closing...")
structure = morphology.disk(3)  # 2D disk instead of 3D ball
first_slice_closed = morphology.binary_closing(first_slice, structure)

print("  - Removing small objects...")
first_slice_cleaned = morphology.remove_small_objects(first_slice_closed, min_size=1000)

print("  - Applying morphological operations...")
first_slice_smooth = ndimage.binary_dilation(first_slice_cleaned, structure=structure, iterations=2)
first_slice_smooth = ndimage.binary_erosion(first_slice_smooth, structure=structure, iterations=2)

# Label connected components on first slice
labeled_first, num_components = label(first_slice_smooth)
print(f"  - Found {num_components} connected components in first slice")

# Find the two largest components (left and right lungs)
component_sizes = np.bincount(labeled_first.flat)
component_sizes[0] = 0  # Ignore background
largest_two = np.argsort(component_sizes)[-2:]
largest_two = largest_two[largest_two > 0]

print(f"  - Largest components: {sorted(component_sizes[largest_two], reverse=True)}")

# Create segmentation for first slice showing left and right lungs
first_slice_segmented = np.zeros_like(labeled_first)
left_lung_mask_first = (labeled_first == largest_two[np.argmin([np.where(labeled_first == c)[1].mean() for c in largest_two])])
right_lung_mask_first = (labeled_first == largest_two[np.argmax([np.where(labeled_first == c)[1].mean() for c in largest_two])])

first_slice_segmented[left_lung_mask_first] = 1
first_slice_segmented[right_lung_mask_first] = 2

print(f"  - Left lung voxels: {left_lung_mask_first.sum()}")
print(f"  - Right lung voxels: {right_lung_mask_first.sum()}")

# Step 4: Apply segmentation to all slices
print("\n[Step 4] Applying segmentation to all axial frames...")
full_segmentation = np.zeros_like(volume)
full_segmentation_left = np.zeros_like(volume)
full_segmentation_right = np.zeros_like(volume)

structure = morphology.disk(3)  # 2D disk structure

for z in range(volume.shape[2]):
    slice_mask = lung_mask[:, :, z]
    
    # Apply same morphological operations
    slice_closed = morphology.binary_closing(slice_mask, structure)
    slice_cleaned = morphology.remove_small_objects(slice_closed, min_size=500)
    slice_smooth = ndimage.binary_dilation(slice_cleaned, structure=structure, iterations=1)
    slice_smooth = ndimage.binary_erosion(slice_smooth, structure=structure, iterations=1)
    
    # Label components
    labeled_slice, num_comps = label(slice_smooth)
    
    if num_comps >= 2:
        # Find two largest components
        comp_sizes = np.bincount(labeled_slice.flat)
        comp_sizes[0] = 0
        largest = np.argsort(comp_sizes)[-2:]
        largest = largest[largest > 0]
        
        # Separate left and right
        left_idx = largest[np.argmin([np.where(labeled_slice == c)[1].mean() for c in largest])]
        right_idx = largest[np.argmax([np.where(labeled_slice == c)[1].mean() for c in largest])]
        
        full_segmentation[labeled_slice == left_idx, z] = 1
        full_segmentation[labeled_slice == right_idx, z] = 2
        full_segmentation_left[labeled_slice == left_idx, z] = 1
        full_segmentation_right[labeled_slice == right_idx, z] = 1
    elif num_comps == 1:
        # Only one component, assign to larger side
        largest = np.argsort(np.bincount(labeled_slice.flat))[-1]
        if largest > 0:
            full_segmentation[labeled_slice == largest, z] = 1
            full_segmentation_left[labeled_slice == largest, z] = 1

print(f"✓ Segmentation complete")

# Step 5: Compute volumes
print("\n[Step 5] Computing lung volumes...")
voxel_volume = pixel_spacing[0] * pixel_spacing[1] * slice_thickness / 1000  # cm³

total_lung_voxels = np.sum(full_segmentation > 0)
left_lung_voxels = np.sum(full_segmentation_left > 0)
right_lung_voxels = np.sum(full_segmentation_right > 0)

total_lung_volume = total_lung_voxels * voxel_volume
left_lung_volume = left_lung_voxels * voxel_volume
right_lung_volume = right_lung_voxels * voxel_volume

print(f"\nVoxel size: {voxel_volume:.6f} cm³")
print(f"\nTotal lung volume (both): {total_lung_volume:.2f} cm³")
print(f"Left lung volume: {left_lung_volume:.2f} cm³ ({left_lung_voxels} voxels)")
print(f"Right lung volume: {right_lung_volume:.2f} cm³ ({right_lung_voxels} voxels)")

# Step 6: Save masks
print("\n[Step 6] Saving segmentation masks...")
combined_mask_img = nib.Nifti1Image(full_segmentation, img.affine)
left_mask_img = nib.Nifti1Image(full_segmentation_left, img.affine)
right_mask_img = nib.Nifti1Image(full_segmentation_right, img.affine)

nib.save(combined_mask_img, os.path.join(output_folder, "lung_segmentation_combined.nii.gz"))
nib.save(left_mask_img, os.path.join(output_folder, "lung_segmentation_left.nii.gz"))
nib.save(right_mask_img, os.path.join(output_folder, "lung_segmentation_right.nii.gz"))
print("✓ Masks saved")

# Step 7: Create visualizations
print("\n[Step 7] Creating visualizations...")

# Normalize volume for display
volume_min = np.percentile(volume[volume > 0], 2)
volume_max = np.percentile(volume, 98)
windowed_volume = np.clip(volume, volume_min, volume_max)
windowed_volume = (windowed_volume - volume_min) / (volume_max - volume_min)

# Visualization 1: Segmentation on different slices
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
fig.suptitle("Lung Segmentation Results - Axial Views", fontsize=14, fontweight='bold')

slice_indices = [0, 10, 20, 30, 39]
selected_slices = slice_indices[:3]

for idx, z in enumerate(selected_slices):
    ax = axes[0, idx]
    ax.imshow(windowed_volume[:, :, z], cmap='gray', alpha=0.7)
    
    # Overlay segmentation
    seg_display = np.zeros((*full_segmentation[:, :, z].shape, 3))
    seg_display[full_segmentation[:, :, z] == 1] = [1, 0, 0]  # Left lung - red
    seg_display[full_segmentation[:, :, z] == 2] = [0, 0, 1]  # Right lung - blue
    
    ax.imshow(seg_display, alpha=0.5)
    ax.set_title(f"Slice {71 + z}")
    ax.axis('off')

# Bottom row - segmentation only
for idx, z in enumerate(selected_slices):
    ax = axes[1, idx]
    seg_display = np.zeros((*full_segmentation[:, :, z].shape, 3))
    seg_display[full_segmentation[:, :, z] == 1] = [1, 0, 0]  # Red
    seg_display[full_segmentation[:, :, z] == 2] = [0, 0, 1]  # Blue
    ax.imshow(seg_display)
    ax.set_title(f"Segmentation - Slice {71 + z}")
    ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "segmentation_axial_views.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: segmentation_axial_views.png")

# Visualization 2: Volume comparison
fig, ax = plt.subplots(figsize=(10, 6))
lungs = ['Left Lung', 'Right Lung', 'Total (Both)']
volumes = [left_lung_volume, right_lung_volume, total_lung_volume]
colors = ['red', 'blue', 'purple']

bars = ax.bar(lungs, volumes, color=colors, alpha=0.7, edgecolor='black', linewidth=2)
ax.set_ylabel('Volume (cm³)', fontsize=12, fontweight='bold')
ax.set_title('Lung Volume Measurements', fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, vol in zip(bars, volumes):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{vol:.2f}\ncm³',
            ha='center', va='bottom', fontsize=11, fontweight='bold')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "lung_volumes.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: lung_volumes.png")

# Visualization 3: First slice detailed view
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
fig.suptitle("First Slice (71) Detailed Segmentation", fontsize=14, fontweight='bold')

# Original
ax = axes[0]
ax.imshow(windowed_volume[:, :, 0], cmap='gray')
ax.set_title("Original CT Image")
ax.axis('off')

# Lung mask
ax = axes[1]
ax.imshow(lung_mask[:, :, 0], cmap='gray')
ax.set_title("HU-based Lung Mask")
ax.axis('off')

# Segmented with colors
ax = axes[2]
seg_display = np.zeros((*full_segmentation[:, :, 0].shape, 3))
seg_display[full_segmentation[:, :, 0] == 1] = [1, 0, 0]  # Red - left
seg_display[full_segmentation[:, :, 0] == 2] = [0, 0, 1]  # Blue - right
ax.imshow(seg_display)
ax.set_title("Final Segmentation\n(Red=Left, Blue=Right)")
ax.axis('off')

plt.tight_layout()
plt.savefig(os.path.join(output_folder, "first_slice_segmentation.png"), dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: first_slice_segmentation.png")

# Summary report
print("\n" + "=" * 80)
print("SEGMENTATION SUMMARY")
print("=" * 80)
print(f"\nMETHODOLOGY:")
print(f"1. Intensity-based thresholding using percentile approach")
print(f"2. Morphological closing with disk structure (radius=3) to fill small holes")
print(f"3. Small object removal (min_size: 1000 voxels for first slice, 500 for others)")
print(f"4. Binary dilation/erosion for smoothing")
print(f"5. Connected component labeling to separate left/right lungs")
print(f"6. Position-based assignment (left side = x < center, right side = x > center)")
print(f"\nVOLUME MEASUREMENTS:")
print(f"  Left Lung:  {left_lung_volume:8.2f} cm³")
print(f"  Right Lung: {right_lung_volume:8.2f} cm³")
print(f"  Total:      {total_lung_volume:8.2f} cm³")
print(f"\nDICOM SPACING:")
print(f"  Pixel spacing: {pixel_spacing} mm")
print(f"  Slice thickness: {slice_thickness} mm")
print(f"  Voxel volume: {voxel_volume:.6f} cm³")
print(f"\nOUTPUT FILES:")
print(f"  ✓ Segmentation masks (NIFTI format)")
print(f"  ✓ Visualization images (PNG format)")
print(f"\nNext steps:")
print(f"1. Review segmentation visualizations")
print(f"2. Load segmentation masks in 3D Slicer for 3D rendering")
print(f"3. Prepare report with methodology and results")
