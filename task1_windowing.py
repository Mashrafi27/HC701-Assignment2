"""
Task 1.2: Intensity Windowing and Visualization
This script visualizes CT scans with different intensity windows to find the optimal one
"""

import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from pathlib import Path

# Configuration
nifti_file = "/Users/mashrafi/dev/HC701/assignment2/ct_lungs_slices_71_110.nii.gz"
output_folder = "/Users/mashrafi/dev/HC701/assignment2/windowing_results"

# Create output folder
os.makedirs(output_folder, exist_ok=True)

print("=" * 80)
print("TASK 1.2: INTENSITY WINDOWING ANALYSIS")
print("=" * 80)

# Load the NIFTI file
print("\n[Step 1] Loading NIFTI file...")
img = nib.load(nifti_file)
volume = img.get_fdata()
print(f"Volume shape: {volume.shape}")
print(f"Data type: {volume.dtype}")
print(f"Value range: [{volume.min():.2f}, {volume.max():.2f}]")

# Get middle slice indices
mid_z = volume.shape[2] // 2
mid_x = volume.shape[0] // 2
mid_y = volume.shape[1] // 2

print(f"\nMiddle slice indices:")
print(f"  Axial (z): {mid_z}")
print(f"  Coronal (x): {mid_x}")
print(f"  Sagittal (y): {mid_y}")

# Define different windowing presets for CT
# (center in HU, width in HU)
window_presets = {
    "Brain": (40, 80),
    "Chest (Mediastinum)": (40, 400),
    "Lung": (-500, 1500),
    "Bone": (400, 1500),
    "Liver": (60, 150),
    "Wide": (0, 512),
}

print("\n[Step 2] Testing different window presets...")
print("Window presets:")
for name, (center, width) in window_presets.items():
    print(f"  {name}: center={center}, width={width}")

# Apply windowing function
def apply_window(data, center, width):
    """Apply windowing to CT data"""
    min_val = center - width / 2
    max_val = center + width / 2
    windowed = np.clip(data, min_val, max_val)
    # Normalize to 0-1
    windowed = (windowed - min_val) / (max_val - min_val)
    return windowed

# Create visualization for each window
print("\n[Step 3] Creating visualizations...")

for window_name, (center, width) in window_presets.items():
    print(f"  Processing: {window_name}")
    
    # Apply windowing
    windowed = apply_window(volume, center, width)
    
    # Create figure with 3 views
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f"{window_name} Window (center={center}, width={width})", 
                 fontsize=14, fontweight='bold')
    
    # Axial view (xy plane at middle z)
    ax = axes[0]
    ax.imshow(windowed[:, :, mid_z], cmap='gray')
    ax.set_title(f"Axial View (slice {mid_z})")
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.grid(True, alpha=0.3)
    
    # Coronal view (yz plane at middle x)
    ax = axes[1]
    ax.imshow(windowed[mid_x, :, :], cmap='gray', aspect='auto')
    ax.set_title(f"Coronal View (x={mid_x})")
    ax.set_xlabel("Y")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    
    # Sagittal view (xz plane at middle y)
    ax = axes[2]
    ax.imshow(windowed[:, mid_y, :], cmap='gray', aspect='auto')
    ax.set_title(f"Sagittal View (y={mid_y})")
    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    output_file = os.path.join(output_folder, f"window_{window_name.replace(' ', '_').lower()}.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"    Saved: {output_file}")

# Create a comparison figure showing the optimal lung window
print("\n[Step 4] Creating optimal lung window comparison...")

fig, axes = plt.subplots(2, 3, figsize=(18, 10))
fig.suptitle("Intensity Window Comparison for Lung CT Analysis", fontsize=16, fontweight='bold')

# Use Lung window for all views
center, width = window_presets["Lung"]
windowed = apply_window(volume, center, width)

# Row 1: Different windows for axial view
windows_to_compare = ["Chest (Mediastinum)", "Lung", "Bone"]
for idx, window_name in enumerate(windows_to_compare):
    c, w = window_presets[window_name]
    w_data = apply_window(volume, c, w)
    
    ax = axes[0, idx]
    ax.imshow(w_data[:, :, mid_z], cmap='gray')
    ax.set_title(f"{window_name}\n(center={c}, width={w})")
    ax.set_xlabel("X (mm)")
    ax.set_ylabel("Y (mm)")
    ax.grid(True, alpha=0.3)

# Row 2: Lung window in all three views
views = [
    (windowed[:, :, mid_z], f"Axial (z={mid_z})", "X (mm)", "Y (mm)"),
    (windowed[mid_x, :, :], f"Coronal (x={mid_x})", "Y (mm)", "Z (mm)"),
    (windowed[:, mid_y, :], f"Sagittal (y={mid_y})", "X (mm)", "Z (mm)"),
]

for idx, (data, title, xlabel, ylabel) in enumerate(views):
    ax = axes[1, idx]
    if idx == 0:
        ax.imshow(data, cmap='gray')
    else:
        ax.imshow(data, cmap='gray', aspect='auto')
    ax.set_title(f"Optimal Lung Window - {title}")
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
output_file = os.path.join(output_folder, "optimal_lung_window.png")
plt.savefig(output_file, dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved: {output_file}")

# Print summary
print("\n" + "=" * 80)
print("WINDOWING ANALYSIS SUMMARY")
print("=" * 80)
print(f"\n✓ Created {len(window_presets)} window preset visualizations")
print(f"✓ Created optimal lung window comparison")
print(f"\nRECOMMENDED OPTIMAL WINDOW FOR LUNG:\n  Center: {window_presets['Lung'][0]} HU\n  Width: {window_presets['Lung'][1]} HU")
print(f"\nOutput folder: {output_folder}")
print("\nVisualization files created:")
for window_name in window_presets.keys():
    file_path = os.path.join(output_folder, f"window_{window_name.replace(' ', '_').lower()}.png")
    if os.path.exists(file_path):
        print(f"  ✓ {window_name}")

print("\nNext steps:")
print("1. Review the windowing visualizations")
print("2. Take screenshots from 3D Slicer with the optimal window")
print("3. Continue with Task 1.3 (lung segmentation)")
