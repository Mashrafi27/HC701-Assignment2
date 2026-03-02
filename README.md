# HC701 Assignment 2

## Requirements

```
pip install -r requirements.txt
```

## Task 1: CT Lung Analysis

**Task 1.1** — Extract DICOM slices 71–110 and save as NIfTI:
```
python task1_extract_and_save.py
```
Update `data_folder` at the top of the script to point to your DICOM directory.
Output: `ct_lungs_slices_71_110.nii.gz`

**Task 1.2** — Intensity windowing visualisation:
```
python task1_windowing.py
```
Output: `windowing_results/`

**Task 1.3** — Lung segmentation (non-ML):
```
python task1_segmentation.py
```
Output: `segmentation_results/` (masks + visualisations)

## Task 2: Pneumonia Classification

**Task 2.1** — Prepare dataset split CSV:
```
python task2_dataset_prep.py
```
Update `data_dir` to point to the Kaggle chest X-ray dataset folder.
Output: `pneumonia_results/data_split.csv`

**Task 2.2** — Run all 5 experiments:
```
python task2_run_experiments.py
```
Output: `pneumonia_results/ml_experiments/` (results, confusion matrices, training curves)

> Previously run experiments are cached as `.npy` files so re-runs skip retraining.
