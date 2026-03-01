# TASK 2: X-ray Pneumonia Classification - Setup Complete

## Current Status: 14/29 tasks completed (Task 1 ✓)

**Task 2 is ready to execute. Complete the steps below.**

---

## STEP 1: Download Kaggle Dataset (5 minutes)

### Quick Setup:
```bash
# 1. Get Kaggle API token from https://www.kaggle.com/account
# 2. Click "Create New API Token" → saves to ~/Downloads/kaggle.json

# 3. Configure Kaggle credentials:
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# 4. Download the dataset:
cd /Users/mashrafi/dev/HC701/assignment2
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p pneumonia_data

# 5. Extract the dataset:
unzip pneumonia_data/chest-xray-pneumonia.zip -d pneumonia_data
```

### Expected Result:
```
pneumonia_data/chest_xray/
├── train/
│   ├── NORMAL/      (1341 images)
│   └── PNEUMONIA/   (3875 images)
├── val/
│   ├── NORMAL/      (8 images)
│   └── PNEUMONIA/   (8 images)
└── test/
    ├── NORMAL/      (234 images)
    └── PNEUMONIA/   (390 images)
```

---

## STEP 2: Split Training Data (2 minutes)

```bash
cd /Users/mashrafi/dev/HC701/assignment2
conda run -n CV8501 python task2_dataset_prep.py
```

**Output:**
- `pneumonia_results/data_split.csv` - Contains 80% train / 10% val / 10% test split
- Automatically stratified by class (Normal vs Pneumonia)

---

## STEP 3: Run 5 ML Experiments (1-3 hours)

Once dataset is prepared, run:
```bash
conda run -n CV8501 python task2_ml_experiments_full.py
```

### Experiments Included:

| # | Experiment | Architecture | Rationale |
|---|-----------|--------------|-----------|
| 1 | Baseline CNN | 4-layer custom CNN | LeCun et al., 1998 - Simple baseline |
| 2 | ResNet50 | Pre-trained ResNet50 | He et al., 2016 - Residual connections |
| 3 | DenseNet121 | Pre-trained DenseNet | Huang et al., 2017 - Dense connections |
| 4 | EfficientNet-B3 | Pre-trained EfficientNet | Tan & Le, 2019 - Compound scaling |
| 5 | Ensemble | Voting (top 3 models) | Combine predictions for robustness |

### Common Hyperparameters:
- Batch size: 32
- Learning rate: 0.001 (Adam optimizer)
- Epochs: 50 (with early stopping patience=5)
- Image size: 224×224
- Data augmentation: Rotation ±10°, translation 10%, color jitter

### Generated Outputs:
- ✓ `model_results.csv` - Accuracy, F1, AUC for all experiments
- ✓ `confusion_matrices_*.png` - Per-experiment confusion matrices
- ✓ `training_curves_exp*.png` - Loss/accuracy curves for top 2 experiments
- ✓ `model_parameters_flops.csv` - Parameters and FLOPS table
- ✓ `experiment_logs.txt` - Detailed results and timestamps

---

## STEP 4: Generate Final Report

After experiments complete:
```bash
python task2_compile_report.py
```

This will:
- Compile all metrics into summary table
- Include all confusion matrices
- Add training curves
- Create final PDF report

---

## Command Summary

```bash
# Navigate to assignment directory
cd /Users/mashrafi/dev/HC701/assignment2

# 1. Download (requires Kaggle setup - see above)
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p pneumonia_data
unzip pneumonia_data/chest-xray-pneumonia.zip -d pneumonia_data

# 2. Prepare data splits
conda run -n CV8501 python task2_dataset_prep.py

# 3. Run all 5 ML experiments
conda run -n CV8501 python task2_ml_experiments_full.py

# 4. Compile report (coming next)
# python task2_compile_report.py
```

---

## Files Ready for Execution

| File | Purpose | Status |
|------|---------|--------|
| `task2_dataset_prep.py` | Download & split data | ✓ Ready |
| `task2_ml_experiments_full.py` | Run 5 experiments | ✓ Ready |
| `KAGGLE_SETUP.md` | API setup guide | ✓ Ready |
| `setup_kaggle.py` | Auto-downloader | ✓ Ready |

---

## Timeline

| Step | Time | Task |
|------|------|------|
| 1 | 5 min | Get Kaggle token & configure |
| 2 | 2 min | Download & extract dataset (~1 GB) |
| 3 | 3 min | Run data split script |
| 4 | 1-3 hrs | Run ML experiments (GPU: 1hr, CPU: 3hrs) |
| 5 | 15 min | Generate report |
| **Total** | **1-3.5 hrs** | **Complete Task 2** |

---

## Next Steps After Dataset Download

Once you've:
1. ✓ Downloaded Kaggle API token
2. ✓ Configured ~/.kaggle/kaggle.json
3. ✓ Downloaded chest X-ray dataset

**Run these commands in sequence:**

```bash
# Verify dataset is extracted
ls /Users/mashrafi/dev/HC701/assignment2/pneumonia_data/chest_xray/train/NORMAL/ | wc -l
# Should show: 1341

# Prepare splits
conda run -n CV8501 python task2_dataset_prep.py

# Run experiments (this will take 1-3 hours)
conda run -n CV8501 python task2_ml_experiments_full.py
```

Then report back when ready for the final report compilation!

---

## Troubleshooting

**Q: "Dataset not found" error**
- A: Verify the zip was extracted: `ls pneumonia_data/chest_xray/train/`

**Q: Script says "CSV file not found"**
- A: Run `python task2_dataset_prep.py` first to generate the splits

**Q: GPU not detected**
- A: Script will automatically fall back to CPU (slower but works)

**Q: Kaggle CLI says "credentials not configured"**
- A: Ensure `chmod 600 ~/.kaggle/kaggle.json` was run

---

## Assignment Progress

```
Task 1: CT Lung Windowing & Segmentation ✓ COMPLETE
  └─ 14/14 subtasks done
  └─ All outputs saved to segmentation_results/

Task 2: Pneumonia Classification (70% of grade)
  ├─ 2.1a: Download dataset → AWAITING KAGGLE SETUP
  ├─ 2.1b: Split data → READY (after download)
  ├─ 2.1c: Generate CSV → READY (after download)
  ├─ 2.2a-j: Run 5 ML experiments → READY (after download)
  └─ Final Report → READY (after experiments)
```

**Total Progress: 14/29 tasks (48%) - Ready for Task 2 execution!**

