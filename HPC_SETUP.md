# HPC GPU Setup Guide

## Summary
- ✅ **Git Initialized** - All code and Task 1 results committed
- ✅ **Task 1 Complete** - CT lung segmentation done locally
- ✅ **Task 2 Data Ready** - X-ray dataset downloaded & split (5,216 images total)
- 🔄 **Task 2 ML Experiments** - Ready to run on HPC GPU

## Step 1: Push to GitHub

```bash
# Create a new repository on GitHub (https://github.com/new)
# Name: HC701-Assignment2
# Then run:

cd /Users/mashrafi/dev/HC701/assignment2
git remote add origin https://github.com/YOUR_USERNAME/HC701-Assignment2.git
git branch -M main
git push -u origin main
```

## Step 2: Clone on HPC and Setup

```bash
# Login to HPC
ssh username@hpc.server.com

# Clone repository
cd ~/
git clone https://github.com/YOUR_USERNAME/HC701-Assignment2.git
cd HC701-Assignment2

# Load modules (EXAMPLE - adjust for your HPC)
module load cuda/11.8
module load cudnn/8.6
module load python/3.10
module load pytorch/2.0-gpu

# Or create conda environment with GPU support
conda create -n ml_gpu pytorch::pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda activate ml_gpu

# Install requirements
pip install pandas numpy matplotlib scikit-learn opencv-python pydicom nibabel
```

## Step 3: Download Dataset on HPC

```bash
# Copy Kaggle credentials from local machine
scp ~/.kaggle/kaggle.json username@hpc.server.com:~/.kaggle/kaggle.json
chmod 600 ~/.kaggle/kaggle.json

# On HPC, download dataset
cd ~/HC701-Assignment2
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p pneumonia_data
unzip pneumonia_data/chest-xray-pneumonia.zip -d pneumonia_data
rm -rf pneumonia_data/chest_xray/chest_xray pneumonia_data/chest_xray/__MACOSX
```

## Step 4: Run ML Experiments on GPU

### Quick Test (Single Model)
```bash
# Test with 1 epoch on GPU
python task2_run_experiments.py --epochs 1 --device cuda
```

### Full Run (All 5 Models)
```bash
# Run with GPU acceleration
python task2_run_experiments.py --epochs 50 --device cuda

# Or submit as batch job:
sbatch submit_job.sh
```

### Example SLURM Job Script (submit_job.sh)
```bash
#!/bin/bash
#SBATCH --job-name=ml_pneumonia
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu

module load cuda/11.8
source activate ml_gpu

cd $SLURM_SUBMIT_DIR
python task2_run_experiments.py --epochs 50 --device cuda --output_dir results_gpu
```

## Step 5: Download Results

After experiments complete:

```bash
# On local machine
scp -r username@hpc.server.com:~/HC701-Assignment2/pneumonia_results/ml_experiments ./

# Push results back to GitHub
git add pneumonia_results/
git commit -m "ML experiments complete on HPC GPU"
git push
```

## Estimated Time

| Setup | Device | Time |
|-------|--------|------|
| Model Training | GPU (V100/A100) | 30-45 min |
| Model Training | GPU (RTX3090) | 45-60 min |
| Model Training | CPU | 2-3 hours |

## Files in Repository

```
HC701-Assignment2/
├── Task 1 (Complete)
│   ├── task1_dicom_inspection.py         ✓
│   ├── task1_extract_and_save.py         ✓
│   ├── task1_windowing.py                ✓
│   ├── task1_segmentation.py             ✓
│   ├── task1_3d_visualization.py         ✓
│   ├── ct_lungs_slices_71_110.nii.gz     ✓
│   ├── windowing_results/                ✓
│   └── segmentation_results/             ✓
│
├── Task 2 (In Progress)
│   ├── task2_dataset_prep.py             ✓
│   ├── task2_ml_experiments_full.py      ✓
│   ├── task2_run_experiments.py          (GPU)
│   ├── pneumonia_data/                   (5.2GB - local only)
│   └── pneumonia_results/
│       ├── data_split.csv                ✓
│       └── ml_experiments/               (to be generated)
│
└── Documentation
    ├── KAGGLE_SETUP.md                   ✓
    ├── TASK2_SETUP_GUIDE.md              ✓
    └── .gitignore                        ✓
```

## Notes for HPC

1. **Large Files**: Dataset (5.2GB) is excluded from Git via `.gitignore`
   - Download on HPC or transfer separately if needed
   
2. **GPU Memory**: Models require ~4GB VRAM (fits on most modern GPUs)
   - Can run 3-4 models in parallel with 16GB+ GPU

3. **Data Caching**: Set `num_workers=4` in DataLoader for faster I/O on HPC

4. **Results**: All outputs (metrics, matrices, curves) will be in `pneumonia_results/ml_experiments/`

## Quick Reference Commands

```bash
# Check GPU availability
nvidia-smi

# Run with debug output
python task2_run_experiments.py --verbose

# Run specific experiments only
python task2_run_experiments.py --experiments baseline resnet50

# Monitor training
tail -f pneumonia_results/experiment_logs.txt
```
