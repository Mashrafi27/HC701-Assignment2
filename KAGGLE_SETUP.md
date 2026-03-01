# Kaggle Dataset Setup Instructions

## Step 1: Get Kaggle API Token

1. Go to https://www.kaggle.com/account
2. Scroll down to "API" section
3. Click "Create New API Token"
4. This downloads `kaggle.json`

## Step 2: Configure Kaggle

### Option A: Using bash (Recommended)
```bash
# Copy the kaggle.json to the correct location
mkdir -p ~/.kaggle
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json

# Set proper permissions
chmod 600 ~/.kaggle/kaggle.json

# Verify setup
kaggle datasets list
```

### Option B: Manual Setup
1. Create directory: `~/.kaggle/` (if it doesn't exist)
2. Move downloaded `kaggle.json` to `~/.kaggle/kaggle.json`
3. Run in terminal:
   ```bash
   chmod 600 ~/.kaggle/kaggle.json
   ```

## Step 3: Download Dataset

Once Kaggle is configured, run:

```bash
cd /Users/mashrafi/dev/HC701/assignment2

# Download the dataset
kaggle datasets download -d paultimothymooney/chest-xray-pneumonia -p pneumonia_data

# Extract it
unzip pneumonia_data/chest-xray-pneumonia.zip -d pneumonia_data
```

## Step 4: Verify Download

After extraction, you should see:
```
pneumonia_data/chest_xray/
├── train/
│   ├── NORMAL/ (1341 images)
│   └── PNEUMONIA/ (3875 images)
├── val/
│   ├── NORMAL/ (8 images)
│   └── PNEUMONIA/ (8 images)
└── test/
    ├── NORMAL/ (234 images)
    └── PNEUMONIA/ (390 images)
```

## Troubleshooting

**Error: "Kaggle credentials not configured"**
- Ensure `kaggle.json` is in `~/.kaggle/`
- Check permissions: `ls -la ~/.kaggle/kaggle.json` should show `rw-------`

**Error: "Dataset not found"**
- Verify Kaggle is working: `kaggle datasets list`
- Check dataset name: "Chest X-Ray Images (Pneumonia)"

**Slow Download**
- Dataset is ~1.2 GB, may take several minutes
- Don't interrupt the download

## Next Steps

Once dataset is downloaded and extracted:
1. Run: `python task2_dataset_prep.py`
2. This will automatically split into 80% train / 10% val / 10% test
3. Creates `pneumonia_results/data_split.csv` with file listings
4. Ready to proceed with ML experiments (Task 2.2)
