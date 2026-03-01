"""
Task 2.2: Machine Learning Experiments for Pneumonia Classification
Framework for 5 experiments with different architectures

EXPERIMENTAL DESIGN:
1. Baseline CNN: Simple custom architecture (reference: LeCun et al., 1998)
2. ResNet50: Deep residual networks with transfer learning
3. DenseNet121: Dense connections with pre-trained weights
4. EfficientNet-B3: Compound scaling approach (Tan & Le, 2019)
5. Ensemble: Voting classifier combining top 3 models

All experiments use:
- Data augmentation: RandomRotation(10), RandomAffine(translate=(0.1,0.1)), ColorJitter
- Optimizer: Adam with learning rate 0.001
- Loss: Binary Cross-Entropy (for binary classification: Normal vs Pneumonia)
- Batch size: 32
- Epochs: 50 (with early stopping patience=5)
"""

import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from datetime import datetime
import time

# Configuration
CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "early_stopping_patience": 5,
}

OUTPUT_DIR = Path("/Users/mashrafi/dev/HC701/assignment2/pneumonia_results/ml_experiments")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("TASK 2.2: PNEUMONIA CLASSIFICATION - ML EXPERIMENTS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Device: {CONFIG['device']}")
print(f"  Batch Size: {CONFIG['batch_size']}")
print(f"  Learning Rate: {CONFIG['learning_rate']}")
print(f"  Epochs: {CONFIG['num_epochs']}")
print(f"  Image Size: {CONFIG['image_size']}x{CONFIG['image_size']}")

# PLACEHOLDER: Data loading and experiment execution
print("\n" + "="*80)
print("READY FOR EXECUTION")
print("="*80)

print("""
Once the dataset is downloaded and prepared:

1. Ensure data_split.csv exists with train/val/test splits
2. Run: python task2_ml_experiments_full.py

This will execute all 5 experiments in sequence:
  ✓ Exp 1: Baseline CNN
  ✓ Exp 2: ResNet50
  ✓ Exp 3: DenseNet121
  ✓ Exp 4: EfficientNet-B3
  ✓ Exp 5: Ensemble

Output files will be generated:
  - model_results.csv (metrics table)
  - experiment_logs.txt (detailed results)
  - confusion_matrices_*.png (confusion matrices for all experiments)
  - training_curves_exp*.png (loss/accuracy curves for top 2 experiments)
  - model_parameters_flops.csv (model complexity metrics)
""")

print("\nEstimated time: 30-60 minutes on GPU, 2-3 hours on CPU")
print("="*80)
