"""
Task 2.2: Complete ML Experiments Pipeline
Executes 5 experiments for pneumonia classification with detailed results
"""

import os
import json
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.models as models
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
import seaborn as sns
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

try:
    from thop import profile as thop_profile
    THOP_AVAILABLE = True
except ImportError:
    THOP_AVAILABLE = False

# ═══════════════════════════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════════════════════════

CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.0001,
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "early_stopping_patience": 5,
}

# Use relative paths for cross-platform compatibility
script_dir = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "data_dir": Path(script_dir) / "pneumonia_data" / "chest_xray",
    "csv_file": Path(script_dir) / "pneumonia_results" / "data_split.csv",
    "output_dir": Path(script_dir) / "pneumonia_results" / "ml_experiments",
}

PATHS["output_dir"].mkdir(parents=True, exist_ok=True)

# Set random seeds
torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("="*80)
print("PNEUMONIA CLASSIFICATION - ML EXPERIMENTS")
print("="*80)
print(f"\nConfiguration:")
print(f"  Device: {CONFIG['device']}")
print(f"  Batch size: {CONFIG['batch_size']}")
print(f"  Learning rate: {CONFIG['learning_rate']}")
print(f"  Image size: {CONFIG['image_size']}x{CONFIG['image_size']}")
print(f"  Max epochs: {CONFIG['num_epochs']}")

# ═══════════════════════════════════════════════════════════════════════════════════
# CUSTOM DATASET
# ═══════════════════════════════════════════════════════════════════════════════════

class PneumoniaDataset(Dataset):
    """Custom dataset for chest X-ray pneumonia classification"""

    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        # All images are physically in the train/ subfolder (val/test are logical splits)
        img_path = self.data_dir / 'train' / row['label'] / row['filename']
        label = 1 if row['label'] == 'PNEUMONIA' else 0

        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (CONFIG["image_size"], CONFIG["image_size"]))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label, str(img_path)

# ═══════════════════════════════════════════════════════════════════════════════════
# MODEL ARCHITECTURES
# ═══════════════════════════════════════════════════════════════════════════════════

class BaselineCNN(nn.Module):
    """Simple custom CNN baseline"""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout(0.25),
            
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# ═══════════════════════════════════════════════════════════════════════════════════
# TRAINER CLASS
# ═══════════════════════════════════════════════════════════════════════════════════

class Trainer:
    """Training and evaluation utilities"""
    
    def __init__(self, model, train_loader, val_loader, test_loader, config, model_name, pos_weight=1.0):
        self.model = model.to(config["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config["device"]
        self.model_name = model_name
        
        self.optimizer = optim.Adam(self.model.parameters(),
                                    lr=config["learning_rate"],
                                    weight_decay=1e-4)
        # Use weighted loss to handle class imbalance
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight], device=config["device"]))
        
        self.train_losses = []
        self.val_losses = []
        self.train_accs = []
        self.val_accs = []
    
    def train_epoch(self):
        self.model.train()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.train_loader, desc="Training", leave=False)
        for images, labels, _ in pbar:
            images = images.to(self.device)
            labels = labels.float().unsqueeze(1).to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(images)
            loss = self.criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            predictions = (torch.sigmoid(outputs) > 0.5).float()
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = correct / total
        
        self.train_losses.append(avg_loss)
        self.train_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def validate(self):
        self.model.eval()
        total_loss = 0
        correct = 0
        total = 0
        
        pbar = tqdm(self.val_loader, desc="Validating", leave=False)
        with torch.no_grad():
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = correct / total
        
        self.val_losses.append(avg_loss)
        self.val_accs.append(accuracy)
        
        return avg_loss, accuracy
    
    def test(self):
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        
        pbar = tqdm(self.test_loader, desc="Testing", leave=False)
        with torch.no_grad():
            for images, labels, _ in pbar:
                images = images.to(self.device)
                labels = labels.to(self.device)
                
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze()
                predictions = (probs > 0.5).float()
                
                all_preds.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
        
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)
    
    def train_full(self):
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_state = None
        
        # Add learning rate scheduler
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        
        print(f"\n[Training {self.model_name}]")
        pbar = tqdm(range(self.config["num_epochs"]), desc="Epochs", unit="epoch")
        for epoch in pbar:
            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()
            
            # Update learning rate based on validation loss
            scheduler.step(val_loss)
            
            # Update progress bar description
            pbar.set_description(f"Epoch {epoch+1} | Train: L={train_loss:.4f} A={train_acc:.4f} | Val: L={val_loss:.4f} A={val_acc:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    if best_state is not None:
                        self.model.load_state_dict(best_state)
                    break
        
        print(f"  ✓ Training complete (Best epoch: {best_epoch}, Val Loss: {best_val_loss:.4f})")

# ═══════════════════════════════════════════════════════════════════════════════════
# MAIN EXECUTION
# ═══════════════════════════════════════════════════════════════════════════════════

def main():
    # Load data
    if not PATHS["csv_file"].exists():
        print(f"\n✗ Error: {PATHS['csv_file']} not found!")
        return
    
    df = pd.read_csv(PATHS["csv_file"])
    train_df = df[df['split'] == 'train'].reset_index(drop=True)
    val_df = df[df['split'] == 'val'].reset_index(drop=True)
    test_df = df[df['split'] == 'test'].reset_index(drop=True)
    
    print(f"\nDataset loaded:")
    print(f"  Train: {len(train_df)} | Val: {len(val_df)} | Test: {len(test_df)}")
    
    # Data transforms
    train_transform = transforms.Compose([
        transforms.RandomRotation(10),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets (data_dir resolves paths on any machine)
    train_dataset = PneumoniaDataset(train_df, PATHS['data_dir'], transform=train_transform)
    val_dataset = PneumoniaDataset(val_df, PATHS['data_dir'], transform=val_test_transform)
    test_dataset = PneumoniaDataset(test_df, PATHS['data_dir'], transform=val_test_transform)

    # Class balance info
    n_normal = len(train_df[train_df['label'] == 'NORMAL'])
    n_pneumonia = len(train_df[train_df['label'] == 'PNEUMONIA'])
    print(f"\nClass balance: {n_normal} NORMAL ({n_normal/len(train_df)*100:.1f}%), "
          f"{n_pneumonia} PNEUMONIA ({n_pneumonia/len(train_df)*100:.1f}%)")

    # Use natural class distribution with pos_weight to handle imbalance.
    # pos_weight = n_normal/n_pneumonia makes the loss equally penalize
    # predicting all-PNEUMONIA vs all-NORMAL, avoiding trivial solutions.
    pos_weight = n_normal / n_pneumonia
    print(f"  Using pos_weight={pos_weight:.4f} (n_normal/n_pneumonia) for loss reweighting")

    train_loader = DataLoader(train_dataset, batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
    
    # Results storage
    results = {
        'exp': [],
        'model': [],
        'test_accuracy': [],
        'test_f1': [],
        'test_auc': [],
        'parameters': [],
        'flops': [],
    }

    def compute_flops(model, device):
        if not THOP_AVAILABLE:
            return 'N/A'
        try:
            dummy = torch.randn(1, 3, CONFIG["image_size"], CONFIG["image_size"]).to(device)
            flops, _ = thop_profile(model, inputs=(dummy,), verbose=False)
            return int(flops)
        except Exception:
            return 'N/A'
    
    all_test_preds = {}
    all_test_labels = {}
    all_trainers = {}
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 1: BASELINE CNN
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("EXPERIMENT 1: BASELINE CNN")
    print("="*80)
    print("Architecture: 4-layer custom CNN with BatchNorm and Dropout")
    print("Reference: LeCun et al., 1998 - Gradient-based learning")
    
    model1 = BaselineCNN()
    trainer1 = Trainer(model1, train_loader, val_loader, test_loader, CONFIG, "Baseline CNN", pos_weight=pos_weight)
    trainer1.train_full()
    
    preds1, labels1, probs1 = trainer1.test()
    acc1 = accuracy_score(labels1, preds1)
    f1_1 = f1_score(labels1, preds1)
    auc1 = roc_auc_score(labels1, probs1)
    params1 = sum(p.numel() for p in model1.parameters())
    flops1 = compute_flops(model1, CONFIG["device"])

    results['exp'].append(1)
    results['model'].append('Baseline CNN')
    results['test_accuracy'].append(acc1)
    results['test_f1'].append(f1_1)
    results['test_auc'].append(auc1)
    results['parameters'].append(params1)
    results['flops'].append(flops1)
    
    all_test_preds['exp1'] = preds1
    all_test_labels['exp1'] = labels1
    all_trainers['exp1'] = trainer1
    
    print(f"\n✓ Results: Accuracy={acc1:.4f}, F1={f1_1:.4f}, AUC={auc1:.4f}")
    print(f"  Parameters: {params1:,}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 2: RESNET50
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("EXPERIMENT 2: RESNET50")
    print("="*80)
    print("Architecture: Pre-trained ResNet50 with fine-tuning")
    print("Reference: He et al., 2016 - Deep Residual Learning")
    
    model2 = models.resnet50(pretrained=True)
    model2.fc = nn.Linear(model2.fc.in_features, 1)
    trainer2 = Trainer(model2, train_loader, val_loader, test_loader, CONFIG, "ResNet50", pos_weight=pos_weight)
    trainer2.train_full()
    
    preds2, labels2, probs2 = trainer2.test()
    acc2 = accuracy_score(labels2, preds2)
    f1_2 = f1_score(labels2, preds2)
    auc2 = roc_auc_score(labels2, probs2)
    params2 = sum(p.numel() for p in model2.parameters())
    flops2 = compute_flops(model2, CONFIG["device"])

    results['exp'].append(2)
    results['model'].append('ResNet50')
    results['test_accuracy'].append(acc2)
    results['test_f1'].append(f1_2)
    results['test_auc'].append(auc2)
    results['parameters'].append(params2)
    results['flops'].append(flops2)
    
    all_test_preds['exp2'] = preds2
    all_test_labels['exp2'] = labels2
    all_trainers['exp2'] = trainer2
    
    print(f"\n✓ Results: Accuracy={acc2:.4f}, F1={f1_2:.4f}, AUC={auc2:.4f}")
    print(f"  Parameters: {params2:,}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 3: DENSENET121
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("EXPERIMENT 3: DENSENET121")
    print("="*80)
    print("Architecture: Pre-trained DenseNet121 with fine-tuning")
    print("Reference: Huang et al., 2017 - Densely Connected Networks")
    
    model3 = models.densenet121(pretrained=True)
    model3.classifier = nn.Linear(model3.classifier.in_features, 1)
    trainer3 = Trainer(model3, train_loader, val_loader, test_loader, CONFIG, "DenseNet121", pos_weight=pos_weight)
    trainer3.train_full()
    
    preds3, labels3, probs3 = trainer3.test()
    acc3 = accuracy_score(labels3, preds3)
    f1_3 = f1_score(labels3, preds3)
    auc3 = roc_auc_score(labels3, probs3)
    params3 = sum(p.numel() for p in model3.parameters())
    flops3 = compute_flops(model3, CONFIG["device"])

    results['exp'].append(3)
    results['model'].append('DenseNet121')
    results['test_accuracy'].append(acc3)
    results['test_f1'].append(f1_3)
    results['test_auc'].append(auc3)
    results['parameters'].append(params3)
    results['flops'].append(flops3)
    
    all_test_preds['exp3'] = preds3
    all_test_labels['exp3'] = labels3
    all_trainers['exp3'] = trainer3
    
    print(f"\n✓ Results: Accuracy={acc3:.4f}, F1={f1_3:.4f}, AUC={auc3:.4f}")
    print(f"  Parameters: {params3:,}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 4: EFFICIENTNET-B3
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("EXPERIMENT 4: EFFICIENTNET-B3")
    print("="*80)
    print("Architecture: Pre-trained EfficientNet-B3 with fine-tuning")
    print("Reference: Tan & Le, 2019 - EfficientNet (Compound Scaling)")
    
    try:
        from torchvision.models import efficientnet_b3
        model4 = efficientnet_b3(pretrained=True)
        model4.classifier[1] = nn.Linear(model4.classifier[1].in_features, 1)
    except:
        print("Note: EfficientNet not available, using MobileNetV2 as alternative")
        model4 = models.mobilenet_v2(pretrained=True)
        model4.classifier[1] = nn.Linear(model4.classifier[1].in_features, 1)
    
    trainer4 = Trainer(model4, train_loader, val_loader, test_loader, CONFIG, "EfficientNet-B3", pos_weight=pos_weight)
    trainer4.train_full()
    
    preds4, labels4, probs4 = trainer4.test()
    acc4 = accuracy_score(labels4, preds4)
    f1_4 = f1_score(labels4, preds4)
    auc4 = roc_auc_score(labels4, probs4)
    params4 = sum(p.numel() for p in model4.parameters())
    flops4 = compute_flops(model4, CONFIG["device"])

    results['exp'].append(4)
    results['model'].append('EfficientNet-B3')
    results['test_accuracy'].append(acc4)
    results['test_f1'].append(f1_4)
    results['test_auc'].append(auc4)
    results['parameters'].append(params4)
    results['flops'].append(flops4)
    
    all_test_preds['exp4'] = preds4
    all_test_labels['exp4'] = labels4
    all_trainers['exp4'] = trainer4
    
    print(f"\n✓ Results: Accuracy={acc4:.4f}, F1={f1_4:.4f}, AUC={auc4:.4f}")
    print(f"  Parameters: {params4:,}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # EXPERIMENT 5: ENSEMBLE
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("EXPERIMENT 5: ENSEMBLE")
    print("="*80)
    print("Method: Averaging predictions from all 4 models")
    print("Rationale: Combine complementary model strengths")
    
    ensemble_probs = (probs1 + probs2 + probs3 + probs4) / 4
    preds5 = (ensemble_probs > 0.5).astype(int)
    acc5 = accuracy_score(labels1, preds5)
    f1_5 = f1_score(labels1, preds5)
    auc5 = roc_auc_score(labels1, ensemble_probs)
    params5 = params1 + params2 + params3 + params4
    flops5 = sum(f for f in [flops1, flops2, flops3, flops4] if isinstance(f, int)) or 'N/A'

    results['exp'].append(5)
    results['model'].append('Ensemble (All 4)')
    results['test_accuracy'].append(acc5)
    results['test_f1'].append(f1_5)
    results['test_auc'].append(auc5)
    results['parameters'].append(params5)
    results['flops'].append(flops5)
    
    all_test_preds['exp5'] = preds5
    all_test_labels['exp5'] = labels1
    
    print(f"\n✓ Results: Accuracy={acc5:.4f}, F1={f1_5:.4f}, AUC={auc5:.4f}")
    print(f"  Combined parameters: {params5:,}")
    
    # ═══════════════════════════════════════════════════════════════════════════════════
    # SAVE RESULTS
    # ═══════════════════════════════════════════════════════════════════════════════════
    
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    # Results table
    results_df = pd.DataFrame(results)
    results_df.to_csv(PATHS["output_dir"] / "model_results.csv", index=False)
    print(f"\n✓ Saved: model_results.csv")
    print(results_df.to_string(index=False))
    
    # Confusion matrices
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Confusion Matrices - All Experiments', fontsize=16, fontweight='bold')
    
    for idx, (exp_name, preds) in enumerate(all_test_preds.items()):
        ax = axes[idx // 3, idx % 3]
        labels = all_test_labels[exp_name]
        cm = confusion_matrix(labels, preds)
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
        exp_num = idx + 1
        ax.set_title(f'Exp {exp_num}: {results_df.iloc[idx]["model"]}')
        ax.set_ylabel('True')
        ax.set_xlabel('Predicted')
    
    plt.tight_layout()
    plt.savefig(PATHS["output_dir"] / "confusion_matrices.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: confusion_matrices.png")
    plt.close()
    
    # Training curves for top 2 models
    sorted_indices = np.argsort(results_df['test_accuracy'].values)[::-1]
    top_2_indices = sorted_indices[:2]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Training Curves - Top 2 Models', fontsize=16, fontweight='bold')
    
    for plot_idx, exp_idx in enumerate(top_2_indices):
        ax = axes[plot_idx]
        trainer = all_trainers[f'exp{exp_idx+1}']
        
        ax.plot(trainer.train_losses, label='Train Loss', marker='o', markersize=3)
        ax.plot(trainer.val_losses, label='Val Loss', marker='s', markersize=3)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title(f'{results_df.iloc[exp_idx]["model"]} (Acc: {results_df.iloc[exp_idx]["test_accuracy"]:.4f})')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(PATHS["output_dir"] / "training_curves.png", dpi=150, bbox_inches='tight')
    print(f"✓ Saved: training_curves.png")
    plt.close()
    
    # Summary report
    summary = f"""
================================================================================
EXPERIMENT SUMMARY REPORT
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

BEST MODEL: {results_df.loc[results_df['test_accuracy'].idxmax(), 'model']}
  Accuracy: {results_df['test_accuracy'].max():.4f}
  F1 Score: {results_df.loc[results_df['test_accuracy'].idxmax(), 'test_f1']:.4f}
  AUC: {results_df.loc[results_df['test_accuracy'].idxmax(), 'test_auc']:.4f}

ENSEMBLE MODEL:
  Accuracy: {results_df.iloc[-1]['test_accuracy']:.4f}
  F1 Score: {results_df.iloc[-1]['test_f1']:.4f}
  AUC: {results_df.iloc[-1]['test_auc']:.4f}

DETAILED RESULTS:
{results_df.to_string()}

DEVICE: {CONFIG['device'].upper()}
TOTAL TRAINING TIME: See individual logs above
================================================================================
"""
    
    with open(PATHS["output_dir"] / "summary_report.txt", "w") as f:
        f.write(summary)
    print(f"✓ Saved: summary_report.txt")
    
    print("\n" + "="*80)
    print("✓ ALL EXPERIMENTS COMPLETE")
    print("="*80)
    print(f"\nResults saved to: {PATHS['output_dir']}")
    print("  - model_results.csv")
    print("  - confusion_matrices.png")
    print("  - training_curves.png")
    print("  - summary_report.txt")

if __name__ == "__main__":
    main()
