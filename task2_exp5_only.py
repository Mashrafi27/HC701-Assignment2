"""
Experiment 5 only: DenseNet121 with frozen backbone (feature extraction ablation).
Run this on HPC after exp 1-4 are done to get Exp 5 results without retraining.
Results are appended to (or update row 5 of) model_results.csv.
"""

import os
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
import seaborn as sns
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
# CONFIG (must match main script)
# ───────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
    "learning_rate": 0.001,   # head-only training — higher LR is fine
    "image_size": 224,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    "seed": 42,
    "early_stopping_patience": 5,
}

script_dir = os.path.dirname(os.path.abspath(__file__))
PATHS = {
    "data_dir":   Path(script_dir) / "pneumonia_data" / "chest_xray",
    "csv_file":   Path(script_dir) / "pneumonia_results" / "data_split.csv",
    "output_dir": Path(script_dir) / "pneumonia_results" / "ml_experiments",
}
PATHS["output_dir"].mkdir(parents=True, exist_ok=True)

torch.manual_seed(CONFIG["seed"])
np.random.seed(CONFIG["seed"])

print("=" * 80)
print("EXPERIMENT 5: DENSENET121 (FROZEN BACKBONE)")
print("=" * 80)
print(f"Device: {CONFIG['device']}")

# ───────────────────────────────────────────────────────────────────────────────
# DATASET
# ───────────────────────────────────────────────────────────────────────────────
class PneumoniaDataset(Dataset):
    def __init__(self, dataframe, data_dir, transform=None):
        self.dataframe = dataframe
        self.data_dir = Path(data_dir)
        self.transform = transform

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, idx):
        row = self.dataframe.iloc[idx]
        img_path = self.data_dir / 'train' / row['label'] / row['filename']
        label = 1 if row['label'] == 'PNEUMONIA' else 0
        try:
            image = Image.open(img_path).convert('RGB')
        except:
            image = Image.new('RGB', (CONFIG["image_size"], CONFIG["image_size"]))
        if self.transform:
            image = self.transform(image)
        return image, label, str(img_path)

# ───────────────────────────────────────────────────────────────────────────────
# TRAINER
# ───────────────────────────────────────────────────────────────────────────────
class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader, config, model_name, pos_weight=1.0):
        self.model = model.to(config["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.config = config
        self.device = config["device"]
        self.model_name = model_name
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config["learning_rate"], weight_decay=1e-4
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=config["device"])
        )
        self.train_losses, self.val_losses = [], []
        self.train_accs, self.val_accs = [], []

    def _run_loader(self, loader, train=False):
        self.model.train(train)
        total_loss, correct, total = 0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels, _ in tqdm(loader, desc="Train" if train else "Val", leave=False):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                if train:
                    self.optimizer.zero_grad()
                outputs = self.model(images)
                loss = self.criterion(outputs, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    self.optimizer.step()
                total_loss += loss.item()
                predictions = (torch.sigmoid(outputs) > 0.5).float()
                correct += (predictions == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def test(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_loader, desc="Testing", leave=False):
                images = images.to(self.device)
                outputs = self.model(images)
                probs = torch.sigmoid(outputs).squeeze()
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def train_full(self):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )
        best_val_loss, patience_counter, best_epoch, best_state = float('inf'), 0, 0, None
        print(f"\n[Training {self.model_name}]")
        for epoch in tqdm(range(self.config["num_epochs"]), desc="Epochs", unit="epoch"):
            train_loss, train_acc = self._run_loader(self.train_loader, train=True)
            val_loss, val_acc = self._run_loader(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accs.append(train_acc)
            self.val_accs.append(val_acc)
            scheduler.step(val_loss)
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= self.config["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    if best_state:
                        self.model.load_state_dict(best_state)
                    break
        print(f"  ✓ Done (best epoch: {best_epoch}, val loss: {best_val_loss:.4f})")

# ───────────────────────────────────────────────────────────────────────────────
# MAIN
# ───────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(PATHS["csv_file"])
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'val'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

n_normal    = len(train_df[train_df['label'] == 'NORMAL'])
n_pneumonia = len(train_df[train_df['label'] == 'PNEUMONIA'])
pos_weight  = n_normal / n_pneumonia
print(f"Class balance: {n_normal} NORMAL / {n_pneumonia} PNEUMONIA  pos_weight={pos_weight:.4f}")

train_transform = transforms.Compose([
    transforms.RandomRotation(10),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])
val_test_transform = transforms.Compose([
    transforms.Resize((CONFIG["image_size"], CONFIG["image_size"])),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

train_loader = DataLoader(PneumoniaDataset(train_df, PATHS['data_dir'], train_transform),
                          batch_size=CONFIG["batch_size"], shuffle=True, num_workers=4)
val_loader   = DataLoader(PneumoniaDataset(val_df,   PATHS['data_dir'], val_test_transform),
                          batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)
test_loader  = DataLoader(PneumoniaDataset(test_df,  PATHS['data_dir'], val_test_transform),
                          batch_size=CONFIG["batch_size"], shuffle=False, num_workers=4)

# Build model — freeze backbone, train only classifier
model5 = models.densenet121(pretrained=True)
for param in model5.parameters():
    param.requires_grad = False
model5.classifier = nn.Linear(model5.classifier.in_features, 1)  # trainable by default

params5    = sum(p.numel() for p in model5.parameters())
trainable5 = sum(p.numel() for p in model5.parameters() if p.requires_grad)
print(f"Total params: {params5:,} | Trainable (head only): {trainable5:,}")

trainer5 = Trainer(model5, train_loader, val_loader, test_loader,
                   CONFIG, "DenseNet121 (Frozen)", pos_weight=pos_weight)
trainer5.train_full()

preds5, labels5, probs5 = trainer5.test()
acc5  = accuracy_score(labels5, preds5)
f1_5  = f1_score(labels5, preds5)
auc5  = roc_auc_score(labels5, probs5)

print(f"\n{'='*60}")
print(f"EXP 5 RESULTS:")
print(f"  Accuracy : {acc5:.4f}")
print(f"  F1 Score : {f1_5:.4f}")
print(f"  AUC      : {auc5:.4f}")
print(f"  Params   : {params5:,} total / {trainable5:,} trainable")
print(f"{'='*60}")

# Save confusion matrix for exp 5
cm5 = confusion_matrix(labels5, preds5)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm5, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False)
ax.set_title('Exp 5: DenseNet121 (Frozen Backbone)')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(PATHS["output_dir"] / "confusion_matrix_exp5.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrix_exp5.png")

# Save training curve for exp 5
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(trainer5.train_losses, label='Train Loss', marker='o', markersize=3)
ax.plot(trainer5.val_losses,   label='Val Loss',   marker='s', markersize=3)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('DenseNet121 (Frozen) Training Curve')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(PATHS["output_dir"] / "training_curve_exp5.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: training_curve_exp5.png")

# Update model_results.csv
results_path = PATHS["output_dir"] / "model_results.csv"
if results_path.exists():
    results_df = pd.read_csv(results_path)
    # Remove old exp5 row if present
    results_df = results_df[results_df['exp'] != 5]
else:
    results_df = pd.DataFrame(columns=['exp','model','test_accuracy','test_f1','test_auc','parameters','flops'])

new_row = pd.DataFrame([{
    'exp': 5,
    'model': 'DenseNet121 (Frozen)',
    'test_accuracy': acc5,
    'test_f1': f1_5,
    'test_auc': auc5,
    'parameters': params5,
    'flops': 'N/A',
}])
results_df = pd.concat([results_df, new_row], ignore_index=True)
results_df.to_csv(results_path, index=False)
print(f"✓ Updated: model_results.csv")
print(results_df.to_string(index=False))
