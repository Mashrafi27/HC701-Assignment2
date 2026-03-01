"""
Targeted script: trains only ResNet50 (Exp 2), DenseNet121 full (Exp 3),
and DenseNet121 frozen (Exp 5).

Outputs:
  - training_curves.png          (ResNet50 vs DenseNet121 — top 2)
  - confusion_matrix_exp5.png    (frozen DenseNet121 only)
  - training_curve_exp5.png      (frozen DenseNet121)
  - model_results.csv            (exp 5 row appended; exp 1/4 rows preserved)
  - summary_exp5.txt             (exp 5 accuracy / F1 / AUC to paste into report)
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
from datetime import datetime
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ───────────────────────────────────────────────────────────────────────────────
CONFIG = {
    "batch_size": 32,
    "num_epochs": 50,
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
print("TARGETED RUN: ResNet50 (Exp 2) + DenseNet121 (Exp 3) + Frozen DenseNet121 (Exp 5)")
print("=" * 80)
print(f"Device: {CONFIG['device']}")

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


class Trainer:
    def __init__(self, model, train_loader, val_loader, test_loader,
                 learning_rate, model_name, pos_weight=1.0):
        self.model = model.to(CONFIG["device"])
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.device = CONFIG["device"]
        self.model_name = model_name
        self.optimizer = optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=learning_rate, weight_decay=1e-4
        )
        self.criterion = nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight], device=self.device)
        )
        self.train_losses, self.val_losses = [], []

    def _run(self, loader, train=False):
        self.model.train(train)
        total_loss, correct, total = 0, 0, 0
        with torch.set_grad_enabled(train):
            for images, labels, _ in tqdm(loader, desc="Train" if train else "Val", leave=False):
                images = images.to(self.device)
                labels = labels.float().unsqueeze(1).to(self.device)
                if train:
                    self.optimizer.zero_grad()
                out = self.model(images)
                loss = self.criterion(out, labels)
                if train:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    self.optimizer.step()
                total_loss += loss.item()
                preds = (torch.sigmoid(out) > 0.5).float()
                correct += (preds == labels).sum().item()
                total += labels.size(0)
        return total_loss / len(loader), correct / total

    def test(self):
        self.model.eval()
        all_preds, all_labels, all_probs = [], [], []
        with torch.no_grad():
            for images, labels, _ in tqdm(self.test_loader, desc="Testing", leave=False):
                images = images.to(self.device)
                out = self.model(images)
                probs = torch.sigmoid(out).squeeze()
                preds = (probs > 0.5).float()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probs.cpu().numpy())
        return np.array(all_preds), np.array(all_labels), np.array(all_probs)

    def train_full(self):
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3)
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        best_state = None

        print(f"\n[Training {self.model_name}]")
        for epoch in tqdm(range(CONFIG["num_epochs"]), desc="Epochs", unit="epoch"):
            train_loss, train_acc = self._run(self.train_loader, train=True)
            val_loss, val_acc = self._run(self.val_loader, train=False)
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_epoch = epoch + 1
                patience_counter = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                patience_counter += 1
                if patience_counter >= CONFIG["early_stopping_patience"]:
                    print(f"  Early stopping at epoch {epoch+1} (best: {best_epoch})")
                    if best_state:
                        self.model.load_state_dict(best_state)
                    break
        print(f"  ✓ Done (best epoch: {best_epoch}, val loss: {best_val_loss:.4f})")


# ───────────────────────────────────────────────────────────────────────────────
# DATA LOADERS
# ───────────────────────────────────────────────────────────────────────────────
df = pd.read_csv(PATHS["csv_file"])
train_df = df[df['split'] == 'train'].reset_index(drop=True)
val_df   = df[df['split'] == 'val'].reset_index(drop=True)
test_df  = df[df['split'] == 'test'].reset_index(drop=True)

n_normal    = (train_df['label'] == 'NORMAL').sum()
n_pneumonia = (train_df['label'] == 'PNEUMONIA').sum()
pos_weight  = n_normal / n_pneumonia
print(f"pos_weight = {pos_weight:.4f}")

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


# ───────────────────────────────────────────────────────────────────────────────
# EXP 2: RESNET50
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("EXP 2: ResNet50 (full fine-tuning, LR=0.0001)")
print("="*80)
model2 = models.resnet50(pretrained=True)
model2.fc = nn.Linear(model2.fc.in_features, 1)
trainer2 = Trainer(model2, train_loader, val_loader, test_loader,
                   learning_rate=0.0001, model_name="ResNet50", pos_weight=pos_weight)
trainer2.train_full()
preds2, labels2, probs2 = trainer2.test()
acc2 = accuracy_score(labels2, preds2)
f1_2 = f1_score(labels2, preds2)
auc2 = roc_auc_score(labels2, probs2)
print(f"✓ ResNet50 — Acc={acc2:.4f}, F1={f1_2:.4f}, AUC={auc2:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# EXP 3: DENSENET121 (full fine-tuning)
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("EXP 3: DenseNet121 (full fine-tuning, LR=0.0001)")
print("="*80)
model3 = models.densenet121(pretrained=True)
model3.classifier = nn.Linear(model3.classifier.in_features, 1)
trainer3 = Trainer(model3, train_loader, val_loader, test_loader,
                   learning_rate=0.0001, model_name="DenseNet121", pos_weight=pos_weight)
trainer3.train_full()
preds3, labels3, probs3 = trainer3.test()
acc3 = accuracy_score(labels3, preds3)
f1_3 = f1_score(labels3, preds3)
auc3 = roc_auc_score(labels3, probs3)
print(f"✓ DenseNet121 — Acc={acc3:.4f}, F1={f1_3:.4f}, AUC={auc3:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# EXP 5: DENSENET121 (frozen backbone)
# ───────────────────────────────────────────────────────────────────────────────
print("\n" + "="*80)
print("EXP 5: DenseNet121 (frozen backbone, LR=0.001)")
print("="*80)
model5 = models.densenet121(pretrained=True)
for param in model5.parameters():
    param.requires_grad = False
model5.classifier = nn.Linear(model5.classifier.in_features, 1)
trainable5 = sum(p.numel() for p in model5.parameters() if p.requires_grad)
params5    = sum(p.numel() for p in model5.parameters())
print(f"Trainable params (head only): {trainable5:,} / {params5:,} total")

trainer5 = Trainer(model5, train_loader, val_loader, test_loader,
                   learning_rate=0.001, model_name="DenseNet121 (Frozen)", pos_weight=pos_weight)
trainer5.train_full()
preds5, labels5, probs5 = trainer5.test()
acc5 = accuracy_score(labels5, preds5)
f1_5 = f1_score(labels5, preds5)
auc5 = roc_auc_score(labels5, probs5)
print(f"✓ DenseNet121 (Frozen) — Acc={acc5:.4f}, F1={f1_5:.4f}, AUC={auc5:.4f}")


# ───────────────────────────────────────────────────────────────────────────────
# TRAINING CURVES — top 2 (ResNet50 and DenseNet121 full)
# ───────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
fig.suptitle('Training Curves — Top 2 Models', fontsize=16, fontweight='bold')

for ax, trainer, label, acc in [
    (axes[0], trainer2, "ResNet50",  acc2),
    (axes[1], trainer3, "DenseNet121", acc3),
]:
    ax.plot(trainer.train_losses, label='Train Loss', marker='o', markersize=3)
    ax.plot(trainer.val_losses,   label='Val Loss',   marker='s', markersize=3)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title(f'{label}  (Test Acc: {acc:.4f})')
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
out = PATHS["output_dir"] / "training_curves.png"
plt.savefig(out, dpi=150, bbox_inches='tight')
plt.close()
print(f"\n✓ Saved: training_curves.png")


# ───────────────────────────────────────────────────────────────────────────────
# CONFUSION MATRIX — exp 5 only (exp 1-4 from previous run)
# ───────────────────────────────────────────────────────────────────────────────
cm5 = confusion_matrix(labels5, preds5)
fig, ax = plt.subplots(figsize=(5, 4))
sns.heatmap(cm5, annot=True, fmt='d', cmap='Blues', ax=ax, cbar=False,
            xticklabels=['NORMAL', 'PNEUMONIA'],
            yticklabels=['NORMAL', 'PNEUMONIA'])
ax.set_title('Exp 5: DenseNet121 (Frozen Backbone)')
ax.set_ylabel('True')
ax.set_xlabel('Predicted')
plt.tight_layout()
plt.savefig(PATHS["output_dir"] / "confusion_matrix_exp5.png", dpi=150, bbox_inches='tight')
plt.close()
print("✓ Saved: confusion_matrix_exp5.png")


# ───────────────────────────────────────────────────────────────────────────────
# UPDATE model_results.csv (add/overwrite exp 5 row)
# ───────────────────────────────────────────────────────────────────────────────
results_path = PATHS["output_dir"] / "model_results.csv"
if results_path.exists():
    results_df = pd.read_csv(results_path)
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
results_df = pd.concat([results_df, new_row], ignore_index=True).sort_values('exp')
results_df.to_csv(results_path, index=False)
print(f"✓ Updated: model_results.csv")
print(results_df.to_string(index=False))


# ───────────────────────────────────────────────────────────────────────────────
# SUMMARY (paste into report.tex)
# ───────────────────────────────────────────────────────────────────────────────
summary = f"""
================================================================================
RESULTS TO PASTE INTO report.tex
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
================================================================================

Exp 2: ResNet50          Acc={acc2:.4f}  F1={f1_2:.4f}  AUC={auc2:.4f}
Exp 3: DenseNet121       Acc={acc3:.4f}  F1={f1_3:.4f}  AUC={auc3:.4f}
Exp 5: DenseNet121 (Frozen)  Acc={acc5:.4f}  F1={f1_5:.4f}  AUC={auc5:.4f}

Ablation gap (Exp 3 vs Exp 5):
  Accuracy: {acc3 - acc5:+.4f}
  F1:       {f1_3 - f1_5:+.4f}

Replace TBD in report.tex with:
  Exp 5 Accuracy : {acc5:.4f}
  Exp 5 F1       : {f1_5:.4f}
  Exp 5 AUC      : {auc5:.4f}
================================================================================
"""
with open(PATHS["output_dir"] / "summary_exp5.txt", "w") as f:
    f.write(summary)
print(summary)
