"""
Compute FLOPs and parameter counts for all 5 models using thop.
No training — just a single forward pass with a dummy input.
"""

import torch
import torch.nn as nn
import torchvision.models as models
from thop import profile

device = "cuda" if torch.cuda.is_available() else "cpu"
dummy = torch.randn(1, 3, 224, 224).to(device)

# ── Baseline CNN (same architecture as in the main script) ──────────────────
class BaselineCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.ReLU(inplace=True), nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256), nn.ReLU(inplace=True), nn.AdaptiveAvgPool2d((1, 1))
        )
        self.classifier = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(inplace=True), nn.Dropout(0.5), nn.Linear(128, 1)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        return self.classifier(x)


def get_flops(model, name):
    model = model.to(device).eval()
    flops, params = profile(model, inputs=(dummy,), verbose=False)
    gflops = flops / 1e9
    print(f"  {name:<30}  params={params:>12,.0f}  FLOPs={gflops:.4f} GFLOPs")
    return params, gflops


print("=" * 70)
print("MODEL FLOPs AND PARAMETER COUNTS  (input: 1 × 3 × 224 × 224)")
print("=" * 70)

# Exp 1
m1 = BaselineCNN()
p1, f1 = get_flops(m1, "Exp 1: Baseline CNN")

# Exp 2
m2 = models.resnet50(pretrained=False)
m2.fc = nn.Linear(m2.fc.in_features, 1)
p2, f2 = get_flops(m2, "Exp 2: ResNet50")

# Exp 3
m3 = models.densenet121(pretrained=False)
m3.classifier = nn.Linear(m3.classifier.in_features, 1)
p3, f3 = get_flops(m3, "Exp 3: DenseNet121 (full)")

# Exp 4
try:
    from torchvision.models import efficientnet_b3
    m4 = efficientnet_b3(pretrained=False)
    m4.classifier[1] = nn.Linear(m4.classifier[1].in_features, 1)
    label4 = "Exp 4: EfficientNet-B3"
except Exception:
    m4 = models.mobilenet_v2(pretrained=False)
    m4.classifier[1] = nn.Linear(m4.classifier[1].in_features, 1)
    label4 = "Exp 4: MobileNetV2 (fallback)"
p4, f4 = get_flops(m4, label4)

# Exp 5 — frozen DenseNet121 (same architecture, same FLOPs)
m5 = models.densenet121(pretrained=False)
for param in m5.parameters():
    param.requires_grad = False
m5.classifier = nn.Linear(m5.classifier.in_features, 1)
p5, f5 = get_flops(m5, "Exp 5: DenseNet121 (frozen)")

print()
print("=" * 70)
print("PASTE INTO report.tex")
print("=" * 70)
rows = [
    ("Exp 1: Baseline CNN",         p1, f1),
    ("Exp 2: ResNet50",             p2, f2),
    ("Exp 3: DenseNet121 (full)",   p3, f3),
    (label4,                        p4, f4),
    ("Exp 5: DenseNet121 (frozen)", p5, f5),
]
for name, p, f in rows:
    # Format params with LaTeX-friendly comma groups
    p_str = f"{int(p):,}".replace(",", "{,}")
    print(f"  {name:<35} & {p_str} & {f:.2f} \\\\")
