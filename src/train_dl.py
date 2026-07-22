"""
train_dl.py
===========
Training pipeline for Deep Learning Models (Custom CNN & ResNet18 Transfer Learning).

Pipeline:
  1. Data Augmentation & Loaders (ImageFolder with 128x128 resolution)
     - Augmentations: RandomHorizontalFlip, RandomVerticalFlip, RandomRotation(180), ColorJitter
  2. Train CustomRetinalCNN (from scratch, 10 epochs)
  3. Train ResNet18 Transfer Learning (Pass 1: FC head 5 epochs, Pass 2: unfreeze layer4 5 epochs)
  4. Track Train/Val Accuracy & Loss curves
  5. Save trained models to models/custom_cnn.pt and models/resnet18_dr.pt
"""

import os
import sys
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.cnn_model import CustomRetinalCNN
from src.transfer_model import build_transfer_resnet18, unfreeze_layer4

warnings.filterwarnings("ignore")

RANDOM_SEED = 42
torch.manual_seed(RANDOM_SEED)

IMAGES_DIR = os.path.join(ROOT, "data", "images")
MODELS_DIR = os.path.join(ROOT, "models")
PROC_DIR = os.path.join(ROOT, "data", "processed")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(PROC_DIR, exist_ok=True)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BATCH_SIZE = 32
IMAGE_SIZE = (128, 128)


# ── Data Transforms ───────────────────────────────────────────────────────────

def get_data_loaders():
    # Retinal images are rotation and flip invariant
    train_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomRotation(degrees=180),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize(IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_dataset = datasets.ImageFolder(os.path.join(IMAGES_DIR, "train"), transform=train_transforms)
    val_dataset = datasets.ImageFolder(os.path.join(IMAGES_DIR, "val"), transform=val_transforms)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    return train_loader, val_loader, train_dataset.classes


# ── Training Epoch Function ───────────────────────────────────────────────────

def train_one_epoch(model, loader, criterion, optimizer):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, targets in loader:
        inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += inputs.size(0)

    return running_loss / total, correct / total


def validate(model, loader, criterion):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            correct += (preds == targets).sum().item()
            total += inputs.size(0)

    return running_loss / total, correct / total


# ── Main Training Loop ────────────────────────────────────────────────────────

def main():
    print("=" * 65)
    print("Healthcare Pathways AI — Deep Learning Training Pipeline")
    print("=" * 65)
    print(f"Device: {DEVICE}")

    train_loader, val_loader, classes = get_data_loaders()
    print(f"Dataset classes ({len(classes)}): {classes}")
    print(f"Train batches: {len(train_loader)} | Val batches: {len(val_loader)}")

    criterion = nn.CrossEntropyLoss()
    logs = {"custom_cnn": [], "resnet18": []}

    # ── 1. Train Custom CNN from Scratch ──────────────────────────────────────
    print("\n" + "=" * 50)
    print("1. Training Custom CNN From Scratch (10 Epochs)")
    print("=" * 50)

    cnn_model = CustomRetinalCNN(num_classes=len(classes)).to(DEVICE)
    cnn_optimizer = optim.Adam(cnn_model.parameters(), lr=1e-3, weight_decay=1e-4)

    for epoch in range(1, 11):
        t_loss, t_acc = train_one_epoch(cnn_model, train_loader, criterion, cnn_optimizer)
        v_loss, v_acc = validate(cnn_model, val_loader, criterion)
        print(f"  Epoch {epoch:2d}/10 | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        logs["custom_cnn"].append({"epoch": epoch, "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})

    torch.save(cnn_model.state_dict(), os.path.join(MODELS_DIR, "custom_cnn.pt"))
    print(f"  [OK] Saved: models/custom_cnn.pt")

    # ── 2. Train ResNet18 Transfer Learning ───────────────────────────────────
    print("\n" + "=" * 50)
    print("2. Training ResNet18 Transfer Learning (Pass 1: Head, Pass 2: Fine-tune)")
    print("=" * 50)

    resnet_model = build_transfer_resnet18(num_classes=len(classes), freeze_base=True).to(DEVICE)
    resnet_optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=1e-3)

    # Pass 1: Head only (5 epochs)
    print("  [Pass 1] Training FC Head (5 Epochs)...")
    for epoch in range(1, 6):
        t_loss, t_acc = train_one_epoch(resnet_model, train_loader, criterion, resnet_optimizer)
        v_loss, v_acc = validate(resnet_model, val_loader, criterion)
        print(f"    Epoch {epoch:2d}/5 | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        logs["resnet18"].append({"epoch": epoch, "phase": "head_only", "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})

    # Pass 2: Unfreeze layer4 fine-tuning (5 epochs)
    print("  [Pass 2] Unfreezing Layer4 Fine-Tuning (5 Epochs)...")
    unfreeze_layer4(resnet_model)
    ft_optimizer = optim.Adam(filter(lambda p: p.requires_grad, resnet_model.parameters()), lr=1e-4)

    for epoch in range(6, 11):
        t_loss, t_acc = train_one_epoch(resnet_model, train_loader, criterion, ft_optimizer)
        v_loss, v_acc = validate(resnet_model, val_loader, criterion)
        print(f"    Epoch {epoch:2d}/10 | Train Loss: {t_loss:.4f} Acc: {t_acc:.4f} | Val Loss: {v_loss:.4f} Acc: {v_acc:.4f}")
        logs["resnet18"].append({"epoch": epoch, "phase": "fine_tuning", "train_loss": t_loss, "train_acc": t_acc, "val_loss": v_loss, "val_acc": v_acc})

    torch.save(resnet_model.state_dict(), os.path.join(MODELS_DIR, "resnet18_dr.pt"))
    print(f"  [OK] Saved: models/resnet18_dr.pt")

    # Save training logs JSON
    with open(os.path.join(PROC_DIR, "dl_training_logs.json"), "w") as f:
        json.dump(logs, f, indent=2)
    print(f"\n[OK] Saved training logs: data/processed/dl_training_logs.json")

    print("\n" + "=" * 65)
    print("Training Complete! Run src/evaluate_dl.py for model comparison.")
    print("=" * 65)


if __name__ == "__main__":
    main()
