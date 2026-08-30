import sys
import random
from pathlib import Path
from collections import Counter

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import torchvision.transforms as transforms
import cv2
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

from resnet.classifier import ResNet18Classifier, MASTER_CLASSES
from utils.roi_utils import expand_and_clamp_bbox

def load_all_rois(dataset_dir: str = "SIH_Dataset_27class", split: str = "train"):
    split_dir = Path(dataset_dir) / split
    img_dir = split_dir / "images"
    lbl_dir = split_dir / "labels"

    print(f"📦 Pre-caching ALL {split} images into RAM...", flush=True)

    img_tensors = []
    labels = []
    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
    norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    all_files = sorted(img_dir.iterdir())
    for img_path in all_files:
        if img_path.suffix.lower() not in valid_exts:
            continue
        lbl_path = lbl_dir / f"{img_path.stem}.txt"
        if not lbl_path.exists():
            continue

        lines = lbl_path.read_text(encoding="utf-8").strip().splitlines()
        if not lines:
            continue

        full_img = cv2.imread(str(img_path))
        if full_img is None:
            continue

        h_img, w_img = full_img.shape[:2]

        for line in lines:
            parts = line.strip().split()
            if len(parts) >= 5:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:5])
                x1 = (cx - bw / 2.0) * w_img
                y1 = (cy - bh / 2.0) * h_img
                x2 = (cx + bw / 2.0) * w_img
                y2 = (cy + bh / 2.0) * h_img

                rx1, ry1, rx2, ry2 = expand_and_clamp_bbox([x1, y1, x2, y2], full_img.shape, padding_ratio=0.20)
                crop = full_img[ry1:ry2, rx1:rx2]
                if crop.size == 0:
                    crop = full_img

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                crop_resized = cv2.resize(crop_rgb, (224, 224), interpolation=cv2.INTER_LINEAR)

                t = torch.from_numpy(crop_resized.transpose(2, 0, 1)).float() / 255.0
                t = norm(t)

                img_tensors.append(t)
                labels.append(cid)

    stacked_imgs = torch.stack(img_tensors)
    stacked_labels = torch.tensor(labels, dtype=torch.long)

    class_counts = Counter(labels)
    print(f"✅ Loaded {len(stacked_imgs)} {split} crops ({len(class_counts)}/{len(MASTER_CLASSES)} classes) -> Tensor shape: {stacked_imgs.shape}", flush=True)
    return stacked_imgs, stacked_labels


def train_resnet18(epochs: int = 20, batch_size: int = 64, lr: float = 3e-4):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("=" * 75, flush=True)
    print(f" 🚀 ULTRA-FAST RESNET-18 TRAINING (27 SIH CLASSES) on {device}", flush=True)
    print("=" * 75, flush=True)

    train_x, train_y = load_all_rois(split="train")
    val_x, val_y     = load_all_rois(split="val")

    train_ds = TensorDataset(train_x, train_y)
    val_ds   = TensorDataset(val_x, val_y)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    model = ResNet18Classifier(num_classes=len(MASTER_CLASSES), pretrained=True).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    weights_dir = Path("weights")
    weights_dir.mkdir(parents=True, exist_ok=True)
    best_weights_path = weights_dir / "resnet18_debris_best.pt"

    best_val_acc = 0.0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct_train += (preds == labels).sum().item()
            total_train += labels.size(0)

        scheduler.step()
        train_loss = running_loss / max(total_train, 1)
        train_acc = (correct_train / max(total_train, 1)) * 100.0

        # Validation
        model.eval()
        val_loss = 0.0
        correct_val = 0
        total_val = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                correct_val += (preds == labels).sum().item()
                total_val += labels.size(0)

        val_loss = val_loss / max(total_val, 1)
        val_acc = (correct_val / max(total_val, 1)) * 100.0

        print(f"Epoch [{epoch:02d}/{epochs:02d}] | Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%", flush=True)

        if val_acc >= best_val_acc or epoch == epochs:
            best_val_acc = val_acc
            torch.save(model.state_dict(), str(best_weights_path))
            print(f"  ⭐ Saved Best Weights -> {best_weights_path} (Val Acc: {val_acc:.2f}%)", flush=True)

    print("=" * 75, flush=True)
    print(f" ✅ ALL 27 CLASSES RESNET-18 TRAINING COMPLETE! Best Val Accuracy: {best_val_acc:.2f}%", flush=True)
    print(f" 📁 Saved to: {best_weights_path.resolve()}", flush=True)
    print("=" * 75, flush=True)

if __name__ == "__main__":
    train_resnet18(epochs=20, batch_size=64, lr=3e-4)
