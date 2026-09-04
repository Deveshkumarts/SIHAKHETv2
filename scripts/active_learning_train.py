"""
Active Learning Fine-Tuning Pipeline.
Loads human-verified and relabeled acoustic crop samples from the review archive,
fine-tunes the ResNet-18 classifier, and updates the model weights.
"""

import sys
import json
import time
from pathlib import Path
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from resnet.classifier import MASTER_CLASSES


class ActiveLearningDataset(Dataset):
    def __init__(self, samples, transform=None):
        self.samples = samples
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        item = self.samples[idx]
        crop_p = Path(item["crop_path"])
        if crop_p.exists():
            img = cv2.imread(str(crop_p))
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        else:
            img = np.zeros((224, 224, 3), dtype=np.uint8)

        cls_name = item.get("final_class", item.get("class_name", "bottle"))
        label_idx = MASTER_CLASSES.index(cls_name) if cls_name in MASTER_CLASSES else 0

        if self.transform:
            img = self.transform(img)

        return img, label_idx


def run_active_learning_retraining(
    archive_path: str = "data/active_learning/reviewed_samples.json",
    epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 1e-4
):
    print("🔁 Starting Active Learning Retraining Pipeline...")
    p = Path(archive_path)
    if not p.exists():
        print(f"⚠️ No reviewed samples file found at {p}. Please review samples in the Active Learning tab first.")
        return

    with open(p, "r", encoding="utf-8") as f:
        samples = json.load(f)

    # Filter only approved or relabeled samples
    valid_samples = [s for s in samples if s.get("action") in ("CONFIRM", "RELABEL") and Path(s.get("crop_path", "")).exists()]
    print(f"   Found {len(valid_samples)} validated training crops in archive.")

    if len(valid_samples) < 2:
        print("ℹ️ Insufficient reviewed samples for a full retraining pass (need >= 2).")
        return

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = ActiveLearningDataset(valid_samples, transform=transform)
    loader = DataLoader(dataset, batch_size=min(batch_size, len(dataset)), shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
    model.fc = nn.Linear(model.fc.in_features, len(MASTER_CLASSES))
    model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)

    model.train()
    for epoch in range(epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = (correct / total) * 100 if total > 0 else 0
        print(f"   Epoch [{epoch+1}/{epochs}] — Loss: {running_loss/len(loader):.4f} | Accuracy: {acc:.1f}%")

    out_weights = Path("weights/resnet18_active_learning.pth")
    out_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), str(out_weights))
    print(f"✅ Retraining Complete! Fine-tuned weights saved to: {out_weights}")


if __name__ == "__main__":
    run_active_learning_retraining()
