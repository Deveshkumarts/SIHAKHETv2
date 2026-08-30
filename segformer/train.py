"""
SegFormer Training Pipeline for Sonar ROI Segmentation.
Trains SegFormer-B0 model with BCE + Dice loss, evaluating mIoU, Dice Score,
Pixel Precision, and Pixel Recall across epochs.
"""

import sys
import argparse
import json
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segformer.dataset import SonarSegDataset
from segformer.model import SegFormerB0
from utils.device_utils import select_device


class BCEDiceLoss(nn.Module):
    """Combined Binary Cross Entropy + Dice Loss for robust segmentation."""
    def __init__(self, bce_weight=0.5, smooth=1e-6):
        super().__init__()
        self.bce_weight = bce_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.smooth = smooth

    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)

        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        cardinality = (probs + targets).sum(dim=(2, 3))

        dice = (2.0 * intersection + self.smooth) / (cardinality + self.smooth)
        dice_loss = 1.0 - dice.mean()

        return self.bce_weight * bce_loss + (1.0 - self.bce_weight) * dice_loss


def calculate_metrics(logits: torch.Tensor, targets: torch.Tensor, threshold: float = 0.5):
    """
    Calculate Pixel Precision, Pixel Recall, IoU, and Dice Coefficient.
    """
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    tp = (preds * targets).sum().item()
    fp = (preds * (1 - targets)).sum().item()
    fn = ((1 - preds) * targets).sum().item()

    precision = tp / (tp + fp + 1e-6)
    recall = tp / (tp + fn + 1e-6)
    iou = tp / (tp + fp + fn + 1e-6)
    dice = (2 * tp) / (2 * tp + fp + fn + 1e-6)

    return precision, recall, iou, dice


def main():
    parser = argparse.ArgumentParser(description="Train SegFormer for Sonar ROI Target Segmentation.")
    parser.add_argument("--epochs", type=int, default=25, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--img-size", type=int, default=224, help="ROI image size")
    parser.add_argument("--device", type=str, default="0", help="Device")
    parser.add_argument("--save-dir", type=str, default="outputs/segformer", help="Save directory")
    args = parser.parse_args()

    save_dir = Path(args.save_dir).resolve()
    weights_dir = save_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)

    device_str = select_device(args.device)
    selected_device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    print("\n" + "=" * 75)
    print(" 🚀 SEGFORMER ROI TARGET SEGMENTATION TRAINING")
    print("=" * 75)
    print(f"📦 Model: SegFormer-B0 (3.7M parameters)")
    print(f"🎯 Target Device: {selected_device}")
    print(f"⚙️ Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | ImgSize: {args.img_size}px\n")

    # Load datasets
    train_dataset = SonarSegDataset("dataset_seg/train/rois", "dataset_seg/train/masks", (args.img_size, args.img_size), is_train=True)
    val_dataset = SonarSegDataset("dataset_seg/val/rois", "dataset_seg/val/masks", (args.img_size, args.img_size), is_train=False)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2, pin_memory=True)

    model = SegFormerB0(in_channels=3, num_classes=1).to(selected_device)
    criterion = BCEDiceLoss(bce_weight=0.5)
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    best_iou = 0.0
    history = []

    for epoch in range(1, args.epochs + 1):
        # Training Phase
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in train_loader:
            imgs = imgs.to(selected_device)
            masks = masks.to(selected_device)

            optimizer.zero_grad()
            logits = model(imgs)
            loss = criterion(logits, masks)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * imgs.size(0)

        scheduler.step()
        train_loss /= len(train_dataset)

        # Validation Phase
        model.eval()
        val_loss = 0.0
        val_p, val_r, val_iou, val_dice = 0.0, 0.0, 0.0, 0.0
        num_val_batches = len(val_loader)

        with torch.no_grad():
            for imgs, masks, _ in val_loader:
                imgs = imgs.to(selected_device)
                masks = masks.to(selected_device)

                logits = model(imgs)
                loss = criterion(logits, masks)
                val_loss += loss.item() * imgs.size(0)

                bp, br, biou, bdice = calculate_metrics(logits, masks)
                val_p += bp
                val_r += br
                val_iou += biou
                val_dice += bdice

        val_loss /= len(val_dataset)
        val_p /= num_val_batches
        val_r /= num_val_batches
        val_iou /= num_val_batches
        val_dice /= num_val_batches

        print(
            f"Epoch [{epoch:02d}/{args.epochs:02d}] - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
            f"mIoU: {val_iou*100:5.2f}% | Dice: {val_dice*100:5.2f}% | P: {val_p*100:5.2f}% | R: {val_r*100:5.2f}%"
        )

        epoch_stats = {
            "epoch": epoch,
            "train_loss": round(train_loss, 4),
            "val_loss": round(val_loss, 4),
            "val_iou": round(val_iou, 4),
            "val_dice": round(val_dice, 4),
            "val_precision": round(val_p, 4),
            "val_recall": round(val_r, 4),
        }
        history.append(epoch_stats)

        # Checkpoint saving
        if val_iou > best_iou:
            best_iou = val_iou
            torch.save(model.state_dict(), weights_dir / "best.pt")
            print(f"   ⭐ New best SegFormer checkpoint saved (mIoU: {val_iou*100:.2f}%)")

        torch.save(model.state_dict(), weights_dir / "last.pt")

    # Save training history JSON
    with open(save_dir / "training_history.json", "w", encoding="utf-8") as f:
        json.dump(history, f, indent=2)

    print("\n" + "=" * 75)
    print(" 🎉 SEGFORMER TRAINING COMPLETE")
    print("=" * 75)
    print(f"⭐ Best Model Weights: {weights_dir / 'best.pt'}")
    print(f"📊 Best Validation mIoU: {best_iou*100:.2f}%")
    print("=" * 75 + "\n")


if __name__ == "__main__":
    main()
