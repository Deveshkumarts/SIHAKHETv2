"""
SegFormer Evaluation Script for Sonar ROI Segmentation.
Evaluates best.pt checkpoint on test set, reports mIoU, Dice, Pixel Precision/Recall,
and saves predicted mask visualizations.
"""

import sys
import json
from pathlib import Path
import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from segformer.dataset import SonarSegDataset
from segformer.model import SegFormerB0
from segformer.train import calculate_metrics
from utils.device_utils import select_device


def run_evaluation(
    weights_path: str = "outputs/segformer/weights/best.pt",
    dataset_dir: str = "dataset_seg",
    split: str = "test",
    img_size: int = 224,
    device_str: str = "0",
    save_dir: str = "outputs/segformer/evaluation"
) -> dict:
    save_path = Path(save_dir).resolve()
    preds_dir = save_path / "predictions"
    overlays_dir = save_path / "overlays"
    preds_dir.mkdir(parents=True, exist_ok=True)
    overlays_dir.mkdir(parents=True, exist_ok=True)

    device_str = select_device(device_str)
    selected_device = torch.device(f"cuda:{device_str}" if device_str.isdigit() else device_str)

    # Load model
    model = SegFormerB0(in_channels=3, num_classes=1)
    state_dict = torch.load(weights_path, map_location=selected_device)
    model.load_state_dict(state_dict)
    model.to(selected_device)
    model.eval()

    dataset = SonarSegDataset(
        f"{dataset_dir}/{split}/rois",
        f"{dataset_dir}/{split}/masks",
        (img_size, img_size),
        is_train=False
    )
    loader = DataLoader(dataset, batch_size=8, shuffle=False, num_workers=2)

    print("\n" + "=" * 75)
    print(f" 📊 SEGFORMER EVALUATION — SPLIT: {split.upper()}")
    print("=" * 75)
    print(f"📦 Model: {weights_path}")
    print(f"📂 Dataset: {dataset_dir}/{split} ({len(dataset)} samples)")

    total_p, total_r, total_iou, total_dice = 0.0, 0.0, 0.0, 0.0
    num_batches = len(loader)

    with torch.no_grad():
        for batch_idx, (imgs, masks, names) in enumerate(loader):
            imgs = imgs.to(selected_device)
            masks = masks.to(selected_device)
            logits = model(imgs)

            bp, br, biou, bdice = calculate_metrics(logits, masks)
            total_p += bp
            total_r += br
            total_iou += biou
            total_dice += bdice

            # Save predicted masks and overlays
            probs = torch.sigmoid(logits).cpu().numpy()
            imgs_np = (imgs.cpu().numpy().transpose(0, 2, 3, 1) * 255).astype(np.uint8)

            for i, name in enumerate(names):
                pred_binary = (probs[i, 0] >= 0.5).astype(np.uint8) * 255
                cv2.imwrite(str(preds_dir / name), pred_binary)

                # Overlay: green mask on original ROI image
                roi_bgr = cv2.cvtColor(imgs_np[i], cv2.COLOR_RGB2BGR)
                overlay = roi_bgr.copy()
                overlay[pred_binary > 0] = (
                    0.5 * overlay[pred_binary > 0] + 0.5 * np.array([0, 255, 0])
                ).astype(np.uint8)
                cv2.imwrite(str(overlays_dir / name), overlay)

    avg_p = total_p / num_batches
    avg_r = total_r / num_batches
    avg_iou = total_iou / num_batches
    avg_dice = total_dice / num_batches

    results = {
        "split": split,
        "num_samples": len(dataset),
        "pixel_precision": round(avg_p, 4),
        "pixel_recall": round(avg_r, 4),
        "mean_iou": round(avg_iou, 4),
        "dice_coefficient": round(avg_dice, 4),
    }

    with open(save_path / "evaluation_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("\n" + "=" * 75)
    print(" 🏆 SEGFORMER TEST-SET RESULTS")
    print("=" * 75)
    print(f"   • Pixel Precision:    {avg_p*100:6.2f}%")
    print(f"   • Pixel Recall:       {avg_r*100:6.2f}%")
    print(f"   • Mean IoU (mIoU):    {avg_iou*100:6.2f}%")
    print(f"   • Dice Coefficient:   {avg_dice*100:6.2f}%")
    print("=" * 75)
    print(f"✅ Evaluation summary saved: {save_path / 'evaluation_summary.json'}")
    print(f"🖼️  Predicted masks:          {preds_dir}")
    print(f"🖼️  Overlay visualizations:   {overlays_dir}")
    print("=" * 75 + "\n")

    return results


if __name__ == "__main__":
    run_evaluation()
