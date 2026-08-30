"""
Turntable Marine Litter Dataset Converter for YOLO11.
Converts 4,942 cropped marine debris images into a standardized YOLO detection dataset.
Groups 17 fine-grained object categories into 5 standard Marine Litter material classes:
  [0] plastic
  [1] glass
  [2] metal
  [3] tire_rubber
  [4] carton
(with 'rotating-platform' as negative background frames)
"""

import sys
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Category mapping
CATEGORY_MAPPING = {
    # Plastic
    "plastic-bottle": "plastic",
    "plastic-bidon": "plastic",
    "plastic-pipe": "plastic",
    "plastic-propeller": "plastic",
    "drink-sachet": "plastic",
    # Glass
    "glass-bottle": "glass",
    "brown-glass-bottle": "glass",
    "potion-glass-bottle": "glass",
    "glass-jar": "glass",
    # Metal
    "can": "metal",
    "metal-bottle": "metal",
    "metal-box": "metal",
    "valve": "metal",
    "wrench": "metal",
    # Rubber
    "large-tire": "tire_rubber",
    "small-tire": "tire_rubber",
    # Carton
    "drink-carton": "carton",
    # Background
    "rotating-platform": "background"
}

LITTER_CLASSES = ["plastic", "glass", "metal", "tire_rubber", "carton"]


def convert_turntable_to_yolo(
    src_dir: str = "turntable-cropped",
    out_dir: str = "Debris_Litter_YOLO",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
):
    src_path = Path(src_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cls2id = {c: i for i, c in enumerate(LITTER_CLASSES)}

    print("\n" + "=" * 75)
    print(" 🛠️ CONVERTING TURNTABLE MARINE LITTER DATASET TO YOLO FORMAT")
    print("=" * 75)

    all_samples = []
    for cat_dir in sorted(src_path.iterdir()):
        if cat_dir.is_dir():
            folder_name = cat_dir.name
            super_class = CATEGORY_MAPPING.get(folder_name, "plastic")
            for img_file in cat_dir.glob("*.*"):
                if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                    all_samples.append((img_file, folder_name, super_class))

    print(f"📦 Total Images Found: {len(all_samples)}")

    # Shuffle deterministically
    np.random.seed(42)
    np.random.shuffle(all_samples)

    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": all_samples[:n_train],
        "val": all_samples[n_train:n_train + n_val],
        "test": all_samples[n_train + n_val:]
    }

    stats = {s: 0 for s in splits}
    class_stats = {c: 0 for c in LITTER_CLASSES}

    for s_name, s_samples in splits.items():
        img_out = out_path / s_name / "images"
        lbl_out = out_path / s_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_file, orig_cat, super_class in s_samples:
            dest_img = img_out / f"{orig_cat}_{img_file.name}"
            shutil.copy2(str(img_file), str(dest_img))

            dest_lbl = lbl_out / f"{dest_img.stem}.txt"

            if super_class == "background":
                # Background frame: empty label
                dest_lbl.write_text("", encoding="utf-8")
            else:
                cid = cls2id[super_class]
                # Normalized bounding box for centered debris object
                cx, cy, w, h = 0.50, 0.50, 0.90, 0.90
                dest_lbl.write_text(f"{cid} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n", encoding="utf-8")
                class_stats[super_class] += 1

            stats[s_name] += 1

    # Create data.yaml
    yaml_dict = {
        "path": str(out_path).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(LITTER_CLASSES),
        "names": LITTER_CLASSES
    }
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"📁 Output Directory: {out_path}")
    print(f"📦 Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print("🏷️ Material Classes:")
    for c, cnt in class_stats.items():
        print(f"   • {c:<12}: {cnt:>5} instances")
    print("=" * 75 + "\n")

    return str(out_path / "data.yaml")


if __name__ == "__main__":
    convert_turntable_to_yolo()
