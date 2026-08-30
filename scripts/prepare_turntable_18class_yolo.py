"""
Turntable Marine Litter Dataset Converter — 18 Fine-Grained Classes for YOLO11.
Converts 4,942 cropped marine debris images into a YOLO detection dataset
keeping all 17 object categories as individual classes.
'rotating-platform' is treated as background (empty label).

Classes (17):
  [0]  brown-glass-bottle
  [1]  can
  [2]  drink-carton
  [3]  drink-sachet
  [4]  glass-bottle
  [5]  glass-jar
  [6]  large-tire
  [7]  metal-bottle
  [8]  metal-box
  [9]  plastic-bidon
  [10] plastic-bottle
  [11] plastic-pipe
  [12] plastic-propeller
  [13] potion-glass-bottle
  [14] small-tire
  [15] valve
  [16] wrench
"""

import sys
import shutil
import numpy as np
import yaml
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# All 17 object classes (rotating-platform = background, excluded)
CLASSES = [
    "brown-glass-bottle",
    "can",
    "drink-carton",
    "drink-sachet",
    "glass-bottle",
    "glass-jar",
    "large-tire",
    "metal-bottle",
    "metal-box",
    "plastic-bidon",
    "plastic-bottle",
    "plastic-pipe",
    "plastic-propeller",
    "potion-glass-bottle",
    "small-tire",
    "valve",
    "wrench",
]

BACKGROUND_FOLDER = "rotating-platform"


def convert_turntable_18class(
    src_dir: str = "turntable-cropped",
    out_dir: str = "Turntable_18class_YOLO",
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
):
    src_path = Path(src_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cls2id = {c: i for i, c in enumerate(CLASSES)}

    print("\n" + "=" * 75)
    print("  CONVERTING TURNTABLE DATASET → YOLO FORMAT (17 Fine-Grained Classes)")
    print("=" * 75)

    all_samples = []
    for cat_dir in sorted(src_path.iterdir()):
        if not cat_dir.is_dir():
            continue
        folder_name = cat_dir.name
        for img_file in cat_dir.glob("*.*"):
            if img_file.suffix.lower() in [".png", ".jpg", ".jpeg", ".bmp"]:
                all_samples.append((img_file, folder_name))

    print(f"  Total images found : {len(all_samples)}")
    print(f"  Classes            : {len(CLASSES)}")
    print(f"  Background (skip)  : {BACKGROUND_FOLDER}")
    print()

    # Deterministic shuffle
    np.random.seed(42)
    indices = np.arange(len(all_samples))
    np.random.shuffle(indices)
    all_samples = [all_samples[i] for i in indices]

    n_total = len(all_samples)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)

    splits = {
        "train": all_samples[:n_train],
        "val":   all_samples[n_train: n_train + n_val],
        "test":  all_samples[n_train + n_val:],
    }

    stats = {s: 0 for s in splits}
    class_stats = {c: 0 for c in CLASSES}
    bg_count = 0

    for split_name, samples in splits.items():
        img_out = out_path / split_name / "images"
        lbl_out = out_path / split_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_file, folder_name in samples:
            dest_img = img_out / f"{folder_name}_{img_file.name}"
            shutil.copy2(str(img_file), str(dest_img))
            dest_lbl = lbl_out / f"{dest_img.stem}.txt"

            if folder_name == BACKGROUND_FOLDER:
                # Background: empty label file
                dest_lbl.write_text("", encoding="utf-8")
                bg_count += 1
            else:
                cid = cls2id[folder_name]
                # Full-frame centered bbox (object occupies most of the cropped image)
                cx, cy, w, h = 0.50, 0.50, 0.90, 0.90
                dest_lbl.write_text(
                    f"{cid} {cx:.4f} {cy:.4f} {w:.4f} {h:.4f}\n",
                    encoding="utf-8",
                )
                class_stats[folder_name] += 1

            stats[split_name] += 1

    # Write data.yaml
    yaml_dict = {
        "path":  str(out_path).replace("\\", "/"),
        "train": "train/images",
        "val":   "val/images",
        "test":  "test/images",
        "nc":    len(CLASSES),
        "names": CLASSES,
    }
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"  Output : {out_path}")
    print(f"  Train  : {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print(f"  Background (skipped as objects): {bg_count}")
    print()
    print("  Per-class image counts:")
    for cls_name, cnt in class_stats.items():
        print(f"    [{cls2id[cls_name]:02d}] {cls_name:<25}: {cnt:>4} images")
    print("=" * 75)
    print(f"  data.yaml written to: {out_path / 'data.yaml'}")
    print("=" * 75 + "\n")

    return str(out_path / "data.yaml")


if __name__ == "__main__":
    convert_turntable_18class()
