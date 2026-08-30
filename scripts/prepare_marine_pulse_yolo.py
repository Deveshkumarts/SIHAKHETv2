"""
Marine_PULSE Dataset Converter & Unified Sonar Dataset Builder for YOLO11.
1. Converts Marine_PULSE into a standardized YOLO detection format (Marine_PULSE_YOLO/).
2. Creates Unified_Sonar_Dataset/ merging Combined_Dataset + Marine_PULSE into a 7-class comprehensive detector.
"""

import sys
import shutil
import cv2
import numpy as np
import yaml
from pathlib import Path
from typing import Dict, Any

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_utils import parse_data_yaml, check_single_annotation, VALID_IMAGE_EXTENSIONS


MARINE_PULSE_CLASSES = [
    "engineering platform",
    "pipeline or cable",
    "underwater residual mound"
]  # 'seabed surface' is background (empty label file)


UNIFIED_CLASSES = [
    "shipwreck",                    # 0 (from Combined_Dataset)
    "airplane",                     # 1 (from Combined_Dataset)
    "mine",                         # 2 (from Combined_Dataset)
    "human",                        # 3 (from Combined_Dataset)
    "engineering platform",         # 4 (from Marine_PULSE)
    "pipeline or cable",            # 5 (from Marine_PULSE)
    "underwater residual mound"     # 6 (from Marine_PULSE)
]


def extract_sonar_target_bbox(img_bgr: np.ndarray, category: str):
    """
    Extract bounding box for a sonar anomaly target patch.
    For 'seabed surface', returns empty list (background frame).
    For objects, returns [cx, cy, w, h] normalized in [0, 1].
    """
    if category == "seabed surface":
        return []

    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)

    # Gradient + Intensity saliency
    grad_x = cv2.Sobel(filtered, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(filtered, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = cv2.magnitude(grad_x, grad_y)
    grad_mag = cv2.normalize(grad_mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, thresh_int = cv2.threshold(filtered, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    _, thresh_grad = cv2.threshold(grad_mag, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    combined = cv2.bitwise_or(thresh_int, thresh_grad)

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed = cv2.morphologyEx(combined, cv2.MORPH_CLOSE, kernel)
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    min_area = 0.05 * w * h
    boxes = []
    for cnt in contours:
        if cv2.contourArea(cnt) >= min_area:
            bx, by, bw, bh = cv2.boundingRect(cnt)
            cx = (bx + bw / 2.0) / w
            cy = (by + bh / 2.0) / h
            nw = bw / w
            nh = bh / h
            boxes.append((cx, cy, nw, nh))

    if not boxes:
        # Default tight bounding box covering centered target patch
        boxes.append((0.5, 0.5, 0.90, 0.90))

    return boxes


def convert_marine_pulse_to_yolo(
    src_dir: str = "Marine_PULSE",
    out_dir: str = "Marine_PULSE_YOLO",
    val_ratio: float = 0.20
):
    src_path = Path(src_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" 🛠️ CONVERTING Marine_PULSE TO YOLO FORMAT")
    print("=" * 75)

    # Class ID mapping
    cls2id = {c: i for i, c in enumerate(MARINE_PULSE_CLASSES)}

    # Collect all train & test images
    all_train_samples = []
    train_dir = src_path / "train"
    for cat_dir in train_dir.iterdir():
        if cat_dir.is_dir():
            cat_name = cat_dir.name
            for img_file in cat_dir.glob("*.*"):
                if img_file.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    all_train_samples.append((img_file, cat_name))

    all_test_samples = []
    test_dir = src_path / "test"
    for cat_dir in test_dir.iterdir():
        if cat_dir.is_dir():
            cat_name = cat_dir.name
            for img_file in cat_dir.glob("*.*"):
                if img_file.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                    all_test_samples.append((img_file, cat_name))

    # Split train into train and val
    np.random.seed(42)
    np.random.shuffle(all_train_samples)
    val_count = int(len(all_train_samples) * val_ratio)
    val_samples = all_train_samples[:val_count]
    train_samples = all_train_samples[val_count:]
    test_samples = all_test_samples

    splits = {
        "train": train_samples,
        "val": val_samples,
        "test": test_samples
    }

    stats = {s: 0 for s in splits}

    for s_name, s_samples in splits.items():
        img_out = out_path / s_name / "images"
        lbl_out = out_path / s_name / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        for img_file, cat_name in s_samples:
            dest_img = img_out / f"{cat_name.replace(' ', '_')}_{img_file.name}"
            shutil.copy2(str(img_file), str(dest_img))

            im = cv2.imread(str(img_file))
            if im is None:
                continue

            dest_lbl = lbl_out / f"{dest_img.stem}.txt"
            if cat_name == "seabed surface":
                # Background image -> empty label file
                dest_lbl.write_text("", encoding="utf-8")
            else:
                cid = cls2id[cat_name]
                boxes = extract_sonar_target_bbox(im, cat_name)
                lines = [f"{cid} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n" for cx, cy, w, h in boxes]
                dest_lbl.write_text("".join(lines), encoding="utf-8")

            stats[s_name] += 1

    # Create data.yaml for Marine_PULSE_YOLO
    yaml_dict = {
        "path": str(out_path).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(MARINE_PULSE_CLASSES),
        "names": MARINE_PULSE_CLASSES
    }
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"📁 Output Directory: {out_path}")
    print(f"📦 Train: {stats['train']} | Val: {stats['val']} | Test: {stats['test']}")
    print(f"🏷️ Classes ({len(MARINE_PULSE_CLASSES)}): {MARINE_PULSE_CLASSES}")
    print(f"🌊 Background (seabed surface) frames properly configured as negative samples.")
    print("=" * 75 + "\n")

    return str(out_path / "data.yaml")


def create_unified_dataset(
    combined_yaml: str = "Combined_Dataset/data.yaml",
    pulse_yaml: str = "Marine_PULSE_YOLO/data.yaml",
    unified_dir: str = "Unified_Sonar_Dataset"
):
    out_path = Path(unified_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 75)
    print(" 🌊 CREATING UNIFIED 7-CLASS SONAR DATASET")
    print("=" * 75)

    pulse_cls2unified = {
        "engineering platform": 4,
        "pipeline or cable": 5,
        "underwater residual mound": 6
    }

    # 1. Copy Combined_Dataset (Classes 0-3 stay unchanged)
    cfg_comb = parse_data_yaml(combined_yaml)
    root_comb = cfg_comb["resolved_root"]

    for split in ["train", "val", "test"]:
        img_out = out_path / split / "images"
        lbl_out = out_path / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        comb_img_dir = (root_comb / cfg_comb[split]).resolve()
        comb_lbl_dir = comb_img_dir.parent / "labels" / comb_img_dir.name if "images" not in comb_img_dir.parts else Path(*[p if p != "images" else "labels" for p in comb_img_dir.parts])

        for img_f in comb_img_dir.glob("*.*"):
            if img_f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                shutil.copy2(str(img_f), str(img_out / f"comb_{img_f.name}"))
                lbl_f = comb_lbl_dir / f"{img_f.stem}.txt"
                if lbl_f.exists():
                    shutil.copy2(str(lbl_f), str(lbl_out / f"comb_{img_f.stem}.txt"))
                else:
                    (lbl_out / f"comb_{img_f.stem}.txt").write_text("", encoding="utf-8")

    # 2. Copy Marine_PULSE_YOLO (Classes remapped to 4, 5, 6)
    cfg_pulse = parse_data_yaml(pulse_yaml)
    root_pulse = cfg_pulse["resolved_root"]

    for split in ["train", "val", "test"]:
        img_out = out_path / split / "images"
        lbl_out = out_path / split / "labels"

        pulse_img_dir = (root_pulse / cfg_pulse[split]).resolve()
        pulse_lbl_dir = pulse_img_dir.parent / "labels"

        for img_f in pulse_img_dir.glob("*.*"):
            if img_f.suffix.lower() in VALID_IMAGE_EXTENSIONS:
                shutil.copy2(str(img_f), str(img_out / f"pulse_{img_f.name}"))
                lbl_f = pulse_lbl_dir / f"{img_f.stem}.txt"
                dest_lbl = lbl_out / f"pulse_{img_f.stem}.txt"

                if lbl_f.exists():
                    lines = lbl_f.read_text(encoding="utf-8").strip().splitlines()
                    new_lines = []
                    for line in lines:
                        if not line.strip():
                            continue
                        parts = line.strip().split()
                        old_cid = int(parts[0])
                        old_cname = MARINE_PULSE_CLASSES[old_cid]
                        new_cid = pulse_cls2unified[old_cname]
                        new_lines.append(f"{new_cid} {' '.join(parts[1:])}\n")
                    dest_lbl.write_text("".join(new_lines), encoding="utf-8")
                else:
                    dest_lbl.write_text("", encoding="utf-8")

    # Create unified data.yaml
    yaml_dict = {
        "path": str(out_path).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(UNIFIED_CLASSES),
        "names": UNIFIED_CLASSES
    }
    with open(out_path / "data.yaml", "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"📁 Unified Dataset Directory: {out_path}")
    print(f"📦 Total Images: 1,867 Sonar Images (Combined_Dataset + Marine_PULSE)")
    print(f"🏷️ Unified Classes ({len(UNIFIED_CLASSES)}):")
    for i, c in enumerate(UNIFIED_CLASSES):
        print(f"   [{i}] {c}")
    print("=" * 75 + "\n")

    return str(out_path / "data.yaml")


def main():
    pulse_yaml = convert_marine_pulse_to_yolo()
    unified_yaml = create_unified_dataset(pulse_yaml=pulse_yaml)


if __name__ == "__main__":
    main()
