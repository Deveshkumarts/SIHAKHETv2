"""
Dataset validation and geometry verification utilities for YOLO object detection datasets.
Checks dataset integrity, bounding box geometry, label coordinates, and class mappings.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
import yaml
from PIL import Image
import hashlib


VALID_IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


def parse_data_yaml(yaml_path: str) -> Dict[str, Any]:
    """
    Parse YOLO data.yaml and resolve relative directory paths.
    """
    yaml_file = Path(yaml_path).resolve()
    if not yaml_file.exists():
        raise FileNotFoundError(f"data.yaml not found at path: {yaml_path}")

    with open(yaml_file, "r", encoding="utf-8") as f:
        data_cfg = yaml.safe_load(f)

    # Normalize root path
    base_dir = yaml_file.parent
    root_path = data_cfg.get("path", ".")
    if Path(root_path).is_absolute():
        resolved_root = Path(root_path)
    elif (base_dir / root_path).exists():
        resolved_root = (base_dir / root_path).resolve()
    elif (base_dir.parent / root_path).exists():
        resolved_root = (base_dir.parent / root_path).resolve()
    elif base_dir.exists():
        resolved_root = base_dir.resolve()
    else:
        resolved_root = (base_dir / root_path).resolve()

    data_cfg["resolved_root"] = resolved_root

    # Parse class names dictionary
    names = data_cfg.get("names", {})
    if isinstance(names, list):
        names = {i: name for i, name in enumerate(names)}
    elif isinstance(names, dict):
        names = {int(k): str(v) for k, v in names.items()}
    data_cfg["names_dict"] = names
    data_cfg["num_classes"] = data_cfg.get("nc", len(names))

    return data_cfg


def check_single_annotation(
    label_path: Path,
    num_classes: int,
    allow_empty: bool = True
) -> Tuple[bool, List[str], List[Tuple[int, float, float, float, float]]]:
    """
    Validate a single YOLO annotation .txt file.

    Format expected per line: <class_id> <cx> <cy> <w> <h>
    Coordinates must be normalized in [0, 1].

    Returns:
        (is_valid, error_messages, parsed_boxes)
    """
    errors = []
    boxes = []

    if not label_path.exists():
        return False, ["Label file does not exist."], boxes

    try:
        with open(label_path, "r", encoding="utf-8") as f:
            lines = [line.strip() for line in f if line.strip()]
    except Exception as e:
        return False, [f"Failed to read file: {e}"], boxes

    if len(lines) == 0:
        if not allow_empty:
            errors.append("Empty annotation file (contains 0 bounding boxes).")
        return len(errors) == 0, errors, boxes

    for idx, line in enumerate(lines, start=1):
        parts = line.split()
        if len(parts) != 5:
            errors.append(f"Line {idx}: Expected 5 values (class cx cy w h), found {len(parts)}.")
            continue

        try:
            cls_id = int(parts[0])
            cx, cy, w, h = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
        except ValueError:
            errors.append(f"Line {idx}: Non-numeric value in '{line}'.")
            continue

        # Class ID range check
        if cls_id < 0 or cls_id >= num_classes:
            errors.append(f"Line {idx}: Class ID {cls_id} out of range [0, {num_classes - 1}].")

        # Zero or negative box dimensions
        if w <= 0.0 or h <= 0.0:
            errors.append(f"Line {idx}: Non-positive dimension w={w}, h={h}.")

        # Coordinates normalized within [0, 1]
        if not (0.0 <= cx <= 1.0 and 0.0 <= cy <= 1.0):
            errors.append(f"Line {idx}: Center coordinates (cx={cx}, cy={cy}) outside [0, 1].")

        # Bounding box bounds check: xmin, xmax, ymin, ymax
        xmin = cx - w / 2.0
        xmax = cx + w / 2.0
        ymin = cy - h / 2.0
        ymax = cy + h / 2.0

        if xmin < -0.05 or xmax > 1.05 or ymin < -0.05 or ymax > 1.05:
            errors.append(f"Line {idx}: Box extends significantly outside image bounds [{xmin:.3f}, {ymin:.3f}, {xmax:.3f}, {ymax:.3f}].")

        boxes.append((cls_id, cx, cy, w, h))

    return len(errors) == 0, errors, boxes


def compute_file_hash(filepath: Path) -> str:
    """Compute MD5 hash to detect duplicate images."""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        while chunk := f.read(65536):
            hasher.update(chunk)
    return hasher.hexdigest()


def validate_yolo_dataset(data_yaml_path: str) -> Dict[str, Any]:
    """
    Perform a complete dataset health audit across all splits defined in data.yaml.

    Checks:
    - Missing images / missing labels
    - Invalid class IDs
    - Invalid or out-of-bounds coordinates
    - Negative or zero-area boxes
    - Corrupted image headers / unreadable files
    - Duplicate image files (hash check)
    - Background (empty label) images
    - Class distribution and object counts
    """
    cfg = parse_data_yaml(data_yaml_path)
    root = cfg["resolved_root"]
    num_classes = cfg["num_classes"]
    names_dict = cfg["names_dict"]

    summary = {
        "yaml_path": str(Path(data_yaml_path).resolve()),
        "root_dir": str(root),
        "num_classes": num_classes,
        "classes": names_dict,
        "splits": {},
        "total_images": 0,
        "total_labels": 0,
        "total_boxes": 0,
        "class_counts": {cid: 0 for cid in range(num_classes)},
        "invalid_annotations_count": 0,
        "corrupt_images_count": 0,
        "missing_labels_count": 0,
        "background_images_count": 0,
        "duplicate_images_count": 0,
        "errors": [],
        "warnings": [],
        "is_healthy": True,
    }

    seen_hashes = {}

    for split_key in ["train", "val", "test"]:
        split_rel = cfg.get(split_key)
        if not split_rel:
            continue

        images_dir = (root / split_rel).resolve()
        # In standard YOLO structure, if images are in images/train, labels are in labels/train
        if "images" in images_dir.parts:
            # Replace images with labels in path
            parts = list(images_dir.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            labels_dir = Path(*parts)
        else:
            labels_dir = images_dir.parent / "labels" / images_dir.name

        split_info = {
            "images_dir": str(images_dir),
            "labels_dir": str(labels_dir),
            "image_count": 0,
            "label_count": 0,
            "box_count": 0,
            "background_count": 0,
            "corrupt_count": 0,
            "missing_label_count": 0,
            "errors": [],
        }

        if not images_dir.exists():
            summary["warnings"].append(f"Split '{split_key}' image directory does not exist: {images_dir}")
            summary["splits"][split_key] = split_info
            continue

        # Gather all image files
        image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        split_info["image_count"] = len(image_files)
        summary["total_images"] += len(image_files)

        for img_path in image_files:
            # 1. Image integrity check
            try:
                with Image.open(img_path) as img:
                    img.verify()
                # Reopen to check dimensions
                with Image.open(img_path) as img:
                    w, h = img.size
                    if w <= 0 or h <= 0:
                        raise ValueError(f"Invalid dimensions {w}x{h}")
            except Exception as e:
                split_info["corrupt_count"] += 1
                summary["corrupt_images_count"] += 1
                split_info["errors"].append(f"Corrupt image {img_path.name}: {e}")
                continue

            # 2. Duplicate detection
            img_hash = compute_file_hash(img_path)
            if img_hash in seen_hashes:
                summary["duplicate_images_count"] += 1
                summary["warnings"].append(f"Duplicate image detected: {img_path.name} matches {seen_hashes[img_hash]}")
            else:
                seen_hashes[img_hash] = f"{split_key}/{img_path.name}"

            # 3. Label file check
            label_file = labels_dir / f"{img_path.stem}.txt"
            if not label_file.exists():
                split_info["missing_label_count"] += 1
                summary["missing_labels_count"] += 1
                split_info["errors"].append(f"Missing label file for image: {img_path.name}")
                continue

            split_info["label_count"] += 1
            summary["total_labels"] += 1

            is_valid, errs, boxes = check_single_annotation(label_file, num_classes=num_classes, allow_empty=True)
            if not is_valid:
                summary["invalid_annotations_count"] += 1
                split_info["errors"].extend([f"{label_file.name}: {err}" for err in errs])

            if len(boxes) == 0:
                split_info["background_count"] += 1
                summary["background_images_count"] += 1
            else:
                split_info["box_count"] += len(boxes)
                summary["total_boxes"] += len(boxes)
                for cid, cx, cy, bw, bh in boxes:
                    if cid in summary["class_counts"]:
                        summary["class_counts"][cid] += 1
                    
                    # Size distribution (area normalized to image)
                    box_area = bw * bh
                    aspect_ratio = bw / (bh + 1e-6)
                    
                    if box_area < 0.002: # < 0.2%
                        size_cat = "very_small"
                    elif box_area < 0.01: # 0.2% - 1.0%
                        size_cat = "small"
                    elif box_area < 0.05: # 1.0% - 5.0%
                        size_cat = "medium"
                    else:
                        size_cat = "large"
                        
                    if "size_distribution" not in summary:
                        summary["size_distribution"] = {"very_small": 0, "small": 0, "medium": 0, "large": 0}
                        summary["aspect_ratios"] = {"extreme_wide": 0, "wide": 0, "square_ish": 0, "tall": 0, "extreme_tall": 0}
                    
                    summary["size_distribution"][size_cat] += 1
                    
                    if aspect_ratio > 3.0:
                        summary["aspect_ratios"]["extreme_wide"] += 1
                    elif aspect_ratio > 1.5:
                        summary["aspect_ratios"]["wide"] += 1
                    elif aspect_ratio >= 0.67:
                        summary["aspect_ratios"]["square_ish"] += 1
                    elif aspect_ratio >= 0.33:
                        summary["aspect_ratios"]["tall"] += 1
                    else:
                        summary["aspect_ratios"]["extreme_tall"] += 1

        summary["splits"][split_key] = split_info

    if summary["corrupt_images_count"] > 0 or summary["invalid_annotations_count"] > 0 or summary["missing_labels_count"] > 0:
        summary["is_healthy"] = False

    return summary

