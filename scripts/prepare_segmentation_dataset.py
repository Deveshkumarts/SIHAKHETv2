"""
Segmentation Dataset Generator for Sonar ROI Crops.
Extracts expanded ROIs from YOLO dataset bounding boxes, generates acoustic pseudo-masks,
and saves a structured dataset for SegFormer segmentation training.
"""

import sys
import json
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Any

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_utils import parse_data_yaml, check_single_annotation, VALID_IMAGE_EXTENSIONS
from utils.roi_utils import expand_and_clamp_bbox
from utils.mask_utils import generate_acoustic_pseudo_mask


def prepare_segmentation_dataset(
    data_yaml_path: str = "Combined_Dataset/data.yaml",
    output_dir: str = "dataset_seg",
    padding_ratio: float = 0.25,
    roi_size: tuple = (224, 224)
) -> Dict[str, Any]:
    cfg = parse_data_yaml(data_yaml_path)
    root = cfg["resolved_root"]
    num_classes = cfg["num_classes"]
    class_names = cfg["names_dict"]

    out_root = Path(output_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Human review directory
    human_review_dir = out_root / "human_review"
    human_review_dir.mkdir(parents=True, exist_ok=True)

    stats = {
        "dataset_name": "Sonar_SegFormer_ROI_Dataset",
        "annotation_type": "pseudo_mask_acoustic_shadow",
        "padding_ratio": padding_ratio,
        "roi_size": list(roi_size),
        "total_rois": 0,
        "split_counts": {},
        "class_counts": {cname: 0 for cname in class_names.values()}
    }

    for split_key in ["train", "val", "test"]:
        split_rel = cfg.get(split_key)
        if not split_rel:
            continue

        images_dir = (root / split_rel).resolve()
        if "images" in images_dir.parts:
            parts = list(images_dir.parts)
            idx = parts.index("images")
            parts[idx] = "labels"
            labels_dir = Path(*parts)
        else:
            labels_dir = images_dir.parent / "labels" / images_dir.name

        out_rois_dir = out_root / split_key / "rois"
        out_masks_dir = out_root / split_key / "masks"
        out_rois_dir.mkdir(parents=True, exist_ok=True)
        out_masks_dir.mkdir(parents=True, exist_ok=True)

        split_roi_count = 0

        image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
        for img_path in image_files:
            orig_img = cv2.imread(str(img_path))
            if orig_img is None:
                continue
            h, w = orig_img.shape[:2]

            label_file = labels_dir / f"{img_path.stem}.txt"
            if not label_file.exists():
                continue

            _, _, parsed_boxes = check_single_annotation(label_file, num_classes=num_classes)
            for idx, (cid, cx, cy, bw, bh) in enumerate(parsed_boxes):
                cname = class_names.get(cid, str(cid))

                x1 = (cx - bw / 2.0) * w
                y1 = (cy - bh / 2.0) * h
                x2 = (cx + bw / 2.0) * w
                y2 = (cy + bh / 2.0) * h

                rx1, ry1, rx2, ry2 = expand_and_clamp_bbox([x1, y1, x2, y2], orig_img.shape, padding_ratio)

                roi_crop = orig_img[ry1:ry2, rx1:rx2].copy()
                if roi_crop.size == 0:
                    continue

                # Generate acoustic pseudo-mask
                pseudo_mask = generate_acoustic_pseudo_mask(roi_crop)

                # Resize image and mask to standard SegFormer input resolution (224x224)
                # IMPORTANT: Image -> INTER_LINEAR, Mask -> INTER_NEAREST
                roi_resized = cv2.resize(roi_crop, roi_size, interpolation=cv2.INTER_LINEAR)
                mask_resized = cv2.resize(pseudo_mask, roi_size, interpolation=cv2.INTER_NEAREST)

                sample_id = f"{img_path.stem}_roi{idx:02d}_{cname}"
                cv2.imwrite(str(out_rois_dir / f"{sample_id}.png"), roi_resized)
                cv2.imwrite(str(out_masks_dir / f"{sample_id}.png"), mask_resized)

                split_roi_count += 1
                stats["total_rois"] += 1
                stats["class_counts"][cname] = stats["class_counts"].get(cname, 0) + 1

                # Save sample visual overlay to human_review directory for initial 10 samples
                if stats["total_rois"] <= 10:
                    overlay = roi_resized.copy()
                    overlay[mask_resized > 0] = (
                        0.5 * overlay[mask_resized > 0] + 0.5 * np.array([0, 255, 0])
                    ).astype(np.uint8)
                    cv2.imwrite(str(human_review_dir / f"review_{sample_id}.png"), overlay)

        stats["split_counts"][split_key] = split_roi_count

    meta_path = out_root / "metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)

    print("\n" + "=" * 75)
    print(" 🛠️ SEGMENTATION DATASET GENERATION COMPLETE")
    print("=" * 75)
    print(f"📁 Output Directory: {out_root}")
    print(f"📦 Total ROI Samples: {stats['total_rois']}")
    for split_key, count in stats["split_counts"].items():
        print(f"   • Split '{split_key:<5}': {count:>4} ROIs")
    print("🏷️ Per-Class ROIs:")
    for cname, count in stats["class_counts"].items():
        print(f"   • {cname:<12}: {count:>4} ROIs")
    print(f"📋 Human Review Samples: {human_review_dir}")
    print("=" * 75 + "\n")

    return stats


def main():
    prepare_segmentation_dataset(
        data_yaml_path="Combined_Dataset/data.yaml",
        output_dir="dataset_seg",
        padding_ratio=0.25,
        roi_size=(224, 224)
    )


if __name__ == "__main__":
    main()
