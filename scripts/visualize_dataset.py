"""
Dataset Visualization Script.
Renders ground truth bounding boxes and class names over sample images to verify annotations.

Usage:
    python scripts/visualize_dataset.py --split train --num 10 --save-dir outputs/visualizations
"""

import argparse
import sys
import cv2
from pathlib import Path
import random

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_utils import parse_data_yaml, check_single_annotation, VALID_IMAGE_EXTENSIONS
from utils.visualization import draw_ground_truth


def main():
    parser = argparse.ArgumentParser(description="Visualize ground truth annotations on dataset images.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--split", type=str, default="train", choices=["train", "val", "test"], help="Dataset split to visualize")
    parser.add_argument("--num", type=int, default=10, help="Number of random samples to visualize")
    parser.add_argument("--save-dir", type=str, default="outputs/visualizations", help="Directory to save visualized images")
    parser.add_argument("--show", action="store_true", help="Display images in an interactive OpenCV window")
    args = parser.parse_args()

    cfg = parse_data_yaml(args.data)
    root = cfg["resolved_root"]
    class_names = cfg["names_dict"]
    num_classes = cfg["num_classes"]

    split_rel = cfg.get(args.split)
    if not split_rel:
        print(f"[Error] Split '{args.split}' not defined in {args.data}")
        sys.exit(1)

    images_dir = (root / split_rel).resolve()
    if "images" in images_dir.parts:
        parts = list(images_dir.parts)
        idx = parts.index("images")
        parts[idx] = "labels"
        labels_dir = Path(*parts)
    else:
        labels_dir = images_dir.parent / "labels" / images_dir.name

    if not images_dir.exists():
        print(f"[Error] Images directory does not exist: {images_dir}")
        sys.exit(1)

    image_files = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in VALID_IMAGE_EXTENSIONS]
    if not image_files:
        print(f"[Warning] No images found in {images_dir}")
        sys.exit(0)

    save_dir = Path(args.save_dir) / args.split
    save_dir.mkdir(parents=True, exist_ok=True)

    samples = random.sample(image_files, min(args.num, len(image_files)))
    print(f"\n🖼️ Visualizing {len(samples)} samples from '{args.split}' split...")

    for i, img_path in enumerate(samples, start=1):
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"[Warning] Could not read image: {img_path.name}")
            continue

        label_path = labels_dir / f"{img_path.stem}.txt"
        boxes = []
        if label_path.exists():
            _, _, boxes = check_single_annotation(label_path, num_classes=num_classes)

        annotated = draw_ground_truth(image, boxes, class_names)

        # Add image index & filename watermark
        h, w = annotated.shape[:2]
        info_text = f"Sample {i}/{len(samples)}: {img_path.name} ({len(boxes)} boxes)"
        cv2.putText(annotated, info_text, (10, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        out_path = save_dir / f"vis_{img_path.stem}.jpg"
        cv2.imwrite(str(out_path), annotated)

        if args.show:
            cv2.imshow(f"Dataset Inspector - {args.split}", annotated)
            key = cv2.waitKey(0)
            if key == 27:  # ESC to quit
                break

    if args.show:
        cv2.destroyAllWindows()

    print(f"✅ Saved visual inspection samples to: {save_dir.resolve()}\n")


if __name__ == "__main__":
    main()
