"""
Dataset Validation and Health Audit Script.
Checks dataset integrity, bounding box geometry, label coordinates, image corruption,
missing labels, background images, duplicate images, and class balance.

Usage:
    python scripts/check_dataset.py --data dataset/data.yaml
"""

import argparse
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows console
if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from utils.dataset_utils import validate_yolo_dataset


def main():
    parser = argparse.ArgumentParser(description="Audit and validate YOLO dataset health.")
    parser.add_argument("--data", type=str, default="dataset/data.yaml", help="Path to data.yaml")
    parser.add_argument("--strict", action="store_true", help="Fail if any background or warning items are found")
    args = parser.parse_args()

    print("\n" + "=" * 70)
    print(" 🌊 YOLO11 Sea Debris Dataset Health & Geometry Audit")
    print("=" * 70)

    try:
        report = validate_yolo_dataset(args.data)
    except Exception as e:
        print(f"\n[FATAL ERROR] Failed to parse or validate dataset: {e}")
        sys.exit(1)

    print(f"\n📂 Dataset Root: {report['root_dir']}")
    print(f"🏷️ Number of Classes: {report['num_classes']}")
    print(f"📋 Classes:")
    for cid, cname in report["classes"].items():
        count = report["class_counts"].get(cid, 0)
        print(f"   [{cid}] {cname:<20} : {count:>5} instances")

    print("\n" + "-" * 70)
    print(f"{'Split':<10} | {'Images':<8} | {'Labels':<8} | {'Boxes':<8} | {'Background (No Debris)':<22}")
    print("-" * 70)

    for split_name, sinfo in report["splits"].items():
        print(
            f"{split_name:<10} | {sinfo['image_count']:<8} | {sinfo['label_count']:<8} | "
            f"{sinfo['box_count']:<8} | {sinfo['background_count']:<22}"
        )
    print("-" * 70)
    print(
        f"{'TOTAL':<10} | {report['total_images']:<8} | {report['total_labels']:<8} | "
        f"{report['total_boxes']:<8} | {report['background_images_count']:<22}"
    )
    print("=" * 70)

    # Detailed Quality Breakdown
    print(f"\n🔍 Integrity & Quality Metrics:")
    print(f"   • Total Images:               {report['total_images']}")
    print(f"   • Total Labels:               {report['total_labels']}")
    print(f"   • Total Bounding Boxes:       {report['total_boxes']}")
    print(f"   • Background Images (Empty):  {report['background_images_count']}")
    print(f"   • Corrupt Image Files:        {report['corrupt_images_count']}")
    print(f"   • Missing Label Files:        {report['missing_labels_count']}")
    print(f"   • Invalid Annotations:        {report['invalid_annotations_count']}")
    print(f"   • Duplicate Image Files:      {report['duplicate_images_count']}")

    if report["errors"]:
        print("\n❌ Dataset Errors Found:")
        for err in report["errors"][:20]:
            print(f"   - {err}")
        if len(report["errors"]) > 20:
            print(f"   ... and {len(report['errors']) - 20} more errors.")

    if "size_distribution" in report:
        print("\n📐 Object Size Distribution (Relative Area):")
        sdist = report["size_distribution"]
        total_b = max(report["total_boxes"], 1)
        print(f"   • Very Small (< 0.2% area) : {sdist['very_small']:>5} ({sdist['very_small']/total_b*100:5.1f}%)")
        print(f"   • Small (0.2% - 1.0% area) : {sdist['small']:>5} ({sdist['small']/total_b*100:5.1f}%)")
        print(f"   • Medium (1.0% - 5.0% area): {sdist['medium']:>5} ({sdist['medium']/total_b*100:5.1f}%)")
        print(f"   • Large (> 5.0% area)      : {sdist['large']:>5} ({sdist['large']/total_b*100:5.1f}%)")

    if "aspect_ratios" in report:
        print("\n📏 Bounding Box Aspect Ratios (Width / Height):")
        ar = report["aspect_ratios"]
        total_b = max(report["total_boxes"], 1)
        print(f"   • Extreme Wide (> 3.0)     : {ar['extreme_wide']:>5} ({ar['extreme_wide']/total_b*100:5.1f}%)")
        print(f"   • Wide (1.5 - 3.0)         : {ar['wide']:>5} ({ar['wide']/total_b*100:5.1f}%)")
        print(f"   • Square-ish (0.67 - 1.5)  : {ar['square_ish']:>5} ({ar['square_ish']/total_b*100:5.1f}%)")
        print(f"   • Tall (0.33 - 0.67)       : {ar['tall']:>5} ({ar['tall']/total_b*100:5.1f}%)")
        print(f"   • Extreme Tall (< 0.33)    : {ar['extreme_tall']:>5} ({ar['extreme_tall']/total_b*100:5.1f}%)")

    if report["warnings"]:
        print("\n⚠️ Dataset Warnings:")
        for warn in report["warnings"][:10]:
            print(f"   - {warn}")
        if len(report["warnings"]) > 10:
            print(f"   ... and {len(report['warnings']) - 10} more warnings.")


    if report["is_healthy"] and (not args.strict or len(report["warnings"]) == 0):
        print("\n✅ DATASET AUDIT PASSED: The dataset is ready for training.\n")
        sys.exit(0)
    else:
        print("\n❌ DATASET AUDIT FAILED: Please fix the dataset errors above before training.\n")
        sys.exit(1)


if __name__ == "__main__":
    main()
