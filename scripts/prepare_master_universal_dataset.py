"""
Master Universal Marine & Sonar 24-Class Dataset Builder.
Combines:
  1. Unified Sonar Dataset (7 classes: shipwreck, airplane, mine, human, engineering platform, pipeline/cable, residual mound)
  2. Turntable 18-Class Debris Dataset (17 classes: cans, bottles, tires, cartons, wrenches, valves, etc.)

Total: 24 Unified Classes across Sonar + Optical Marine Debris.
"""

import sys
import shutil
import yaml
from pathlib import Path
from typing import Dict, Any

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

# Define Master 24-Class Taxonomy
MASTER_CLASSES = [
    # ── Sonar Targets (0 - 6) ──
    "shipwreck",                    # 0
    "airplane",                     # 1
    "mine",                         # 2
    "human",                        # 3
    "engineering platform",         # 4
    "pipeline or cable",            # 5
    "underwater residual mound",     # 6
    
    # ── Marine Debris & Litter Objects (7 - 23) ──
    "brown-glass-bottle",           # 7
    "can",                          # 8
    "drink-carton",                 # 9
    "drink-sachet",                 # 10
    "glass-bottle",                 # 11
    "glass-jar",                    # 12
    "large-tire",                   # 13
    "metal-bottle",                 # 14
    "metal-box",                    # 15
    "plastic-bidon",                # 16
    "plastic-bottle",               # 17
    "plastic-pipe",                 # 18
    "plastic-propeller",            # 19
    "potion-glass-bottle",          # 20
    "small-tire",                   # 21
    "valve",                        # 22
    "wrench",                       # 23
]

SONAR_CLASSES = [
    "shipwreck",
    "airplane",
    "mine",
    "human",
    "engineering platform",
    "pipeline or cable",
    "underwater residual mound"
]

DEBRIS_CLASSES = [
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


def build_master_dataset(
    sonar_dir: str = "Unified_Sonar_Dataset",
    debris_dir: str = "Turntable_18class_YOLO",
    out_dir: str = "Master_Universal_24class_YOLO"
):
    sonar_path = Path(sonar_dir).resolve()
    debris_path = Path(debris_dir).resolve()
    out_path = Path(out_dir).resolve()
    out_path.mkdir(parents=True, exist_ok=True)

    cls2id = {c: i for i, c in enumerate(MASTER_CLASSES)}
    sonar_id_to_master = {i: cls2id[c] for i, c in enumerate(SONAR_CLASSES)}
    debris_id_to_master = {i: cls2id[c] for i, c in enumerate(DEBRIS_CLASSES)}

    print("\n" + "=" * 75)
    print(" 🌟 BUILDING MASTER UNIVERSAL 24-CLASS MARINE & SONAR DATASET")
    print("=" * 75)

    stats = {"train": 0, "val": 0, "test": 0}
    class_counts = {c: 0 for c in MASTER_CLASSES}

    valid_exts = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    for split in ["train", "val", "test"]:
        img_out = out_path / split / "images"
        lbl_out = out_path / split / "labels"
        img_out.mkdir(parents=True, exist_ok=True)
        lbl_out.mkdir(parents=True, exist_ok=True)

        # 1. Process Sonar Dataset
        sonar_img_dir = sonar_path / split / "images"
        sonar_lbl_dir = sonar_path / split / "labels"

        if sonar_img_dir.exists():
            for img_file in sonar_img_dir.iterdir():
                if img_file.suffix.lower() in valid_exts:
                    dst_name = f"sonar_{img_file.name}"
                    shutil.copy2(str(img_file), str(img_out / dst_name))
                    stats[split] += 1

                    lbl_file = sonar_lbl_dir / f"{img_file.stem}.txt"
                    dst_lbl = lbl_out / f"{Path(dst_name).stem}.txt"
                    if lbl_file.exists():
                        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                        remapped_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_cid = int(parts[0])
                            new_cid = sonar_id_to_master.get(old_cid, 0)
                            remapped_lines.append(f"{new_cid} {' '.join(parts[1:])}\n")
                            class_counts[MASTER_CLASSES[new_cid]] += 1
                        dst_lbl.write_text("".join(remapped_lines), encoding="utf-8")
                    else:
                        dst_lbl.write_text("", encoding="utf-8")

        # 2. Process Debris Dataset
        debris_img_dir = debris_path / split / "images"
        debris_lbl_dir = debris_path / split / "labels"

        if debris_img_dir.exists():
            for img_file in debris_img_dir.iterdir():
                if img_file.suffix.lower() in valid_exts:
                    dst_name = f"debris_{img_file.name}"
                    shutil.copy2(str(img_file), str(img_out / dst_name))
                    stats[split] += 1

                    lbl_file = debris_lbl_dir / f"{img_file.stem}.txt"
                    dst_lbl = lbl_out / f"{Path(dst_name).stem}.txt"
                    if lbl_file.exists():
                        lines = lbl_file.read_text(encoding="utf-8").strip().splitlines()
                        remapped_lines = []
                        for line in lines:
                            parts = line.strip().split()
                            if not parts:
                                continue
                            old_cid = int(parts[0])
                            new_cid = debris_id_to_master.get(old_cid, 7)
                            remapped_lines.append(f"{new_cid} {' '.join(parts[1:])}\n")
                            class_counts[MASTER_CLASSES[new_cid]] += 1
                        dst_lbl.write_text("".join(remapped_lines), encoding="utf-8")
                    else:
                        dst_lbl.write_text("", encoding="utf-8")

    # Generate unified data.yaml
    yaml_dict = {
        "path": str(out_path).replace("\\", "/"),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": len(MASTER_CLASSES),
        "names": MASTER_CLASSES,
    }

    yaml_out_path = out_path / "data.yaml"
    with open(yaml_out_path, "w", encoding="utf-8") as f:
        yaml.dump(yaml_dict, f, sort_keys=False)

    print(f"📁 Master Dataset Path : {out_path}")
    print(f"📦 Total Images        : {sum(stats.values())}")
    print(f"   • Train             : {stats['train']}")
    print(f"   • Val               : {stats['val']}")
    print(f"   • Test              : {stats['test']}")
    print(f"🏷️ Total Classes ({len(MASTER_CLASSES)}):")
    for i, name in enumerate(MASTER_CLASSES):
        print(f"   [{i:02d}] {name:<26}: {class_counts[name]:>5} boxes")
    print("=" * 75 + "\n")
    return str(yaml_out_path)


if __name__ == "__main__":
    build_master_dataset()
