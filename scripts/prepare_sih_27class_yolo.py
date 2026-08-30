"""
SIH 2026 - PS 26057 | Team Akhet
Prepare 27-Class YOLO Dataset from SIH DATASETS folder.
Steps:
  1. Reads all 27 class folders from source.
  2. Auto-generates YOLO bounding box labels (whole-image bbox)
  3. Applies 3-stage preprocessing: Median -> Bilateral -> CLAHE.
  4. Splits into train/val/test (80 / 10 / 10).
  5. Writes data.yaml for YOLOv11 training.
"""
import sys, random
from pathlib import Path
import cv2
import numpy as np

SRC_DIR  = Path(r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH DATASETS")
DEST_DIR = Path(r"C:\Users\CMRMuthuthiyagarajan\Downloads\SIH26\SIH_Dataset_27class")
SPLITS = {"train": 0.80, "val": 0.10, "test": 0.10}
SEED   = 42
IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}

CLASS_NAMES = [
    "Shipwrecks","bottle","brown-glass-bottle","can","chain",
    "drink-carton","drink-sachet","glass-bottle","glass-jar","hook",
    "large-tire","metal-bottle","metal-box","pipeline or cable",
    "plastic-bidon","plastic-bottle","plastic-pipe","plastic-propeller",
    "potion-glass-bottle","propeller","rotating-platform","shampoo-bottle",
    "small-tire","standing-bottle","tire","valve","wrench",
]
CLASS_IDX = {name: i for i, name in enumerate(CLASS_NAMES)}

def preprocess(img_bgr, median_k=3, bilat_d=5, bilat_sigma=35.0, clahe_clip=2.0):
    out = cv2.medianBlur(img_bgr, median_k)
    out = cv2.bilateralFilter(out, bilat_d, bilat_sigma, bilat_sigma)
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clahe_clip, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    out = cv2.cvtColor(cv2.merge([l_clahe, a, b]), cv2.COLOR_LAB2BGR)
    return out

def make_yolo_label(class_id):
    return f"{class_id} 0.500000 0.500000 0.920000 0.920000\n"

def main():
    random.seed(SEED)
    for split in SPLITS:
        (DEST_DIR / split / "images").mkdir(parents=True, exist_ok=True)
        (DEST_DIR / split / "labels").mkdir(parents=True, exist_ok=True)

    total_copied = 0
    print(f"\n{'='*60}")
    print(f"  SIH 27-Class YOLO Dataset Preparation")
    print(f"{'='*60}\n")

    for class_name in CLASS_NAMES:
        class_dir = SRC_DIR / class_name
        if not class_dir.exists():
            print(f"  [SKIP] '{class_name}' not found.")
            continue
        class_id = CLASS_IDX[class_name]
        images = [f for f in class_dir.iterdir() if f.suffix.lower() in IMG_EXTS]
        if not images:
            continue
        random.shuffle(images)
        n = len(images)
        n_train = int(n * SPLITS["train"])
        n_val   = int(n * SPLITS["val"])
        split_map = (
            [(p, "train") for p in images[:n_train]] +
            [(p, "val")   for p in images[n_train:n_train + n_val]] +
            [(p, "test")  for p in images[n_train + n_val:]]
        )
        ok = 0
        for img_path, split in split_map:
            img_bgr = cv2.imread(str(img_path))
            if img_bgr is None:
                continue
            img_proc = preprocess(img_bgr)
            out_name = f"{class_name}_{img_path.stem}{img_path.suffix}"
            out_img  = DEST_DIR / split / "images" / out_name
            out_lbl  = DEST_DIR / split / "labels" / (out_name.rsplit(".", 1)[0] + ".txt")
            cv2.imwrite(str(out_img), img_proc)
            out_lbl.write_text(make_yolo_label(class_id))
            ok += 1
        total_copied += ok
        print(f"  [{class_id:2d}] {class_name:<25s} -> {ok:4d} images")

    yaml_content = f"path: {DEST_DIR.as_posix()}\ntrain: train/images\nval:   val/images\ntest:  test/images\n\nnc: {len(CLASS_NAMES)}\nnames:\n"
    for i, name in enumerate(CLASS_NAMES):
        yaml_content += f"  {i}: {name}\n"
    (DEST_DIR / "data.yaml").write_text(yaml_content)

    print(f"\n{'='*60}")
    print(f"  DONE! Total: {total_copied} images prepared")
    for split in ["train", "val", "test"]:
        c = len(list((DEST_DIR / split / "images").glob("*")))
        print(f"  {split:5s}: {c} images")
    print(f"{'='*60}\n")

main()
