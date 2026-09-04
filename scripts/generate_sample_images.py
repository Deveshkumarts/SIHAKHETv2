"""
Generate realistic, high-resolution Side-Scan Sonar & underwater acoustic sample images
for all 27 classes in the SIH 2026 Akhet ontology.
Saves valid PNG images to `samples/` directory.
"""

import os
import sys
from pathlib import Path
import cv2
import numpy as np

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

ROOT_DIR = Path(__file__).resolve().parent.parent
SAMPLES_DIR = ROOT_DIR / "samples"
SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

CLASSES = [
    "Shipwrecks", "bottle", "brown-glass-bottle", "can", "chain",
    "drink-carton", "drink-sachet", "glass-bottle", "glass-jar", "hook",
    "large-tire", "metal-bottle", "metal-box", "pipeline or cable",
    "plastic-bidon", "plastic-bottle", "plastic-pipe", "plastic-propeller",
    "potion-glass-bottle", "propeller", "rotating-platform", "shampoo-bottle",
    "small-tire", "standing-bottle", "tire", "valve", "wrench"
]

def generate_seabed_background(h=640, w=640, seed=42):
    """
    Simulate side-scan sonar seabed backscatter with nadir strip, TVG gradient, and sand ripple textures.
    """
    np.random.seed(seed)
    # Base acoustic backscatter gradient (brighter near nadir, gradual falloff)
    x = np.linspace(-1, 1, w)
    y = np.linspace(-1, 1, h)
    xx, yy = np.meshgrid(x, y)
    
    # Base seabed intensity
    base_intensity = 85 + 25 * (1 - np.abs(xx))
    
    # High-frequency speckle noise (Rayleigh-like)
    speckle = np.random.rayleigh(scale=18, size=(h, w))
    
    # Low-frequency ripple texture (sand dunes / ripples)
    ripple_freq = 0.08 + (seed % 5) * 0.02
    ripples = 15 * np.sin(xx * 50 * ripple_freq + np.sin(yy * 20))
    
    img = base_intensity + speckle + ripples
    
    # Nadir water column line (dark central stripe characteristic of towfish track)
    nadir_center = w // 2 + int(np.sin(seed) * 20)
    nadir_width = 32
    for col in range(nadir_center - nadir_width, nadir_center + nadir_width):
        if 0 <= col < w:
            dist = abs(col - nadir_center) / nadir_width
            darken = 0.15 + 0.85 * dist
            img[:, col] *= darken
            
    img = np.clip(img, 10, 245).astype(np.uint8)
    
    # Convert to 3-channel sonar colormap (copper / acoustic amber-slate)
    bgr = np.zeros((h, w, 3), dtype=np.uint8)
    bgr[:, :, 0] = (img * 0.82).astype(np.uint8)  # Blue
    bgr[:, :, 1] = (img * 0.92).astype(np.uint8)  # Green
    bgr[:, :, 2] = img                            # Red
    return bgr

def draw_acoustic_target(img, class_name, seed=42):
    """
    Draw realistic acoustic highlight + acoustic shadow corresponding to class geometry.
    In side-scan sonar, the shadow extends away from the central nadir line.
    """
    h, w, _ = img.shape
    np.random.seed(seed)
    
    # Position object on port or starboard side (away from center nadir)
    side = 1 if (seed % 2 == 0) else -1
    cx = w // 2 + side * np.random.randint(90, 210)
    cy = np.random.randint(180, 460)
    
    shadow_dir = 1 if cx > w // 2 else -1  # Shadow extends outward from nadir
    
    # Target highlight color (bright specular acoustic return)
    hi_col = (235, 245, 255)
    # Acoustic shadow color (total sound blockage)
    sh_col = (12, 16, 20)
    
    if class_name in ["Shipwrecks", "rotating-platform"]:
        # Large structural target
        length = np.random.randint(140, 220)
        width = np.random.randint(50, 80)
        # Shadow first (long behind object)
        sh_len = int(length * 1.3)
        sh_poly = np.array([
            [cx, cy - length // 2],
            [cx + shadow_dir * sh_len, cy - length // 2 - 20],
            [cx + shadow_dir * sh_len, cy + length // 2 + 20],
            [cx, cy + length // 2]
        ], np.int32)
        cv2.fillPoly(img, [sh_poly], sh_col)
        # Hull highlight
        cv2.ellipse(img, (cx, cy), (width // 2, length // 2), np.random.randint(-15, 15), 0, 360, hi_col, -1)
        # Structural ribs / masts
        for off in range(-length // 3, length // 3, 22):
            cv2.line(img, (cx - width // 2, cy + off), (cx + width // 2, cy + off), (255, 255, 255), 3)

    elif class_name in ["pipeline or cable", "plastic-pipe"]:
        # Long continuous line
        p1 = (cx - shadow_dir * 180, cy - 160)
        p2 = (cx + shadow_dir * 180, cy + 160)
        # Parallel acoustic shadow
        cv2.line(img, (p1[0] + shadow_dir * 25, p1[1]), (p2[0] + shadow_dir * 25, p2[1]), sh_col, 14)
        # Highlight cable
        cv2.line(img, p1, p2, hi_col, 8)
        cv2.line(img, p1, p2, (255, 255, 255), 3)

    elif class_name in ["tire", "small-tire", "large-tire"]:
        radius = 38 if "large" in class_name else (18 if "small" in class_name else 26)
        # Shadow
        sh_poly = np.array([
            [cx, cy - radius],
            [cx + shadow_dir * radius * 3, cy - radius - 8],
            [cx + shadow_dir * radius * 3, cy + radius + 8],
            [cx, cy + radius]
        ], np.int32)
        cv2.fillPoly(img, [sh_poly], sh_col)
        # Torus highlight
        cv2.circle(img, (cx, cy), radius, hi_col, -1)
        cv2.circle(img, (cx, cy), radius // 2, (35, 45, 55), -1)

    elif class_name in ["bottle", "plastic-bottle", "glass-bottle", "brown-glass-bottle", "shampoo-bottle", "potion-glass-bottle", "standing-bottle"]:
        bw, bh = 22, 54
        # Shadow
        sh_poly = np.array([
            [cx, cy - bh // 2],
            [cx + shadow_dir * 70, cy - bh // 2 - 5],
            [cx + shadow_dir * 70, cy + bh // 2 + 5],
            [cx, cy + bh // 2]
        ], np.int32)
        cv2.fillPoly(img, [sh_poly], sh_col)
        # Body highlight
        cv2.rectangle(img, (cx - bw // 2, cy - bh // 3), (cx + bw // 2, cy + bh // 2), hi_col, -1)
        # Bottle neck
        cv2.rectangle(img, (cx - bw // 4, cy - bh // 2), (cx + bw // 4, cy - bh // 3), (255, 255, 255), -1)

    elif class_name in ["can", "metal-bottle", "plastic-bidon", "drink-carton", "drink-sachet", "glass-jar", "metal-box"]:
        cw, ch = 34, 46
        # Shadow
        sh_poly = np.array([
            [cx, cy - ch // 2],
            [cx + shadow_dir * 60, cy - ch // 2 - 6],
            [cx + shadow_dir * 60, cy + ch // 2 + 6],
            [cx, cy + ch // 2]
        ], np.int32)
        cv2.fillPoly(img, [sh_poly], sh_col)
        # Cylinder / Box highlight
        cv2.rectangle(img, (cx - cw // 2, cy - ch // 2), (cx + cw // 2, cy + ch // 2), hi_col, -1)
        cv2.rectangle(img, (cx - cw // 2 + 3, cy - ch // 2 + 3), (cx + cw // 2 - 3, cy + ch // 2 - 3), (255, 255, 255), 2)

    elif class_name in ["wrench", "hook", "valve", "chain"]:
        # Angular tool highlight
        # Shadow
        cv2.ellipse(img, (cx + shadow_dir * 45, cy), (18, 50), 30, 0, 360, sh_col, -1)
        # Handle + head
        cv2.line(img, (cx - 15, cy - 35), (cx + 15, cy + 35), hi_col, 7)
        cv2.circle(img, (cx - 15, cy - 35), 14, hi_col, 5)
        cv2.circle(img, (cx + 15, cy + 35), 10, hi_col, 4)

    elif class_name in ["propeller", "plastic-propeller"]:
        # 3 or 4 blade star
        # Shadow
        cv2.circle(img, (cx + shadow_dir * 50, cy), 35, sh_col, -1)
        # Central hub
        cv2.circle(img, (cx, cy), 12, hi_col, -1)
        # Blades
        for angle in [0, 120, 240]:
            rad = np.radians(angle)
            bx = int(cx + 36 * np.cos(rad))
            by = int(cy + 36 * np.sin(rad))
            cv2.line(img, (cx, cy), (bx, by), hi_col, 8)
            cv2.circle(img, (bx, by), 8, (255, 255, 255), -1)

    else:
        # Generic seabed anomaly
        cv2.ellipse(img, (cx + shadow_dir * 40, cy), (15, 30), 0, 0, 360, sh_col, -1)
        cv2.ellipse(img, (cx, cy), (16, 28), 0, 0, 360, hi_col, -1)

    # Slight Gaussian blur to mimic acoustic transducer beam spread
    blurred = cv2.GaussianBlur(img, (3, 3), 0.8)
    return blurred

def main():
    print(f"Generating realistic Side-Scan Sonar sample images in {SAMPLES_DIR}...")
    for idx, cname in enumerate(CLASSES):
        bg = generate_seabed_background(640, 640, seed=100 + idx * 7)
        sample_img = draw_acoustic_target(bg, cname, seed=100 + idx * 7)
        
        # Save sample
        out_path = SAMPLES_DIR / f"{cname}.png"
        cv2.imwrite(str(out_path), sample_img)
        print(f"  ✓ [{idx+1}/{len(CLASSES)}] Generated: {out_path.name} ({out_path.stat().st_size / 1024:.1f} KB)")
        
    print("\nAll 27 realistic sonar sample images successfully generated!")

if __name__ == "__main__":
    main()
