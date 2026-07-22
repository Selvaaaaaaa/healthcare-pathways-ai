"""
generate_images.py
===================
Synthetic Retinal Fundus Image Generator for Healthcare Pathways AI (Day 3).

Generates procedurally rendered 128x128 synthetic retinal fundus images with
severity-correlated lesions to model a clinical Diabetic Retinopathy (DR)
screening workflow.

Severity Classes:
  0_no_dr    (No DR)       : Healthy fundus canvas, optic disc, clean vessels.
  1_mild     (Mild DR)     : Microaneurysms (small red dots).
  2_moderate (Moderate DR) : Hard exudates (yellow flecks) & hemorrhages (red spots).
  3_severe   (Severe DR)   : Dense exudates, extensive hemorrhages, cotton-wool spots (white lesions).

Directory Structure Created:
  data/images/train/{0_no_dr, 1_mild, 2_moderate, 3_severe}/
  data/images/val/{0_no_dr, 1_mild, 2_moderate, 3_severe}/
  data/images/test/{0_no_dr, 1_mild, 2_moderate, 3_severe}/

Run:
    python src/generate_images.py
"""

import os
import sys
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

RANDOM_SEED = 42
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
IMAGES_DIR = os.path.join(ROOT, "data", "images")

# Class names & subfolders
CLASSES = ["0_no_dr", "1_mild", "2_moderate", "3_severe"]

# Dataset split counts per class
SPLITS = {
    "train": 200,
    "val": 50,
    "test": 50,
}

IMAGE_SIZE = (128, 128)


def create_fundus_image(severity_class: int, rng: np.random.Generator) -> Image.Image:
    """
    Procedurally draw a single retinal fundus image based on severity grade.
    """
    w, h = IMAGE_SIZE
    center = (w // 2, h // 2)
    radius = int(w * 0.45)

    # 1. Background canvas (Dark outer border, warm orange/red fundus interior)
    img = Image.new("RGB", (w, h), (10, 5, 5))
    draw = ImageDraw.Draw(img)

    # Fundus circular disc
    disc_bbox = [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius]
    draw.ellipse(disc_bbox, fill=(185, 65, 30))

    # Inner gradient / color variance
    fundus_inner = [center[0] - int(radius*0.8), center[1] - int(radius*0.8),
                    center[0] + int(radius*0.8), center[1] + int(radius*0.8)]
    draw.ellipse(fundus_inner, fill=(210, 80, 35))

    # 2. Optic Disc (yellowish oval on left)
    optic_cx = int(w * 0.3)
    optic_cy = int(h * 0.5)
    optic_r = int(w * 0.1)
    draw.ellipse([optic_cx - optic_r, optic_cy - optic_r, optic_cx + optic_r, optic_cy + optic_r],
                 fill=(245, 220, 140))

    # 3. Fovea (darker red area on right)
    fovea_cx = int(w * 0.65)
    fovea_cy = int(h * 0.5)
    fovea_r = int(w * 0.08)
    draw.ellipse([fovea_cx - fovea_r, fovea_cy - fovea_r, fovea_cx + fovea_r, fovea_cy + fovea_r],
                 fill=(140, 40, 20))

    # 4. Vascular Branches (dark red lines extending from optic disc)
    vessel_color = (110, 20, 15)
    for _ in range(6):
        end_x = int(optic_cx + rng.uniform(20, 50) * rng.choice([-1, 1]))
        end_y = int(optic_cy + rng.uniform(20, 50) * rng.choice([-1, 1]))
        draw.line([(optic_cx, optic_cy), (end_x, end_y)], fill=vessel_color, width=2)

    # 5. Lesions Correlated with Severity Class
    if severity_class >= 1:
        # Mild DR: Microaneurysms (3–6 tiny dark red dots)
        n_micro = rng.integers(4, 8)
        for _ in range(n_micro):
            lx = rng.integers(int(w * 0.35), int(w * 0.75))
            ly = rng.integers(int(h * 0.25), int(h * 0.75))
            draw.ellipse([lx, ly, lx + 2, ly + 2], fill=(120, 10, 10))

    if severity_class >= 2:
        # Moderate DR: Hard Exudates (bright yellow flecks) + Hemorrhages (dark red spots)
        n_exudates = rng.integers(10, 20)
        for _ in range(n_exudates):
            lx = rng.integers(int(w * 0.35), int(w * 0.8))
            ly = rng.integers(int(h * 0.2), int(h * 0.8))
            draw.rectangle([lx, ly, lx + 3, ly + 3], fill=(250, 240, 130))

        n_hems = rng.integers(8, 15)
        for _ in range(n_hems):
            lx = rng.integers(int(w * 0.35), int(w * 0.8))
            ly = rng.integers(int(h * 0.2), int(h * 0.8))
            draw.ellipse([lx, ly, lx + 4, ly + 4], fill=(90, 5, 5))

    if severity_class >= 3:
        # Severe DR: Dense exudates, extensive hemorrhages, cotton-wool spots (fluffy white lesions)
        n_cotton = rng.integers(3, 7)
        for _ in range(n_cotton):
            lx = rng.integers(int(w * 0.4), int(w * 0.75))
            ly = rng.integers(int(h * 0.25), int(h * 0.75))
            draw.ellipse([lx, ly, lx + 10, ly + 8], fill=(240, 240, 230))

        n_large_hems = rng.integers(12, 22)
        for _ in range(n_large_hems):
            lx = rng.integers(int(w * 0.35), int(w * 0.8))
            ly = rng.integers(int(h * 0.2), int(h * 0.8))
            draw.ellipse([lx, ly, lx + 6, ly + 6], fill=(80, 0, 0))

    # Apply mild Gaussian blur to simulate camera focus
    img = img.filter(ImageFilter.GaussianBlur(radius=0.5))

    return img


def generate_all_images(seed: int = RANDOM_SEED):
    print("=" * 65)
    print("Healthcare Pathways AI — Synthetic Retinal Fundus Dataset Generator")
    print("=" * 65)

    rng = np.random.default_rng(seed)
    total_generated = 0

    for split, count in SPLITS.items():
        print(f"\nGenerating '{split}' split ({count} images per class)...")
        for cls_idx, cls_name in enumerate(CLASSES):
            dir_path = os.path.join(IMAGES_DIR, split, cls_name)
            os.makedirs(dir_path, exist_ok=True)

            for i in range(count):
                img = create_fundus_image(cls_idx, rng)
                filename = f"fundus_{split}_{cls_name}_{i+1:04d}.png"
                img.save(os.path.join(dir_path, filename))
                total_generated += 1

            print(f"  [OK] Saved {count} images in {dir_path}")

    print("\n" + "=" * 65)
    print(f"[OK] Synthetic Dataset Generation Complete! Total: {total_generated} images")
    print(f"     Directory: {IMAGES_DIR}")
    print("=" * 65)


if __name__ == "__main__":
    generate_all_images()
