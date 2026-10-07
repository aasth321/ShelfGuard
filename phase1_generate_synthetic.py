"""PHASE 1a - Generate synthetic shelf images WITH automatic YOLO labels.

Lets you run the whole pipeline end-to-end without photographing anything.
Each image has 4 shelves x 8 slots. Shelves mostly hold one product type
(a mini planogram), ~15% of slots are empty and ~10% hold a wrong product.
That gives Phases 7-9 realistic situations later (gaps, misplacements).

Output: data/raw/images/*.jpg and data/raw/labels/*.txt
"""
import argparse, random
import cv2
import numpy as np
from common import ROOT, class_names

W, H = 960, 720
SHELF_BOTTOMS = [180, 360, 540, 700]
X0, X1, SLOTS = 40, 920, 8                      # 880 px / 8 = 110 px per slot
COLORS = [(40, 40, 220), (30, 160, 240), (200, 120, 30),
          (60, 180, 60), (0, 200, 230), (180, 60, 160)]          # BGR
SIZES = [(60, 110), (90, 120), (100, 70), (70, 130), (80, 80), (70, 40)]  # (w, h)


def draw_pattern(img, cid, x1, y1, x2, y2):
    cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
    w, h = x2 - x1, y2 - y1
    white = (245, 245, 245)
    if cid == 0:   cv2.rectangle(img, (x1, cy - h // 8), (x2, cy + h // 8), white, -1)
    elif cid == 1: cv2.circle(img, (cx, cy), min(w, h) // 3, white, -1)
    elif cid == 2: cv2.line(img, (x1, y1), (x2, y2), white, 4)
    elif cid == 3: cv2.rectangle(img, (cx - w // 8, y1), (cx + w // 8, y2), white, -1)
    elif cid == 4:
        cv2.circle(img, (x1 + w // 3, cy), h // 6, white, -1)
        cv2.circle(img, (x2 - w // 3, cy), h // 6, white, -1)
    else:
        cv2.line(img, (x1, cy), (x2, cy), white, 3)
        cv2.line(img, (cx, y1), (cx, y2), white, 3)


def make_image(rng, n_classes):
    bg = rng.randint(150, 215)
    img = np.full((H, W, 3), bg, np.uint8)
    img[:] = (img * np.linspace(0.85, 1.05, W)[None, :, None]).clip(0, 255).astype(np.uint8)
    labels = []
    slot_w = (X1 - X0) / SLOTS
    for bottom in SHELF_BOTTOMS:
        cv2.rectangle(img, (X0 - 15, bottom), (X1 + 15, bottom + 12), (60, 70, 90), -1)
        main = rng.randrange(n_classes)
        for s in range(SLOTS):
            if rng.random() < 0.15:                      # empty slot
                continue
            cid = main if rng.random() > 0.10 else rng.randrange(n_classes)
            pw, ph = SIZES[cid % len(SIZES)]
            pw += rng.randint(-4, 4); ph += rng.randint(-4, 4)
            cx = int(X0 + (s + 0.5) * slot_w + rng.randint(-6, 6))
            x1, x2 = cx - pw // 2, cx + pw // 2
            y2 = bottom - 3 + rng.randint(-2, 2)
            y1 = y2 - ph
            col = tuple(int(np.clip(c + rng.randint(-18, 18), 0, 255))
                        for c in COLORS[cid % len(COLORS)])
            cv2.rectangle(img, (x1, y1), (x2, y2), col, -1)
            draw_pattern(img, cid, x1, y1, x2, y2)
            cv2.rectangle(img, (x1, y1), (x2, y2), (25, 25, 25), 2)
            labels.append((cid, (x1 + x2) / 2 / W, (y1 + y2) / 2 / H, pw / W, ph / H))
    # camera-like augmentation
    img = np.clip(img.astype(np.float32) * rng.uniform(0.7, 1.2), 0, 255)
    img += np.random.normal(0, rng.uniform(2, 10), img.shape)
    img = img.clip(0, 255).astype(np.uint8)
    if rng.random() < 0.3:
        img = cv2.GaussianBlur(img, (5, 5), 0)
    return img, labels


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--seed", type=int, default=42)
    a = ap.parse_args()
    rng = random.Random(a.seed); np.random.seed(a.seed)
    names = class_names()
    img_dir = ROOT / "data/raw/images"; lbl_dir = ROOT / "data/raw/labels"
    img_dir.mkdir(parents=True, exist_ok=True); lbl_dir.mkdir(parents=True, exist_ok=True)
    for i in range(a.n):
        img, labels = make_image(rng, len(names))
        stem = f"synth_{i:05d}"
        cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
        with open(lbl_dir / f"{stem}.txt", "w") as f:
            for c, x, y, w, h in labels:
                f.write(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
    print(f"Generated {a.n} images + labels in {img_dir.parent}")


if __name__ == "__main__":
    main()
