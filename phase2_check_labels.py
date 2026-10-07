"""PHASE 2 - Validate annotations and render previews.

Annotating REAL images (pick one tool, export in "YOLO" format):
  * Roboflow  (web, easiest, has auto-label assist)
  * CVAT      (free, team-friendly)
  * labelImg  (offline: pip install labelImg)
Put images in data/raw/images and the .txt files in data/raw/labels.
The class order in config.yaml MUST match the order used in the tool.

This script checks: missing labels, bad class ids, out-of-range values,
tiny boxes, duplicate boxes, and saves preview images with boxes drawn.
"""
import argparse
import cv2
from common import ROOT, class_names, list_images

ap = argparse.ArgumentParser()
ap.add_argument("--preview", type=int, default=10, help="how many previews to save")
a = ap.parse_args()

names = class_names()
img_dir = ROOT / "data/raw/images"; lbl_dir = ROOT / "data/raw/labels"
prev_dir = ROOT / "data/preview"; prev_dir.mkdir(parents=True, exist_ok=True)
images = list_images(img_dir)
problems, counts = [], [0] * len(names)

for idx, ip in enumerate(images):
    lp = lbl_dir / f"{ip.stem}.txt"
    if not lp.exists():
        problems.append(f"{ip.name}: no label file"); continue
    boxes, seen = [], set()
    for ln, line in enumerate(lp.read_text().splitlines(), 1):
        p = line.split()
        if len(p) != 5:
            problems.append(f"{lp.name}:{ln} expected 5 values"); continue
        c = int(float(p[0])); x, y, w, h = map(float, p[1:])
        if not 0 <= c < len(names):
            problems.append(f"{lp.name}:{ln} class id {c} out of range"); continue
        if not all(0 <= v <= 1 for v in (x, y, w, h)):
            problems.append(f"{lp.name}:{ln} value outside 0-1"); continue
        if w * h < 1e-4:
            problems.append(f"{lp.name}:{ln} tiny box")
        key = (c, round(x, 3), round(y, 3))
        if key in seen:
            problems.append(f"{lp.name}:{ln} duplicate box")
        seen.add(key); counts[c] += 1; boxes.append((c, x, y, w, h))
    if idx < a.preview:
        img = cv2.imread(str(ip)); H, W = img.shape[:2]
        for c, x, y, w, h in boxes:
            p1 = (int((x - w / 2) * W), int((y - h / 2) * H))
            p2 = (int((x + w / 2) * W), int((y + h / 2) * H))
            cv2.rectangle(img, p1, p2, (0, 255, 0), 2)
            cv2.putText(img, names[c], (p1[0], p1[1] - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.imwrite(str(prev_dir / ip.name), img)

print(f"Images: {len(images)}")
print("Boxes per class:")
for n, c in zip(names, counts):
    flag = "  <-- LOW (<100 recommended for real data)" if c < 100 else ""
    print(f"  {n:15s} {c}{flag}")
print(f"Problems: {len(problems)}")
for p in problems[:50]:
    print("  -", p)
print(f"Previews saved to {prev_dir}")
