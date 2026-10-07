"""PHASE 3 - Export the YOLO dataset (train/val/test split + data.yaml)."""
import argparse, random, shutil
import yaml
from common import ROOT, class_names, list_images

ap = argparse.ArgumentParser()
ap.add_argument("--val", type=float, default=0.2)
ap.add_argument("--test", type=float, default=0.1)
ap.add_argument("--seed", type=int, default=42)
a = ap.parse_args()

raw_img = ROOT / "data/raw/images"; raw_lbl = ROOT / "data/raw/labels"
out = ROOT / "dataset"
if out.exists():
    shutil.rmtree(out)

pairs = [(p, raw_lbl / f"{p.stem}.txt") for p in list_images(raw_img)]
pairs = [(i, l) for i, l in pairs if l.exists()]
random.Random(a.seed).shuffle(pairs)
n = len(pairs); n_test = int(n * a.test); n_val = int(n * a.val)
splits = {"test": pairs[:n_test], "val": pairs[n_test:n_test + n_val],
          "train": pairs[n_test + n_val:]}

for name, items in splits.items():
    (out / "images" / name).mkdir(parents=True)
    (out / "labels" / name).mkdir(parents=True)
    for i, l in items:
        shutil.copy(i, out / "images" / name / i.name)
        shutil.copy(l, out / "labels" / name / l.name)
    print(f"{name:5s}: {len(items)} images")

names = class_names()
data = {"path": str(out), "train": "images/train", "val": "images/val",
        "test": "images/test", "nc": len(names), "names": {i: n for i, n in enumerate(names)}}
with open(out / "data.yaml", "w") as f:
    yaml.safe_dump(data, f, sort_keys=False)
print(f"Wrote {out / 'data.yaml'}")
