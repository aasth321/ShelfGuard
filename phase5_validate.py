"""PHASE 5 - Validate the trained model (mAP, precision, recall, confusion matrix)."""
import argparse
from ultralytics import YOLO
from common import ROOT, class_names

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "runs/detect/shelf/weights/best.pt"))
ap.add_argument("--split", default="test", choices=["val", "test"])
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--conf", type=float, default=0.001)
a = ap.parse_args()

model = YOLO(a.weights)
m = model.val(data=str(ROOT / "dataset/data.yaml"), split=a.split, imgsz=a.imgsz,
              conf=a.conf, plots=True, project=str(ROOT / "runs/val"),
              name=a.split, exist_ok=True)

print("\n=== OVERALL ===")
print(f"Precision : {m.box.mp:.3f}")
print(f"Recall    : {m.box.mr:.3f}")
print(f"mAP@50    : {m.box.map50:.3f}")
print(f"mAP@50-95 : {m.box.map:.3f}")
print("\n=== PER CLASS mAP@50-95 ===")
for i, v in enumerate(m.box.maps):
    print(f"  {class_names()[i]:15s} {v:.3f}")
print(f"\nPlots (confusion matrix, PR curve) in: {ROOT / 'runs/val' / a.split}")
print("Target for retail: mAP@50 > 0.90 and no class below 0.70.")
print("If a class is weak -> add more images of it, check label quality, train longer.")
