"""PHASE 4 - Train YOLO (Ultralytics).

Quick CPU smoke test :  python phase4_train.py --epochs 3 --imgsz 416
Real training (GPU)   :  python phase4_train.py --model yolov8s.pt --epochs 100
Weights end up in     :  runs/detect/shelf/weights/best.pt
"""
import argparse
from ultralytics import YOLO
from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--model", default="yolov8n.pt", help="n=fast, s/m=more accurate")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--batch", type=int, default=16)
ap.add_argument("--device", default=None, help="0 for GPU, 'cpu' for CPU")
ap.add_argument("--name", default="shelf")
a = ap.parse_args()

model = YOLO(a.model)
model.train(
    data=str(ROOT / "dataset/data.yaml"),
    epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, device=a.device,
    project=str(ROOT / "runs/detect"), name=a.name, exist_ok=True,
    patience=20,          # early stopping
    mosaic=1.0, hsv_v=0.4, degrees=5, translate=0.1, scale=0.4,
)
print("Best weights:", ROOT / "runs/detect" / a.name / "weights/best.pt")
