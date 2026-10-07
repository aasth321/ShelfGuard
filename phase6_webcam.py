"""PHASE 6 - Live test with webcam / video / image.

  python phase6_webcam.py                    # webcam 0
  python phase6_webcam.py --source shelf.mp4
  python phase6_webcam.py --source photo.jpg
Q = quit, S = save screenshot
"""
import argparse, time
from pathlib import Path
import cv2
from ultralytics import YOLO
from common import ROOT, IMG_EXT

ap = argparse.ArgumentParser()
ap.add_argument("--weights", default=str(ROOT / "runs/detect/shelf/weights/best.pt"))
ap.add_argument("--source", default="0")
ap.add_argument("--conf", type=float, default=0.4)
ap.add_argument("--imgsz", type=int, default=640)
a = ap.parse_args()

model = YOLO(a.weights)
out = ROOT / "outputs/phase6"; out.mkdir(parents=True, exist_ok=True)


def annotate(frame, fps=None):
    r = model.predict(frame, conf=a.conf, imgsz=a.imgsz, verbose=False)[0]
    vis = r.plot()
    counts = {}
    for c in r.boxes.cls.tolist():
        counts[model.names[int(c)]] = counts.get(model.names[int(c)], 0) + 1
    y = 25
    for k, v in sorted(counts.items()):
        cv2.putText(vis, f"{k}: {v}", (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        y += 24
    if fps:
        cv2.putText(vis, f"{fps:.1f} FPS", (vis.shape[1] - 120, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return vis


if Path(a.source).suffix.lower() in IMG_EXT:
    vis = annotate(cv2.imread(a.source))
    cv2.imwrite(str(out / f"result_{Path(a.source).name}"), vis)
    cv2.imshow("detections", vis); cv2.waitKey(0)
else:
    cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
    t0 = time.time()
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        t = time.time()
        vis = annotate(frame, 1 / max(time.time() - t, 1e-6))
        cv2.imshow("detections (Q quit, S save)", vis)
        k = cv2.waitKey(1) & 0xFF
        if k == ord("q"):
            break
        if k == ord("s"):
            cv2.imwrite(str(out / f"shot_{int(time.time())}.jpg"), vis)
    cap.release()
cv2.destroyAllWindows()
