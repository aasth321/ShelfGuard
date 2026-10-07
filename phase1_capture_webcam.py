"""PHASE 1b - Collect REAL shelf photos with your webcam / phone camera.

SPACE = save frame   |   Q = quit
Tips: vary lighting, distance (30-100 cm), angle (+-15 deg), shelf fill level
and clutter. Aim for 100-200 images per class minimum for real products.
"""
import argparse, time
import cv2
from common import ROOT

ap = argparse.ArgumentParser()
ap.add_argument("--cam", default="0", help="camera index or IP-camera URL")
ap.add_argument("--out", default=str(ROOT / "data/raw/images"))
a = ap.parse_args()

from pathlib import Path
Path(a.out).mkdir(parents=True, exist_ok=True)
cap = cv2.VideoCapture(int(a.cam) if a.cam.isdigit() else a.cam)
n = 0
while True:
    ok, frame = cap.read()
    if not ok:
        break
    view = frame.copy()
    cv2.putText(view, f"saved: {n}  [SPACE]=save [Q]=quit", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.imshow("capture", view)
    k = cv2.waitKey(1) & 0xFF
    if k == ord(" "):
        cv2.imwrite(str(Path(a.out) / f"real_{int(time.time()*1000)}.jpg"), frame)
        n += 1
    elif k == ord("q"):
        break
cap.release(); cv2.destroyAllWindows()
print(f"Saved {n} images to {a.out}")
