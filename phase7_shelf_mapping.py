"""PHASE 7 - Shelf & slot mapping.

Turns raw detections into a structured shelf layout:
    detections -> shelf rows (top to bottom) -> slot grid (left to right)
    -> every slot is "occupied" (with product) or "empty".

Output JSON is the input for Phase 8 (planogram) and Phase 9 (alerts).

Usage
  python phase7_shelf_mapping.py --source photo.jpg                 # YOLO on an image
  python phase7_shelf_mapping.py --source 0                         # live webcam
  python phase7_shelf_mapping.py --source photo.jpg --slots 8 --x-range 40 920   # fixed camera calibration
  python phase7_shelf_mapping.py --source data/raw/images/synth_00000.jpg --from-labels
        (uses the ground-truth label file as "detections": tests the mapping
         logic without needing a trained model)

Slot grid:
  * default (auto): slot pitch is estimated from spacing between neighbours
  * --slots N [--x-range X0 X1]: fixed grid. Use this for a fixed camera so
    empty slots at the shelf edges are not missed.
"""
import argparse, json, time
from pathlib import Path
import cv2
import numpy as np
from common import ROOT, IMG_EXT, class_names


# ------------------------------------------------------------------ core logic
def cluster_shelves(dets):
    """Group detections into shelf rows using the bottom edge (y2): products
    sit on the shelf, so their bottoms line up even when heights differ."""
    med_h = float(np.median([d["y2"] - d["y1"] for d in dets]))
    thr = 0.6 * med_h
    ordered = sorted(dets, key=lambda d: d["y2"])
    rows = [[ordered[0]]]
    for d in ordered[1:]:
        if d["y2"] - np.mean([x["y2"] for x in rows[-1]]) > thr:
            rows.append([d])
        else:
            rows[-1].append(d)
    return rows


def estimate_pitch(rows):
    """Slot pitch = typical distance between neighbouring product centres."""
    diffs = []
    for row in rows:
        cx = sorted((d["x1"] + d["x2"]) / 2 for d in row)
        diffs += list(np.diff(cx))
    widths = [d["x2"] - d["x1"] for r in rows for d in r]
    diffs = [d for d in diffs if d > 0.4 * np.median(widths)]
    if not diffs:
        return float(np.median(widths)) * 1.15
    lo = np.percentile(diffs, 10)
    close = [d for d in diffs if d <= 1.35 * lo]       # ignore multiples caused by gaps
    return float(np.median(close))


def map_shelves(dets, frame_w, slots=None, x_range=None):
    if not dets:
        return {"frame_width": frame_w, "grid": None, "shelves": []}
    rows = cluster_shelves(dets)

    if slots and x_range:
        origin, end, n_cols = x_range[0], x_range[1], slots
    elif slots:                                         # fixed count, extent from data
        origin = min(d["x1"] for d in dets); end = max(d["x2"] for d in dets); n_cols = slots
    else:                                               # fully automatic
        pitch0 = estimate_pitch(rows)
        cxs = [(d["x1"] + d["x2"]) / 2 for d in dets]
        origin, end = min(cxs) - pitch0 / 2, max(cxs) + pitch0 / 2
        n_cols = max(1, int(round((end - origin) / pitch0)))
    pitch = (end - origin) / n_cols

    shelves = []
    for r_idx, row in enumerate(rows, 1):
        top = min(d["y1"] for d in row); bottom = float(np.mean([d["y2"] for d in row]))
        cells = {}
        for d in sorted(row, key=lambda d: -d["conf"]):
            cx = (d["x1"] + d["x2"]) / 2
            col = int(np.clip((cx - origin) // pitch, 0, n_cols - 1))
            cells.setdefault(col, []).append(d)
        slot_list = []
        for col in range(n_cols):
            sid = f"S{r_idx}-C{col + 1}"
            x1, x2 = origin + col * pitch, origin + (col + 1) * pitch
            if col in cells:
                best = cells[col][0]
                slot_list.append({
                    "slot_id": sid, "shelf": r_idx, "col": col + 1, "status": "occupied",
                    "product": best["name"], "class_id": best["cls"],
                    "conf": round(best["conf"], 3),
                    "bbox": [round(best[k], 1) for k in ("x1", "y1", "x2", "y2")],
                    "extra_detections": len(cells[col]) - 1,
                    "slot_x": [round(x1, 1), round(x2, 1)],
                })
            else:
                slot_list.append({
                    "slot_id": sid, "shelf": r_idx, "col": col + 1, "status": "empty",
                    "product": None, "class_id": None, "conf": None, "bbox": None,
                    "extra_detections": 0, "slot_x": [round(x1, 1), round(x2, 1)],
                })
        occ = sum(s["status"] == "occupied" for s in slot_list)
        shelves.append({"shelf": r_idx, "y_top": round(top, 1), "y_bottom": round(bottom, 1),
                        "occupied": occ, "empty": n_cols - occ,
                        "occupancy": round(occ / n_cols, 3), "slots": slot_list})
    return {"frame_width": frame_w,
            "grid": {"columns": n_cols, "x_start": round(origin, 1), "x_end": round(end, 1),
                     "pitch": round(pitch, 1)},
            "shelves": shelves}


# ------------------------------------------------------------------ drawing
def draw_mapping(img, layout):
    vis = img.copy()
    for sh in layout["shelves"]:
        pad = 4
        for s in sh["slots"]:
            x1, x2 = int(s["slot_x"][0]) + pad, int(s["slot_x"][1]) - pad
            y1, y2 = int(sh["y_top"]) - pad, int(sh["y_bottom"]) + pad
            ok = s["status"] == "occupied"
            col = (0, 200, 0) if ok else (0, 0, 255)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 1 if ok else 2)
            cv2.putText(vis, s["slot_id"], (x1 + 2, y2 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, col, 1)
            if ok:
                cv2.putText(vis, s["product"][:10], (x1 + 2, y1 + 12),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.38, (255, 0, 0), 1)
            else:
                cv2.putText(vis, "EMPTY", (x1 + 2, (y1 + y2) // 2),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 255), 1)
        cv2.putText(vis, f"Shelf {sh['shelf']}: {sh['occupancy']*100:.0f}% full",
                    (6, int(sh["y_top"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        cv2.putText(vis, f"Shelf {sh['shelf']}: {sh['occupancy']*100:.0f}% full",
                    (6, int(sh["y_top"]) - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
    return vis


# ------------------------------------------------------------------ detection sources
def dets_from_yolo(model, frame, conf, imgsz):
    r = model.predict(frame, conf=conf, imgsz=imgsz, verbose=False)[0]
    out = []
    for (x1, y1, x2, y2), c, p in zip(r.boxes.xyxy.tolist(), r.boxes.cls.tolist(), r.boxes.conf.tolist()):
        out.append({"cls": int(c), "name": model.names[int(c)], "conf": float(p),
                    "x1": x1, "y1": y1, "x2": x2, "y2": y2})
    return out


def dets_from_labels(img_path, shape):
    H, W = shape[:2]
    names = class_names()
    lp = ROOT / "data/raw/labels" / f"{Path(img_path).stem}.txt"
    out = []
    for line in lp.read_text().splitlines():
        c, x, y, w, h = line.split(); c = int(c); x, y, w, h = (float(v) for v in (x, y, w, h))
        out.append({"cls": c, "name": names[c], "conf": 1.0,
                    "x1": (x - w / 2) * W, "y1": (y - h / 2) * H,
                    "x2": (x + w / 2) * W, "y2": (y + h / 2) * H})
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--weights", default=str(ROOT / "runs/detect/shelf/weights/best.pt"))
    ap.add_argument("--source", default="0")
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--slots", type=int, default=None)
    ap.add_argument("--x-range", type=float, nargs=2, default=None)
    ap.add_argument("--from-labels", action="store_true", help="use GT labels instead of YOLO")
    a = ap.parse_args()

    out = ROOT / "outputs/phase7"; out.mkdir(parents=True, exist_ok=True)
    model = None
    if not a.from_labels:
        from ultralytics import YOLO
        model = YOLO(a.weights)

    def process(frame, path_hint=None):
        dets = dets_from_labels(path_hint, frame.shape) if a.from_labels else \
               dets_from_yolo(model, frame, a.conf, a.imgsz)
        layout = map_shelves(dets, frame.shape[1], a.slots, a.x_range)
        return layout, draw_mapping(frame, layout)

    if Path(a.source).suffix.lower() in IMG_EXT:
        frame = cv2.imread(a.source)
        layout, vis = process(frame, a.source)
        stem = Path(a.source).stem
        (out / f"{stem}_layout.json").write_text(json.dumps(layout, indent=2))
        cv2.imwrite(str(out / f"{stem}_mapped.jpg"), vis)
        for sh in layout["shelves"]:
            print(f"Shelf {sh['shelf']}: {sh['occupied']}/{len(sh['slots'])} occupied "
                  f"({sh['occupancy']*100:.0f}%)")
        print(f"Saved {out / (stem + '_layout.json')} and {out / (stem + '_mapped.jpg')}")
        if not a.from_labels:
            cv2.imshow("shelf map", vis); cv2.waitKey(0)
    else:
        cap = cv2.VideoCapture(int(a.source) if a.source.isdigit() else a.source)
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            layout, vis = process(frame)
            cv2.imshow("shelf map (Q quit, S save JSON)", vis)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("s"):
                ts = int(time.time())
                (out / f"layout_{ts}.json").write_text(json.dumps(layout, indent=2))
                cv2.imwrite(str(out / f"mapped_{ts}.jpg"), vis)
                print("saved", ts)
        cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
