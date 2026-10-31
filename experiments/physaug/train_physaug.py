"""YOLO11-m on the released split/labels with physics-informed augmentation (recipe of results/REAL_RESULTS.md R-A2 otherwise identical:
COCO init, 60 epochs, batch 32, 640, flips only, seed 0, deterministic). usage: python train_physaug.py --data DATA_YAML --name NAME [--noaug]"""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import physaug
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("--data", required=True); ap.add_argument("--name", required=True); ap.add_argument("--out", default="/content/runs")
ap.add_argument("--epochs", type=int, default=60); ap.add_argument("--batch", type=int, default=32); ap.add_argument("--seed", type=int, default=0); ap.add_argument("--workers", type=int, default=12)
ap.add_argument("--save_period", type=int, default=-1); ap.add_argument("--noaug", action="store_true"); ap.add_argument("--model", default="yolo11m.pt"); a = ap.parse_args()
if not a.noaug: physaug.patch()
m = YOLO(a.model); t0 = time.time()
m.train(data=a.data, epochs=a.epochs, imgsz=640, batch=a.batch, seed=a.seed, deterministic=True, project=a.out, name=a.name, exist_ok=True, workers=a.workers, amp=True, cache=False,
        fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, erasing=0.0, plots=False, verbose=False, patience=0, save_period=a.save_period)
best = YOLO(os.path.join(a.out, a.name, "weights", "best.pt")); r = best.val(data=a.data, split="test", imgsz=640, batch=a.batch, plots=False, verbose=False)
json.dump(dict(name=a.name, aug=not a.noaug, epochs=a.epochs, seed=a.seed, train_seconds=time.time() - t0, map50=float(r.box.map50), map50_95=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr),
               per_class_ap50={best.names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)}), open(os.path.join(a.out, a.name, "summary.json"), "w"), indent=2)
print("TRAIN_DONE", a.name)
