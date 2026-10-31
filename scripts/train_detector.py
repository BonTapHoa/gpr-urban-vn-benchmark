"""Train one detector on a YOLO-format tree and evaluate it on the official test split.

usage: python scripts/train_detector.py --model yolo11m.pt --data /content/yolo_uncorrected/data.yaml \
           --name yolo11m_uncorrected_s0 --epochs 50 --seed 0
Only horizontal flips are used for augmentation (a B-scan flipped left-right is still a valid B-scan).
Writes <out>/<name>/summary.json with mAP50, mAP50-95 and per-class AP50 on the TEST split, plus timing.
"""
import argparse, json, os, time
from ultralytics import YOLO, RTDETR

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--data", required=True)
ap.add_argument("--name", required=True)
ap.add_argument("--out", default="/content/runs")
ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--batch", type=int, default=32)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--imgsz", type=int, default=640)
ap.add_argument("--fraction", type=float, default=1.0, help="share of the training set (speed tests)")
ap.add_argument("--workers", type=int, default=8)
a = ap.parse_args()

cls = RTDETR if "rtdetr" in a.model.lower() else YOLO
model = cls(a.model)
t0 = time.time()
model.train(data=a.data, epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, seed=a.seed, deterministic=True,
            project=a.out, name=a.name, exist_ok=True, workers=a.workers, amp=True, cache="ram",
            fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0,
            degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, erasing=0.0,
            fraction=a.fraction, plots=False, verbose=False, patience=0)
train_s = time.time() - t0
best = os.path.join(a.out, a.name, "weights", "best.pt")
m = cls(best if os.path.exists(best) else os.path.join(a.out, a.name, "weights", "last.pt"))
r = m.val(data=a.data, split="test", imgsz=a.imgsz, batch=a.batch, plots=False, verbose=False)
res = dict(name=a.name, model=a.model, data=a.data, epochs=a.epochs, seed=a.seed, fraction=a.fraction,
           train_seconds=train_s, sec_per_epoch=train_s / a.epochs,
           map50=float(r.box.map50), map50_95=float(r.box.map), precision=float(r.box.mp), recall=float(r.box.mr),
           per_class_ap50={m.names[int(c)]: float(v) for c, v in zip(r.box.ap_class_index, r.box.ap50)})
json.dump(res, open(os.path.join(a.out, a.name, "summary.json"), "w"), indent=2)
print(json.dumps(res, indent=2))
