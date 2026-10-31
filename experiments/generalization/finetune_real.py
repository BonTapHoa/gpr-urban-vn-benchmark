"""Few-shot fine-tuning on real 'Utility' boxes from different initialisations (resumable).
usage: python scripts/finetune_real.py TREES_DIR SIM_WEIGHTS OUT_JSONL [--epochs 50]
Inits: sim (detector trained on the simulated data set), coco (yolo11m.pt), scratch (yolo11m.yaml)."""
import argparse, glob, json, os, sys, time
from ultralytics import YOLO
ap = argparse.ArgumentParser(); ap.add_argument("trees"); ap.add_argument("sim"); ap.add_argument("out"); ap.add_argument("--epochs", type=int, default=50)
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--fracs", default="10,25,100"); ap.add_argument("--inits", default="sim,coco,scratch"); a = ap.parse_args()
done = set()
if os.path.exists(a.out):
    for l in open(a.out): d = json.loads(l); done.add((d["seed"], d["frac"], d["init"]))
W = dict(sim=a.sim, coco="yolo11m.pt", scratch="yolo11m.yaml")
for seed in map(int, a.seeds.split(",")):
    for frac in map(int, a.fracs.split(",")):
        for init in a.inits.split(","):
            if (seed, frac, init) in done: continue
            tree = f"{a.trees}/s{seed}_f{frac}"; name = f"ft_s{seed}_f{frac}_{init}"; t0 = time.time()
            m = YOLO(W[init])
            m.train(data=f"{tree}/data.yaml", epochs=a.epochs, imgsz=640, batch=16, seed=seed, deterministic=True, project="/content/runs_ft", name=name, exist_ok=True,
                    workers=4, amp=True, fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0, hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0,
                    scale=0.0, shear=0.0, perspective=0.0, erasing=0.0, patience=100, plots=False, verbose=False)
            b = YOLO(f"/content/runs_ft/{name}/weights/best.pt")
            r = b.val(data=f"{tree}/data.yaml", split="test", imgsz=640, batch=32, plots=False, verbose=False)
            rec = dict(seed=seed, frac=frac, init=init, map50=float(r.box.map50), map5095=float(r.box.map), p=float(r.box.mp), r=float(r.box.mr), seconds=time.time() - t0)
            open(a.out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
