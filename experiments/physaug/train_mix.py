"""Joint (mixed) training on simulated + few-shot real data, one run per (seed, fraction), resumable. Evaluated on the SAME real test split as R-A9.
Training set = N_SIM random simulated TRAIN images (7 classes, released labels, fixed subset) + the real few-shot training images repeated r times,
relabelled as an 8th class 'real_utility' (class 7) so the model can learn the field box convention (whole hyperbola) next to the simulated one (first lobe).
Checkpoint = best on the REAL val split (as in finetune_real.py). Test = real test split, AP50 of class 7 (the only class with ground truth).
usage: python train_mix.py SIM_IMAGES SIM_LABELS REAL_TREES SPLIT_JSON OUT_JSONL [--seeds 0,1,2] [--fracs 10,25,100] [--epochs 40] [--nsim 2000] [--real_share 0.3]"""
import argparse, glob, json, os, random, shutil, sys, time
from ultralytics import YOLO
ap = argparse.ArgumentParser(); [ap.add_argument(a) for a in ("sim_images", "sim_labels", "real_trees", "split_json", "out")]
ap.add_argument("--seeds", default="0,1,2"); ap.add_argument("--fracs", default="10,25,100"); ap.add_argument("--epochs", type=int, default=40)
ap.add_argument("--nsim", type=int, default=2000); ap.add_argument("--real_share", type=float, default=0.3); ap.add_argument("--init", default="yolo11m.pt"); a = ap.parse_args()
names = ["pec", "upvc_air", "upvc_water", "upvc_triphase", "cable_bundle", "box_culvert", "sinkhole_void", "real_utility"]
train_ids = json.load(open(a.split_json))["splits"]["train"]; sim = sorted(random.Random(12345).sample(train_ids, a.nsim))
done = set()
if os.path.exists(a.out):
    for l in open(a.out): d = json.loads(l); done.add((d["seed"], d["frac"]))
def link(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if not os.path.lexists(dst): os.symlink(src, dst)
def relabel(src, dst):
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    with open(dst, "w") as f:
        for l in open(src):
            p = l.split()
            if len(p) >= 5: f.write("7 " + " ".join(p[1:5]) + "\n")
for seed in map(int, a.seeds.split(",")):
    for frac in map(int, a.fracs.split(",")):
        if (seed, frac) in done: continue
        R = f"{a.real_trees}/s{seed}_f{frac}"; root = f"/content/mix/s{seed}_f{frac}"; t0 = time.time()
        for s in sim:
            link(f"{a.sim_images}/{s}.png", f"{root}/images/train/{s}.png"); link(f"{a.sim_labels}/{s}.txt", f"{root}/labels/train/{s}.txt")
        rt = sorted(glob.glob(f"{R}/images/train/*.jpg")); r = max(1, min(60, round(a.real_share * a.nsim / len(rt))))
        for p in rt:
            b = os.path.splitext(os.path.basename(p))[0]
            for k in range(r): link(p, f"{root}/images/train/{b}_r{k}.jpg"); relabel(f"{R}/labels/train/{b}.txt", f"{root}/labels/train/{b}_r{k}.txt")
        for sp in ("val", "test"):
            for p in sorted(glob.glob(f"{R}/images/{sp}/*.jpg")):
                b = os.path.splitext(os.path.basename(p))[0]; link(p, f"{root}/images/{sp}/{b}.jpg"); relabel(f"{R}/labels/{sp}/{b}.txt", f"{root}/labels/{sp}/{b}.txt")
        open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
        name = f"mix_s{seed}_f{frac}"
        m = YOLO(a.init); m.train(data=f"{root}/data.yaml", epochs=a.epochs, imgsz=640, batch=16, seed=seed, deterministic=True, project="/content/runs_mix", name=name, exist_ok=True, workers=4, amp=True, fliplr=0.5, flipud=0.0, mosaic=0.0, mixup=0.0,
                                  hsv_h=0.0, hsv_s=0.0, hsv_v=0.0, degrees=0.0, translate=0.0, scale=0.0, shear=0.0, perspective=0.0, erasing=0.0, patience=100, plots=False, verbose=False)
        b = YOLO(f"/content/runs_mix/{name}/weights/best.pt"); rr = b.val(data=f"{root}/data.yaml", split="test", imgsz=640, batch=32, plots=False, verbose=False)
        rec = dict(seed=seed, frac=frac, n_real_train=len(rt), real_repeat=r, n_sim=a.nsim, epochs=a.epochs, map50=float(rr.box.map50), map5095=float(rr.box.map), p=float(rr.box.mp), r=float(rr.box.mr), seconds=time.time() - t0)
        open(a.out, "a").write(json.dumps(rec) + "\n"); print(rec, flush=True)
