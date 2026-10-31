"""Run a trained detector on the TEST split, save predictions, and compute AP50 overall / per class / per stratum with
bootstrap CIs over images (true resampling with multiplicity, vectorised). GPU-ready: threaded image prefetch, large
batches, optional fp16, and a batch-1 latency measurement.
usage: python scripts/eval_detectors_test.py WEIGHTS NAME DATA_DIR OUT_DIR [--device 0] [--batch 64] [--half] [--limit N]
DATA_DIR holds images/ and labels_v31/uncorrected/ (the released labels)."""
import argparse, json, sys, time, zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.label_refraction import LabelConfig, label_scene, load_profiles
ap = argparse.ArgumentParser()
ap.add_argument("weights"); ap.add_argument("name"); ap.add_argument("data"); ap.add_argument("out")
ap.add_argument("--limit", type=int, default=0); ap.add_argument("--conf", type=float, default=0.001)
ap.add_argument("--skip_predict", action="store_true"); ap.add_argument("--device", default="cpu")
ap.add_argument("--batch", type=int, default=16); ap.add_argument("--half", action="store_true")
ap.add_argument("--boot", type=int, default=1000); ap.add_argument("--perturb", default="")
ap.add_argument("--labels_dir", default=""); ap.add_argument("--box_mode", default="first_lobe")
a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); data = Path(a.data)
ids = json.load(open("configs/splits/gpr_urban_vn_split_10k_base.json"))["splits"]["test"]
if a.limit: ids = ids[:a.limit]
man = pd.read_parquet("dataset_manifest_v2_10k.parquet").set_index("sample_id"); P = load_profiles("configs/dataset_physics_v2.json")
cfg = LabelConfig(preset="uncorrected", box_mode=a.box_mode)
NAMES = ["pec", "upvc_air", "upvc_water", "upvc_triphase", "cable_bundle", "box_culvert", "sinkhole_void"]
tag = a.name + ("_fp16" if a.half else "") + (("_" + a.perturb.replace(":", "")) if a.perturb else "")
pred_path = out / f"{tag}_test_pred.json"

if not a.skip_predict:
    import cv2, torch
    from ultralytics import YOLO
    torch.backends.cudnn.benchmark = True
    m = YOLO(a.weights); preds = {}; bs = a.batch
    def perturb(img, key):
        if not a.perturb: return img
        kind, _, val = a.perturb.partition(":"); g = img.astype(np.float32); v = float(val) if val else 0.0
        if kind == "noise": g = g + np.random.default_rng(zlib.crc32(key.encode())).normal(0, v, g.shape).astype(np.float32)
        elif kind == "gain": m = g.mean(); g = (g - m) * v + m
        elif kind == "shift":
            k = int(abs(v)); g = np.roll(g, k if v > 0 else -k, axis=0)
            if k: g[:k] = g[k] if v > 0 else g[:k]; g[g.shape[0] - k:] = g[g.shape[0] - k - 1] if v < 0 else g[g.shape[0] - k:]
        elif kind == "bgremove": g = g - g.mean(axis=1, keepdims=True) + 128.0
        else: raise ValueError(a.perturb)
        return np.clip(g, 0, 255).astype(np.uint8)
    rd = lambda s: perturb(cv2.imread(str(data / "images" / f"{s}.png")), s)
    pool = ThreadPoolExecutor(8)
    kw = dict(imgsz=640, conf=a.conf, iou=0.7, max_det=100, device=a.device, half=a.half, verbose=False)
    m.predict([rd(ids[0])] * 2, **kw)                                  # warm-up (cudnn autotune, lazy init)
    batches = [ids[i:i + bs] for i in range(0, len(ids), bs)]
    fut = pool.map(lambda b: [rd(s) for s in b], batches)                # prefetch in worker threads, in order
    t0 = time.time()
    for b, imgs in zip(batches, fut):
        for s, r in zip(b, m.predict(imgs, **kw)):
            preds[s] = dict(xywhn=r.boxes.xywhn.cpu().numpy().round(5).tolist(), conf=r.boxes.conf.cpu().numpy().round(5).tolist(),
                            cls=r.boxes.cls.cpu().numpy().astype(int).tolist())
    dt = time.time() - t0; print(f"predict_seconds {dt:.1f}  ({len(ids) / dt:.1f} img/s, batch {bs}, half={a.half})", flush=True)
    # batch-1 latency (model forward + NMS, image already in memory), 200 images after warm-up
    lat_imgs = [rd(s) for s in ids[:200]]; torch.cuda.synchronize() if a.device != "cpu" else None
    t1 = time.time()
    for im in lat_imgs: m.predict(im, **kw)
    if a.device != "cpu": torch.cuda.synchronize()
    lat = (time.time() - t1) / len(lat_imgs) * 1000; print(f"batch1_latency_ms {lat:.1f}", flush=True)
    json.dump(dict(predict_seconds=dt, imgs_per_s=len(ids) / dt, batch=bs, half=a.half, batch1_latency_ms=lat,
                   device=str(a.device)), open(out / f"{tag}_speed.json", "w"))
    json.dump(preds, open(pred_path, "w"))
preds = json.load(open(pred_path))

# ---- ground truth with per-target metadata
G = []
for s in ids:
    row = man.loc[s]; objs = json.loads(row.objects_json); f = int(s.split("_")[2]); p = P[f]
    meta = {k: row[k] for k in ("h_asphalt_m", "h_base_m", "er_asphalt", "er_base", "er_subgrade")}
    boxes = [dict(b, k=k) for k, b in enumerate(label_scene(objs, meta, p, cfg)) if b["is_inside"]]
    lines = [l.split() for l in (Path(a.labels_dir) / f"{s}.txt" if a.labels_dir else data / "labels_v31/uncorrected" / f"{s}.txt").read_text().split("\n") if l.strip()]
    assert len(lines) == len(boxes), (s, len(lines), len(boxes))
    for b, l in zip(boxes, lines):
        assert int(l[0]) == b["class_id"] and abs(float(l[1]) - b["cx_norm"]) < 1e-4, s
        cx, cy, w, h = map(float, l[1:])
        G.append(dict(img=s, cls=int(l[0]), x0=cx - w / 2, y0=cy - h / 2, x1=cx + w / 2, y1=cy + h / 2,
                      depth=float(objs[b["k"]]["depth"]), freq=f, er=float(row.er_subgrade)))
G = pd.DataFrame(G); print("gt boxes", len(G)); G.to_csv(out / "test_gt_boxes.csv", index=False)
img_ix = {s: i for i, s in enumerate(ids)}; nI = len(ids)

def iou_xyxy(b, B):
    ix = np.clip(np.minimum(b[2], B[:, 2]) - np.maximum(b[0], B[:, 0]), 0, None)
    iy = np.clip(np.minimum(b[3], B[:, 3]) - np.maximum(b[1], B[:, 1]), 0, None)
    i = ix * iy; return i / ((b[2] - b[0]) * (b[3] - b[1]) + (B[:, 2] - B[:, 0]) * (B[:, 3] - B[:, 1]) - i + 1e-12)

recs = []; G_by = {s: g for s, g in G.groupby("img")}
for s in ids:
    pr = preds[s]; g = G_by.get(s)
    gb = g[["x0", "y0", "x1", "y1"]].to_numpy() if g is not None else np.zeros((0, 4)); gc = g.cls.to_numpy() if g is not None else np.zeros(0, int)
    gi = g.index.to_numpy() if g is not None else np.zeros(0, int); used = np.zeros(len(gb), bool)
    for j in (np.argsort(-np.array(pr["conf"])) if pr["conf"] else []):
        x, y, w, h = pr["xywhn"][j]; bx = np.array([x - w / 2, y - h / 2, x + w / 2, y + h / 2]); c = pr["cls"][j]
        best, bi = 0.0, -1
        if len(gb):
            io = iou_xyxy(bx, gb); io[(gc != c) | used] = -1; bi = int(io.argmax()); best = io[bi]
        if best >= 0.5: used[bi] = True; recs.append((img_ix[s], c, pr["conf"][j], 1, int(gi[bi])))
        else: recs.append((img_ix[s], c, pr["conf"][j], 0, -1))
R = pd.DataFrame(recs, columns=["img", "cls", "conf", "tp", "g"]).sort_values("conf", ascending=False, kind="stable")
Rc = {c: (r.img.to_numpy(), r.tp.to_numpy(), r.g.to_numpy()) for c, r in R.groupby("cls")}
Gimg = G.img.map(img_ix).to_numpy(); Gcls = G.cls.to_numpy()

def wap(mult, gmask=None):
    """AP50 per class with image multiplicities `mult` (nI,), GT rows kept by gmask (len G) -- predictions matched to a
    GT outside the stratum are ignored."""
    keep = np.ones(len(G), bool) if gmask is None else gmask; aps = np.full(7, np.nan)
    for c in range(7):
        n = float((mult[Gimg] * (keep & (Gcls == c))).sum())
        if n == 0 or c not in Rc: continue
        pi, tp, g = Rc[c]; w = mult[pi].astype(float)
        if gmask is not None: w = w * ~((tp == 1) & ~keep[np.where(g >= 0, g, 0)] & (g >= 0))
        ctp = np.cumsum(w * tp); cfp = np.cumsum(w * (1 - tp)); rec = ctp / n; prec = ctp / np.maximum(ctp + cfp, 1e-12)
        env = np.maximum.accumulate(prec[::-1])[::-1]
        aps[c] = float(np.sum(np.diff(np.concatenate([[0], rec])) * env))
    return aps

rng = np.random.default_rng(0); B = a.boot
draws = [np.bincount(rng.integers(0, nI, nI), minlength=nI).astype(float) for _ in range(B)]
def ci(fn):
    v = np.array([fn(m) for m in draws], float); return [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
one = np.ones(nI)
aps0 = wap(one); res = dict(name=tag, n_images=nI, n_gt=int(len(G)), boot=B, map50=float(np.nanmean(aps0)),
                            per_class_ap50={NAMES[c]: float(aps0[c]) for c in range(7)},
                            per_class_n={NAMES[c]: int((Gcls == c).sum()) for c in range(7)})
res["map50_ci"] = ci(lambda m: np.nanmean(wap(m)))
res["per_class_ci"] = {NAMES[c]: ci(lambda m, c=c: wap(m)[c]) for c in range(7)}
op = R[R.conf >= 0.25]; res["precision_c25"] = float(op.tp.mean()); res["recall_c25"] = float(op.tp.sum() / len(G))
fimg = {f: np.array([f"_{f}_" in s for s in ids]) for f in (400, 700)}
er_img = np.array([float(man.loc[s].er_subgrade) for s in ids])
strata = {f"freq_{f}": (fimg[f].astype(float), None) for f in (400, 700)}
for name, lo, hi in [("depth_le0.6", -1, 0.6), ("depth_0.6_1.0", 0.6, 1.0), ("depth_gt1.0", 1.0, 9)]:
    strata[name] = (one, ((G.depth > lo) & (G.depth <= hi)).to_numpy())
for name, lo, hi in [("moist_dry", 0, 14.8), ("moist_mid", 14.8, 19.7), ("moist_wet", 19.7, 99)]:
    strata[name] = (((er_img >= lo) & (er_img < hi)).astype(float), None)
tier_img = np.array([str(man.loc[s_].depth_tier) for s_ in ids])
strata["tier_shallow_medium"] = (((tier_img == "SHALLOW") | (tier_img == "MEDIUM")).astype(float), None)
strata["tier_deep"] = ((tier_img == "DEEP").astype(float), None)
res["strata"] = {}
for k, (im, gm) in strata.items():
    n = int(((im[Gimg] > 0) & (gm if gm is not None else True)).sum())
    res["strata"][k] = dict(n_targets=n, map50=float(np.nanmean(wap(im, gm))), ci=ci(lambda m, im=im, gm=gm: np.nanmean(wap(m * im, gm))))
json.dump(res, open(out / f"{tag}_test_eval.json", "w"), indent=1)
print(json.dumps({k: res[k] for k in ("map50", "map50_ci", "per_class_ap50", "precision_c25", "recall_c25")}, indent=1)); print(json.dumps(res["strata"], indent=1))
