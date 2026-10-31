"""B0 (per-frequency linear time->depth, fitted on TRAIN GT boxes) applied to PREDICTED boxes of the test detectors.
Detections with conf >= 0.25 are matched to ground-truth targets (class-aware, IoU >= 0.5, greedy by confidence).
usage: python scripts/t2_b0_predicted.py PRED.json NAME"""
import json, sys
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.label_refraction import wavelet_support_ns, load_profiles
pred = json.load(open(sys.argv[1])); name = sys.argv[2]
T = pd.read_csv("results/t2/targets.csv"); coef = json.load(open("results/t2/t2_cpu_baselines.json"))["coef_B0"]
G = pd.read_csv("results/detectors_test/test_gt_boxes.csv"); P = load_profiles("configs/dataset_physics_v2.json")
tt = T[T.split == "test"].copy(); tt["j"] = tt.groupby("sample_id").cumcount(); G["j"] = G.groupby("img").cumcount()
M = G.merge(tt, left_on=["img", "j"], right_on=["sample_id", "j"], suffixes=("", "_t"))
assert len(M) == len(G) and np.allclose(M.depth, M.z_c, atol=1e-6) and (M.cls == M.cls_t).all(), "target alignment failed"
def iou(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None); iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1]) - i + 1e-12)
rows = []
for img, g in M.groupby("img"):
    pr = pred[img]; gb = g[["x0", "y0", "x1", "y1"]].to_numpy(); used = np.zeros(len(g), bool); f = int(g.freq.iloc[0]); tmax = float(P[f]["requested_time_window_ns"])
    for j in np.argsort(-np.array(pr["conf"])) if pr["conf"] else []:
        if pr["conf"][j] < 0.25: continue
        x, y, w, h = pr["xywhn"][j]; c = pr["cls"][j]; b = np.array([x - w/2, y - h/2, x + w/2, y + h/2])
        io = iou(b, gb); io[(g.cls.to_numpy() != c) | used] = -1; bi = int(io.argmax())
        if io[bi] >= 0.5:
            used[bi] = True; r = g.iloc[bi]; t_est = (y - h / 2) * tmax + float(r.t_pre)
            rows.append(dict(img=img, freq=f, z_true=float(r.z_top), z_pred=coef[str(f)][0] * t_est + coef[str(f)][1], e3=float(r.e3), shadowed=bool(r.shadowed), type=r.type, tier=r.tier))
D = pd.DataFrame(rows); n_gt = len(G); e = (D.z_pred - D.z_true).to_numpy(); ae = np.abs(e)
def m(d):
    e = (d.z_pred - d.z_true).to_numpy(); a = np.abs(e)
    return dict(n=int(len(d)), mae_cm=round(float(a.mean() * 100), 2), median_cm=round(float(np.median(a) * 100), 2), within10=round(float((a <= .1).mean()), 3), over_deep10=round(float((e > .1).mean()), 3))
res = dict(model=name, n_gt=n_gt, matched=int(len(D)), match_rate=round(len(D) / n_gt, 3), all=m(D))
D["moist"] = pd.cut(D.e3, [0, 14.8, 19.7, 99], labels=["dry", "mid", "wet"], right=False); D["dbin"] = pd.cut(D.z_true, [-1, .6, 1, 9], labels=["<=0.6", "0.6-1.0", ">1.0"])
for fac in ("freq", "moist", "dbin", "shadowed"): res["by_" + fac] = {str(k): m(x) for k, x in D.groupby(fac, observed=True)}
rng = np.random.default_rng(0); sc = D.img.unique(); idx = {s: np.where(D.img.to_numpy() == s)[0] for s in sc}
res["mae_cm_ci"] = [round(float(np.percentile([ae[np.concatenate([idx[s] for s in rng.choice(sc, len(sc))])].mean() * 100 for _ in range(1000)], q)), 2) for q in (2.5, 97.5)]
json.dump(res, open(f"results/t2/b0_predicted_{name}.json", "w"), indent=1); print(json.dumps(res))
