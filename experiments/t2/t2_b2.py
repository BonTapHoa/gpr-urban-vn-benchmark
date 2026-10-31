"""B2: classical hyperbola-fitting depth baseline (GPU grid search), uses only the IMAGE and a region of interest from a box.
For a target, within its box (x-range, time-range) search apex position x0, apex time t0 and velocity v of a point-diffractor hyperbola
  t(x) = sqrt(t0^2 + (2 (x-x0)/v)^2)   (uniform medium)
maximising the mean background-removed amplitude |g - rowmedian(g)| along the curve (|dx| <= 0.5 m, curve inside the image).
Depth = v_hat * (t0_hat - tz_f) / 2 where tz_f is ONE time-zero constant per frequency (median over a TRAIN subset of t0_hat - 2 z_top / v_hat),
the only fitted parameter (B0 has two). No scene metadata. Image axes: time = row/640*window (45 ns at 400 MHz, 30 ns at 700 MHz);
lateral = col/640 * line length (3.0 m at 400 MHz = 120 x 2.5 cm; 2.5 m at 700 MHz = 125 x 2.0 cm).
usage: python t2_b2.py TARGETS_CSV IMAGES LABELS WEIGHTS_DIR OUT_DIR [LIMIT]"""
import sys, os, json
import numpy as np, pandas as pd, torch
from PIL import Image
T, IM, LB, WD, OUT = sys.argv[1:6]; LIM = int(sys.argv[6]) if len(sys.argv) > 6 else None
os.makedirs(OUT, exist_ok=True); dev = "cuda"
TMAX = {400: 45.0, 700: 30.0}; LEN = {400: 3.0, 700: 2.5}
VG = torch.linspace(0.04, 0.14, 41, device=dev); NX, NT, ND = 17, 25, 49; HW = 0.5
D = pd.read_csv(T)
def energy(sid):
    g = torch.from_numpy(np.asarray(Image.open(f"{IM}/{sid}.png").convert("L"), np.float32)).to(dev)
    e = (g - g.median(dim=1, keepdim=True).values).abs()
    k = torch.tensor([1., 2., 1.], device=dev) / 4
    return torch.nn.functional.conv2d(e[None, None], k.view(1, 1, 3, 1), padding=(1, 0))[0, 0]
def fit(E, box, f):
    x0n, y0n, x1n, y1n = box; xc = torch.linspace(x0n, x1n, NX, device=dev) * 639; tc = torch.linspace(y0n, y1n, NT, device=dev) * TMAX[f]
    dxm = torch.linspace(-HW, HW, ND, device=dev); dxpx = dxm / LEN[f] * 640
    t = torch.sqrt(tc[:, None, None] ** 2 + (2 * dxm[None, None, :] / VG[None, :, None]) ** 2)           # NT,NV,ND  (ns)
    row = (t / TMAX[f] * 640).round().long()                                                           # NT,NV,ND
    col = (xc[:, None] + dxpx[None, :]).round().long()                                                 # NX,ND
    ok = (row[None] < 640) & (col[:, None, None, :] >= 0) & (col[:, None, None, :] < 640)             # NX,NT,NV,ND
    val = E[row.clamp(max=639)[None].expand(NX, -1, -1, -1), col.clamp(0, 639)[:, None, None, :].expand(-1, NT, len(VG), -1)]
    n = ok.sum(-1); sc = (val * ok).sum(-1) / n.clamp(min=1); sc[n < int(0.6 * ND)] = -1
    i = int(sc.argmax()); ix, it, iv = i // (NT * len(VG)), (i // len(VG)) % NT, i % len(VG)
    return float(tc[it]), float(VG[iv]), float(sc.flatten()[i]), (iv == 0 or iv == len(VG) - 1)
def boxes_gt(sids):
    out = {}
    for sid in sids:
        L = [list(map(float, l.split()[1:5])) for l in open(f"{LB}/{sid}.txt") if l.strip()]
        out[sid] = np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in L])
    return out
# 1) calibration of tz on a TRAIN subset and B2 on test GT boxes
rng = np.random.default_rng(0)
tr_s = rng.choice(D[D.split == "train"].sample_id.unique(), 1500, replace=False); te = D[D.split == "test"]
sel = pd.concat([D[D.sample_id.isin(tr_s)], te]); sids = sel.sample_id.unique()[:LIM]; GB = boxes_gt(sids); rows = []
for n, sid in enumerate(sids):
    g = sel[sel.sample_id == sid]; E = energy(sid)
    for j, (ix, r) in enumerate(g.iterrows()):
        t0, v, s, edge = fit(E, GB[sid][j], int(r.freq)); rows.append(dict(sample_id=sid, k=int(r.k), split=r.split, freq=int(r.freq), z_top=r.z_top, t0=t0, v=v, score=s, edge=edge, e3=r.e3, shadowed=r.shadowed))
    if n % 200 == 0: print("gt", n, "/", len(sids), flush=True)
R = pd.DataFrame(rows); tz = {}
for f in (400, 700):
    tr = R[(R.split == "train") & (R.freq == f)]; tz[f] = float(np.median(tr.t0 - 2 * tr.z_top / tr.v))
R["z_b2"] = R.v * (R.t0 - R.freq.map(tz)) / 2
cal = {}
for f in (400, 700):
    tr = R[(R.split == "train") & (R.freq == f)]; cal[f] = np.linalg.lstsq(np.c_[tr.z_b2, np.ones(len(tr))], tr.z_top.to_numpy(), rcond=None)[0].tolist()
R["z_b2c"] = [cal[f][0] * z + cal[f][1] for f, z in zip(R.freq, R.z_b2)]; R.to_csv(f"{OUT}/b2_gt_boxes.csv", index=False); print("tz", tz, "v median", R.v.median(), "edge frac", R.edge.mean(), flush=True)
def metr(e):
    a = np.abs(e) * 100
    return dict(n=int(len(e)), mae=float(a.mean()), med=float(np.median(a)), rmse=float(np.sqrt((a**2).mean())), w5=float((a <= 5).mean()), w10=float((a <= 10).mean()), w20=float((a <= 20).mean()), over10=float((e * 100 > 10).mean()), under10=float((e * 100 < -10).mean()))
def summarize(P, zcol):
    P = P.copy(); P["moist"] = pd.cut(P.e3, [0, 14.8, 19.7, 99], labels=["dry", "mid", "wet"], right=False); P["dbin"] = pd.cut(P.z_top, [-1, .6, 1, 9], labels=["<=0.6", "0.6-1.0", ">1.0"])
    out = {}
    for k, d in [("all", P)] + [(f"moist_{a}", P[P.moist == a]) for a in ("dry", "mid", "wet")] + [(f"dbin_{a}", P[P.dbin == a]) for a in ("<=0.6", "0.6-1.0", ">1.0")] + [("shadowed", P[P.shadowed.astype(bool)])]:
        out[k] = metr((d[zcol] - d.z_top).to_numpy())
    sc = P.sample_id.unique(); ix = P.groupby("sample_id").indices; a = np.abs(P[zcol] - P.z_top).to_numpy() * 100; r2 = np.random.default_rng(0)
    out["mae_ci"] = np.percentile([a[np.concatenate([ix[s] for s in r2.choice(sc, len(sc))])].mean() for _ in range(1000)], [2.5, 97.5]).tolist()
    return out
res = dict(tz=tz, cal_b2c=cal, v_median=float(R.v.median()), edge_fraction_test=float(R[R.split == "test"].edge.mean()), gt_boxes=summarize(R[R.split == "test"], "z_b2"), gt_boxes_b2c=summarize(R[R.split == "test"], "z_b2c"))
# 2) predicted boxes of the released detectors (same matching rule as t2_m1_predicted.py)
from ultralytics import YOLO, RTDETR
te = te.copy(); G = {}
for sid, g in te.groupby("sample_id", sort=False):
    G[sid] = (g, boxes_gt([sid])[sid])
def iou(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None); iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1]) - i + 1e-12)
tsids = list(G)[:LIM]
for name, cls in (("rtdetr_l_s0", RTDETR), ("yolo11m_s0", YOLO)):
    m = cls(f"{WD}/{name}_best.pt"); rows = []; n_gt = 0
    for b in range(0, len(tsids), 32):
        bs = tsids[b:b + 32]; rr = m.predict([f"{IM}/{s}.png" for s in bs], imgsz=640, conf=0.25, iou=0.7, device=0, verbose=False)
        for sid, r in zip(bs, rr):
            g, gb = G[sid]; n_gt += len(g); used = np.zeros(len(g), bool); xyxy = r.boxes.xyxyn.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int); E = None
            for j in np.argsort(-cf):
                io = iou(xyxy[j], gb); io[(g.cls.to_numpy() != cl[j]) | used] = -1; bi = int(io.argmax())
                if io[bi] < 0.5: continue
                used[bi] = True; t = g.iloc[bi]; f = int(t.freq); E = energy(sid) if E is None else E
                t0, v, s, edge = fit(E, xyxy[j], f); rows.append(dict(sample_id=sid, freq=f, z_top=float(t.z_top), z_b2=v * (t0 - tz[f]) / 2, e3=float(t.e3), shadowed=bool(t.shadowed)))
    P = pd.DataFrame(rows); P["z_b2c"] = [cal[f][0] * z + cal[f][1] for f, z in zip(P.freq, P.z_b2)]; P.to_csv(f"{OUT}/b2_pred_{name}.csv", index=False); res[f"pred_{name}"] = dict(matched=len(P), n_gt=n_gt, b2=summarize(P, "z_b2"), b2c=summarize(P, "z_b2c")); print(name, len(P), "/", n_gt, res[f"pred_{name}"]["b2"]["all"]["mae"], res[f"pred_{name}"]["b2c"]["all"]["mae"], flush=True)
json.dump(res, open(f"{OUT}/b2_results.json", "w"), indent=1); print("DONE")
