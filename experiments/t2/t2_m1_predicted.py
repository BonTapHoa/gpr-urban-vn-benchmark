"""B0 and M1 on PREDICTED boxes of the released-label detectors (test split).
Detections with conf >= 0.25 (Ultralytics predict, iou=0.7) are matched to ground-truth targets: class-aware, IoU >= 0.5, greedy by confidence
(same rule as t2_b0_predicted.py). For each matched detection the arrival time comes from the PREDICTED box top, B0 gives z_b0, and the three
M1 models (seeds 0-2, frozen best-val checkpoints) predict a correction from a crop centred on the PREDICTED box. No test-based choice anywhere.
usage: python t2_m1_predicted.py TARGETS_CSV IMAGES LABELS M1_DIR WEIGHTS_DIR OUT_DIR"""
import sys, os, json
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
from ultralytics import YOLO, RTDETR
T, IM, LB, M1D, WD, OUT = sys.argv[1:7]; os.makedirs(OUT, exist_ok=True); dev = "cuda"
D = pd.read_csv(T); D["t_est"] = D.t_top_box + D.t_pre; coef = {}
for f in (400, 700):
    tr = D[(D.split == "train") & (D.freq == f)]; s = np.linalg.lstsq(np.c_[tr.t_est, np.ones(len(tr))], tr.z_top.to_numpy(), rcond=None)[0]; coef[f] = s
TMAX = {400: 45.0, 700: 30.0}
class Net(nn.Module):
    def __init__(s):
        super().__init__(); r = torchvision.models.resnet18(weights=None); r.fc = nn.Identity(); s.r = r
        s.h = nn.Sequential(nn.Linear(512 + 5, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, x, z):
        x = (x.float() / 255.0 - 0.45) / 0.225; x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        return s.h(torch.cat([s.r(x), z], 1)).squeeze(1)
nets = []
for sd in (0, 1, 2):
    n = Net().to(dev); n.load_state_dict(torch.load(f"{M1D}/best_s{sd}.pt")); n.eval(); nets.append(n)
te = D[D.split == "test"].copy(); G = {}
for sid, g in te.groupby("sample_id", sort=False):
    L = [list(map(float, l.split()[1:5])) for l in open(f"{LB}/{sid}.txt") if l.strip()]; assert len(L) == len(g)
    G[sid] = (g, np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in L]))
def iou(a, B):
    ix = np.clip(np.minimum(a[2], B[:, 2]) - np.maximum(a[0], B[:, 0]), 0, None); iy = np.clip(np.minimum(a[3], B[:, 3]) - np.maximum(a[1], B[:, 1]), 0, None); i = ix * iy
    return i / ((a[2]-a[0])*(a[3]-a[1]) + (B[:, 2]-B[:, 0])*(B[:, 3]-B[:, 1]) - i + 1e-12)
def metr(e):
    a = np.abs(e) * 100
    return dict(n=int(len(e)), mae=float(a.mean()), med=float(np.median(a)), rmse=float(np.sqrt((a**2).mean())), w5=float((a <= 5).mean()), w10=float((a <= 10).mean()), w20=float((a <= 20).mean()), over10=float((e * 100 > 10).mean()), under10=float((e * 100 < -10).mean()))
sids = list(G)
for name, path, cls in (("rtdetr_l_s0", f"{WD}/rtdetr_l_s0_best.pt", RTDETR), ("yolo11m_s0", f"{WD}/yolo11m_s0_best.pt", YOLO)):
    m = cls(path); rows = []; n_gt = 0
    for b in range(0, len(sids), 32):
        bs = sids[b:b + 32]; res = m.predict([f"{IM}/{s}.png" for s in bs], imgsz=640, conf=0.25, iou=0.7, device=0, verbose=False)
        for sid, r in zip(bs, res):
            g, gb = G[sid]; n_gt += len(g); used = np.zeros(len(g), bool)
            xyxy = r.boxes.xyxyn.cpu().numpy(); cf = r.boxes.conf.cpu().numpy(); cl = r.boxes.cls.cpu().numpy().astype(int)
            im = np.pad(np.asarray(Image.open(f"{IM}/{sid}.png").convert("L")), 128)
            for j in np.argsort(-cf):
                io = iou(xyxy[j], gb); io[(g.cls.to_numpy() != cl[j]) | used] = -1; bi = int(io.argmax())
                if io[bi] < 0.5: continue
                used[bi] = True; t = g.iloc[bi]; f = int(t.freq); x0, y0, x1, y1 = xyxy[j]; cx, cy, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
                t_est = (y0) * TMAX[f] + float(t.t_pre); zb0 = coef[f][0] * t_est + coef[f][1]
                px, py = int(round(cx * 640)) + 128, int(round(cy * 640)) + 128; crop = im[py - 128:py + 128, px - 128:px + 128]
                with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                    x = torch.from_numpy(crop[None].copy()).to(dev); z = torch.tensor([[cy, h, w, float(f == 700), zb0]], dtype=torch.float32, device=dev)
                    d = float(np.mean([n(x, z).float().item() for n in nets])) / 10.0
                rows.append(dict(img=sid, freq=f, z_true=float(t.z_top), z_b0=zb0, z_m1=zb0 + d, e3=float(t.e3), shadowed=bool(t.shadowed)))
    P = pd.DataFrame(rows); P.to_csv(f"{OUT}/pred_{name}.csv", index=False)
    P["moist"] = pd.cut(P.e3, [0, 14.8, 19.7, 99], labels=["dry", "mid", "wet"], right=False); P["dbin"] = pd.cut(P.z_true, [-1, .6, 1, 9], labels=["<=0.6", "0.6-1.0", ">1.0"])
    out = dict(model=name, n_gt=n_gt, matched=len(P), match_rate=len(P) / n_gt, note="M1 = mean of the correction of 3 seed models (ensemble)")
    for k, d in [("all", P)] + [(f"moist_{a}", P[P.moist == a]) for a in ("dry", "mid", "wet")] + [(f"dbin_{a}", P[P.dbin == a]) for a in ("<=0.6", "0.6-1.0", ">1.0")] + [("shadowed", P[P.shadowed])]:
        out[k] = dict(B0=metr((d.z_b0 - d.z_true).to_numpy()), M1=metr((d.z_m1 - d.z_true).to_numpy()))
    rng = np.random.default_rng(0); sc = P.img.unique(); ix = P.groupby("img").indices; a0 = np.abs(P.z_b0 - P.z_true).to_numpy() * 100; a1 = np.abs(P.z_m1 - P.z_true).to_numpy() * 100; bb = []
    for _ in range(1000):
        i = np.concatenate([ix[s] for s in rng.choice(sc, len(sc))]); bb.append((a0[i].mean(), a1[i].mean()))
    bb = np.array(bb); out["bootstrap"] = dict(B0_ci=np.percentile(bb[:, 0], [2.5, 97.5]).tolist(), M1_ci=np.percentile(bb[:, 1], [2.5, 97.5]).tolist(), diff_ci=np.percentile(bb[:, 1] - bb[:, 0], [2.5, 97.5]).tolist())
    json.dump(out, open(f"{OUT}/m1_predicted_{name}.json", "w"), indent=1); print(name, "matched", len(P), "/", n_gt, "B0", out["all"]["B0"]["mae"], "M1", out["all"]["M1"]["mae"], flush=True)
