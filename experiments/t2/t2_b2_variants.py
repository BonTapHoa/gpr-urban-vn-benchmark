"""B2 family: three alternative objectives for estimating the hyperbola velocity (all use only the image + a box ROI).
S  waveform semblance: SIGNED background-removed amplitudes, stack along the curve in a +-W-sample window, score = energy(stack)/(N * sum energy)
H  Hough: binarise |g - rowmedian| at the q-th percentile of the image, dilate +-1 row, score = fraction of curve points on the binary map
P  trace picking: greedy tracking of the first-lobe maximum outward from the apex pick, robust (Huber) regression of t^2 on dx^2 -> v, t0
Hyper-parameters are chosen on 300 TRAIN scenes by MAE after the per-frequency linear calibration; time-zero tz_f (median) and the linear calibration
are fitted on 1500 TRAIN scenes; test (ground-truth boxes) is evaluated once per method with the chosen setting. All settings tried are reported.
usage: python t2_b2_variants.py TARGETS_CSV IMAGES LABELS OUT_DIR [LIMIT]"""
import sys, os, json
import numpy as np, pandas as pd, torch
from PIL import Image
T, IM, LB, OUT = sys.argv[1:5]; LIM = int(sys.argv[5]) if len(sys.argv) > 5 else None; os.makedirs(OUT, exist_ok=True); dev = "cuda"
TMAX = {400: 45.0, 700: 30.0}; LEN = {400: 3.0, 700: 2.5}; VG = torch.linspace(0.04, 0.14, 41, device=dev); NX, NT, ND = 17, 25, 49; HW = 0.5
DXM = torch.linspace(-HW, HW, ND, device=dev); K = torch.tensor([1., 2., 1.], device=dev) / 4
def load(sid):
    g = torch.from_numpy(np.asarray(Image.open(f"{IM}/{sid}.png").convert("L"), np.float32)).to(dev); g = g - g.median(dim=1, keepdim=True).values
    sm = lambda x: torch.nn.functional.conv2d(x[None, None], K.view(1, 1, 3, 1), padding=(1, 0))[0, 0]
    return sm(g), sm(g.abs())                       # signed, energy
def grid(box, f):
    x0n, y0n, x1n, y1n = box; xc = torch.linspace(x0n, x1n, NX, device=dev) * 639; tc = torch.linspace(y0n, y1n, NT, device=dev) * TMAX[f]
    t = torch.sqrt(tc[:, None, None] ** 2 + (2 * DXM[None, None, :] / VG[None, :, None]) ** 2); return xc, tc, t
def pick(S, tc, shape_nt_nv):
    k = int(S.argmax()); it, iv = np.unravel_index(k, shape_nt_nv)
    return float(tc[it]), float(VG[iv]), float(S.flatten()[k])
def fit_S(Es, E, box, f, W):
    xc, tc, t = grid(box, f); dxpx = DXM / LEN[f] * 640; row = (t / TMAX[f] * 640).round().long(); best = None
    offs = torch.arange(-W, W + 1, device=dev); S = torch.empty(NX, NT, len(VG), device=dev)
    for i in range(NX):
        col = (xc[i] + dxpx).round().long(); okc = (col >= 0) & (col < 640)
        rr = row[..., None] + offs; ok = (rr >= 0) & (rr < 640) & okc[None, None, :, None]
        val = Es[rr.clamp(0, 639), col.clamp(0, 639)[None, None, :, None].expand_as(rr)] * ok
        n = ok[..., W].sum(-1); stack = val.sum(2); num = (stack ** 2).sum(-1); den = n * (val ** 2).sum((2, 3)) + 1e-6
        s = num / den; s[n < int(0.6 * ND)] = -1; S[i] = s
    k = int(S.argmax()); ix, it, iv = np.unravel_index(k, S.shape); return float(tc[it]), float(VG[iv]), float(S.flatten()[k])
def fit_H(Es, E, box, f, q):
    xc, tc, t = grid(box, f); dxpx = DXM / LEN[f] * 640; row = (t / TMAX[f] * 640).round().long()
    thr = torch.quantile(E.flatten()[::7], q / 100.0); B = (E > thr).float(); B = torch.nn.functional.max_pool2d(B[None, None], (3, 1), 1, (1, 0))[0, 0]
    S = torch.empty(NX, NT, len(VG), device=dev)
    for i in range(NX):
        col = (xc[i] + dxpx).round().long(); ok = (row < 640) & (col >= 0)[None, None, :] & (col < 640)[None, None, :]
        val = B[row.clamp(max=639), col.clamp(0, 639)[None, None, :].expand_as(row)] * ok; n = ok.sum(-1); s = val.sum(-1) / n.clamp(min=1); s[n < int(0.6 * ND)] = -1; S[i] = s
    k = int(S.argmax()); ix, it, iv = np.unravel_index(k, S.shape); return float(tc[it]), float(VG[iv]), float(S.flatten()[k])
def fit_P(Ee, box, f, thr_frac):
    E = Ee.cpu().numpy(); x0c = int(round((box[0] + box[2]) / 2 * 639)); r0, r1 = int(box[1] * 639), int(box[3] * 639) + 1
    sub = E[r0:r1 + 1, max(x0c - 3, 0):x0c + 4]; r, c = np.unravel_index(sub.argmax(), sub.shape); ra, ca = r0 + r, max(x0c - 3, 0) + c
    nmax = int(HW / LEN[f] * 640); pts = [(0.0, ra * TMAX[f] / 640)]; thr = thr_frac * E[ra, ca]
    for d in (1, -1):
        rp = ra
        for st in range(1, nmax + 1):
            col = ca + d * st
            if col < 0 or col > 639: break
            lo, hi = max(rp - 1, 0), min(rp + 4, 639); rr = lo + int(E[lo:hi + 1, col].argmax())
            if E[rr, col] < thr: break
            rp = rr; pts.append((d * st * LEN[f] / 640, rr * TMAX[f] / 640))
    if len(pts) < 12: return np.nan, np.nan, 0.0
    dx, t = np.array(pts).T; X = np.c_[np.ones_like(dx), dx ** 2]; y = t ** 2; w = np.ones(len(y))
    for _ in range(20):
        beta = np.linalg.lstsq(X * w[:, None] ** .5, y * w ** .5, rcond=None)[0]; res = np.abs(y - X @ beta); s = 1.4826 * np.median(res) + 1e-6; w = np.where(res < 1.5 * s, 1.0, 1.5 * s / res)
    A, Bc = beta
    if Bc <= 0 or A <= 0: return np.nan, np.nan, 0.0
    return float(np.sqrt(A)), float(np.clip(2 / np.sqrt(Bc), 0.04, 0.14)), float(len(pts))
def boxes(sid):
    L = [list(map(float, l.split()[1:5])) for l in open(f"{LB}/{sid}.txt") if l.strip()]; return np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in L])
D = pd.read_csv(T); rng = np.random.default_rng(0); tr_s = rng.choice(D[D.split == "train"].sample_id.unique(), 1500, replace=False); tune_s = tr_s[:300]
te_s = D[D.split == "test"].sample_id.unique()
def run(method, cfg, sids):
    rows = []; g_all = D[D.sample_id.isin(set(sids))]
    for sid, g in g_all.groupby("sample_id", sort=False):
        Es, E = load(sid); bx = boxes(sid)
        for j, (_, r) in enumerate(g.iterrows()):
            f = int(r.freq); b = bx[j]
            t0, v, s = fit_S(Es, E, b, f, cfg) if method == "S" else fit_H(Es, E, b, f, cfg) if method == "H" else fit_P(E, b, f, cfg)
            rows.append(dict(sample_id=sid, k=int(r.k), split=r.split, freq=f, z_top=r.z_top, t0=t0, v=v, score=s, e3=r.e3, shadowed=bool(r.shadowed)))
    return pd.DataFrame(rows)
def calib(R, cal_mask):
    """per-frequency tz (median) and linear calibration, fitted on rows cal_mask; returns z (physical) and zc (calibrated)"""
    R = R.copy(); R["v"] = R.v.fillna(R.v.median()); R["t0"] = R.t0.fillna(R.t0.median()); tz = {}; cal = {}
    for f in (400, 700):
        m = cal_mask & (R.freq == f); tz[f] = float(np.median(R.t0[m] - 2 * R.z_top[m] / R.v[m]))
    R["z"] = R.v * (R.t0 - R.freq.map(tz)) / 2
    for f in (400, 700):
        m = cal_mask & (R.freq == f); cal[f] = np.linalg.lstsq(np.c_[R.z[m], np.ones(m.sum())], R.z_top[m].to_numpy(), rcond=None)[0].tolist()
    R["zc"] = [cal[f][0] * z + cal[f][1] for f, z in zip(R.freq, R.z)]; return R, tz, cal
def metr(e):
    a = np.abs(e) * 100; return dict(n=int(len(e)), mae=float(a.mean()), med=float(np.median(a)), rmse=float(np.sqrt((a**2).mean())), w5=float((a <= 5).mean()), w10=float((a <= 10).mean()), w20=float((a <= 20).mean()), over10=float((e * 100 > 10).mean()), under10=float((e * 100 < -10).mean()))
CFG = dict(S=[1, 2, 3], H=[85, 92, 97], P=[0.15, 0.25, 0.4]); res = {}
tr_use = tr_s[:LIM] if LIM else tr_s; te_use = te_s[:LIM] if LIM else te_s
for m in ("S", "H", "P"):
    tuning = {}
    for c in CFG[m]:
        R = run(m, c, tune_s[:LIM] if LIM else tune_s); R, tz, cal = calib(R, np.ones(len(R), bool)); tuning[str(c)] = dict(mae_cal=float(np.abs(R.zc - R.z_top).mean() * 100), mae_phys=float(np.abs(R.z - R.z_top).mean() * 100), nan_frac=float(R.t0.isna().mean())); print(m, c, tuning[str(c)], flush=True)
    best = min(CFG[m], key=lambda c: tuning[str(c)]["mae_cal"]); R = run(m, best, list(tr_use) + list(te_use)); nanf = float(R.t0.isna().mean())
    R, tz, cal = calib(R, (R.split == "train").to_numpy()); R.to_csv(f"{OUT}/b2{m}_gt_boxes.csv", index=False); Te = R[R.split == "test"]
    res[m] = dict(tuning=tuning, chosen=best, tz=tz, cal=cal, nan_frac_all=nanf, v_median=float(Te.v.median()), edge_frac=float(((Te.v <= 0.0401) | (Te.v >= 0.1399)).mean()), test_phys=metr((Te.z - Te.z_top).to_numpy()), test_cal=metr((Te.zc - Te.z_top).to_numpy()))
    for k, mask in (("dry", Te.e3 < 14.8), ("mid", (Te.e3 >= 14.8) & (Te.e3 < 19.7)), ("wet", Te.e3 >= 19.7)): res[m][f"cal_{k}"] = metr((Te.zc[mask] - Te.z_top[mask]).to_numpy())
    print(m, "chosen", best, "test phys", res[m]["test_phys"]["mae"], "cal", res[m]["test_cal"]["mae"], flush=True); json.dump(res, open(f"{OUT}/b2_variants_results.json", "w"), indent=1, default=float)
print("DONE")
