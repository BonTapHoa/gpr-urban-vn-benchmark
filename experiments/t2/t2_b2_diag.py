"""Diagnostics of B2 (is its poor result an implementation fault or inherent?). GPU.
A synthetic recovery: noise-free/noisy synthetic hyperbola with known (x0,t0,v) -> does fit() recover it?
B score at the TRUE parameters vs the fitted maximum (same objective, point model, true x0, apex time, RMS velocity)
C lateral-scale scan at the true parameters (checks the column->metre conversion): score should peak at scale 1
D velocity-only scan with TRUE x0 and t0: is the velocity observable from the objective at all?
E model variants on a test subset: V0 original (t^2 = t0^2 + (2dx/v)^2), V1 with the system time-zero in the curve,
  V2 = V1 + cylinder radius search. Depth = v (t0 - tz)/2 with tz the SYSTEM time zero (t_air + sqrt(2)/f + 0.37/f), a known constant.
usage: python t2_b2_diag.py TARGETS_CSV IMAGES LABELS B2_GT_CSV OUT_DIR"""
import sys, os, json
import numpy as np, pandas as pd, torch
from PIL import Image
T, IM, LB, B2, OUT = sys.argv[1:6]; os.makedirs(OUT, exist_ok=True); dev = "cuda"
TMAX = {400: 45.0, 700: 30.0}; LEN = {400: 3.0, 700: 2.5}; C0 = 0.2998
VG = torch.linspace(0.04, 0.14, 41, device=dev); NX, NT, ND = 17, 25, 49; HW = 0.5
def energy(sid, arr=None):
    g = torch.from_numpy(np.asarray(Image.open(f"{IM}/{sid}.png").convert("L"), np.float32) if arr is None else arr).to(dev)
    e = (g - g.median(dim=1, keepdim=True).values).abs(); k = torch.tensor([1., 2., 1.], device=dev) / 4
    return torch.nn.functional.conv2d(e[None, None], k.view(1, 1, 3, 1), padding=(1, 0))[0, 0]
def curve_score(E, xc, t, f, scale=1.0):
    """E image, apex column xc (px), t: tensor (...,ND) arrival times (ns) on the dx grid -> mean amplitude along curve (valid points)."""
    dxpx = torch.linspace(-HW, HW, ND, device=dev) / (LEN[f] * scale) * 640
    row = (t / TMAX[f] * 640).round().long(); col = (xc + dxpx).round().long()
    ok = (row < 640) & (row >= 0) & (col >= 0) & (col < 640); val = E[row.clamp(0, 639), col.clamp(0, 639)]
    n = ok.sum(-1); return (val * ok).sum(-1) / n.clamp(min=1), n
DXM = torch.linspace(-HW, HW, ND, device=dev)
def fit(E, box, f, tz=None, radii=(0.0,)):
    x0n, y0n, x1n, y1n = box; xc = torch.linspace(x0n, x1n, NX, device=dev) * 639; tc = torch.linspace(y0n, y1n, NT, device=dev) * TMAX[f]; Rg = torch.tensor(radii, device=dev)
    best = (-1, None)
    # shapes: R,T,V,D
    if tz is None: t = torch.sqrt(tc[None, :, None, None] ** 2 + (2 * DXM[None, None, None, :] / VG[None, None, :, None]) ** 2).expand(len(Rg), -1, -1, -1)
    else:
        tau = (tc - tz)[None, :, None, None]; v = VG[None, None, :, None]; R = Rg[:, None, None, None]
        t = tz + (2 / v) * (torch.sqrt((v * tau / 2 + R) ** 2 + DXM[None, None, None, :] ** 2) - R)
    S = torch.empty(NX, *t.shape[:3], device=dev)
    for i in range(NX):
        sc, n = curve_score(E, xc[i], t, f); sc[n < int(0.6 * ND)] = -1; S[i] = sc
    k = int(S.argmax()); ix, r, it, iv = np.unravel_index(k, S.shape)
    return float(tc[it]), float(VG[iv]), float(Rg[r]), float(S.flatten()[k])
rng = np.random.default_rng(0); res = {}
D = pd.read_csv(T); D["j"] = D.groupby("sample_id").cumcount(); B = pd.read_csv(B2); M = B.merge(D[["sample_id", "k", "j", "type", "z_c", "t_apex", "t_pre", "h1", "h2", "e1", "e2", "e3", "tier"]], on=["sample_id", "k"], suffixes=("", "_t"))
def vrms(r):
    z = r.z_top; zs = [min(z, r.h1), min(max(z - r.h1, 0), r.h2), max(z - r.h1 - r.h2, 0)]; es = [r.e1, r.e2, r.e3]
    t = [d * np.sqrt(e) / C0 for d, e in zip(zs, es)]; v = [C0 / np.sqrt(e) for e in es]; return np.sqrt(sum(vi**2 * ti for vi, ti in zip(v, t)) / sum(t))
M["v_true"] = M.apply(vrms, axis=1)
def boxes(sid):
    L = [list(map(float, l.split()[1:5])) for l in open(f"{LB}/{sid}.txt") if l.strip()]; return np.array([[x - w / 2, y - h / 2, x + w / 2, y + h / 2] for x, y, w, h in L]), np.array([l[0] for l in L])
# ---- A synthetic ----
syn = []
for (f, v, t0, x0) in [(400, 0.08, 22.0, 300), (400, 0.10, 30.0, 320), (700, 0.07, 14.0, 300), (700, 0.09, 18.0, 340)]:
    for noise in (0.0, 8.0):
        img = np.zeros((640, 640), np.float32) + 128; cols = np.arange(640); dx = (cols - x0) / 640 * LEN[f]; tt = np.sqrt(t0**2 + (2 * dx / v) ** 2); rows = tt / TMAX[f] * 640
        for c, r in zip(cols, rows):
            if r < 638: rr = np.arange(640); img[:, c] += 120 * np.exp(-0.5 * ((rr - r) / 2.0) ** 2)
        img += rng.normal(0, noise, img.shape).astype(np.float32); E = energy(None, img)
        box = [(x0 - 60) / 639, (t0 - 2) / TMAX[f], (x0 + 60) / 639, (t0 + 2) / TMAX[f]]; t0h, vh, _, s = fit(E, box, f); syn.append(dict(f=f, v=v, t0=t0, x0=x0, noise=noise, v_hat=vh, t0_hat=t0h))
res["A_synthetic"] = syn
# ---- B, C, D on 400 random test targets ----
Mt = M[M.split == "test"].sample(400, random_state=0); out = []; scales = [0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.25, 1.4, 1.6]; cs = {s: [] for s in scales}; dv = []
for sid, g in Mt.groupby("sample_id"):
    E = energy(sid); bx, cx = boxes(sid); T_all = D[D.sample_id == sid].reset_index(drop=True)
    for _, r in g.iterrows():
        f = int(r.freq); xc = float(cx[int(r.j)]) * 639
        def sc_at(v, t0, scale=1.0):
            t = torch.sqrt(torch.tensor(t0, device=dev) ** 2 + (2 * DXM / v) ** 2); s, n = curve_score(E, xc, t, f, scale); return float(s)
        s_true = sc_at(r.v_true, r.t_apex); out.append(dict(score_true=s_true, score_fit=r.score, edge=bool(r.edge)))
        for s in scales: cs[s].append(sc_at(r.v_true, r.t_apex, s))
        sc = [sc_at(float(v), r.t_apex) for v in VG]; dv.append(dict(v_true=r.v_true, v_scan=float(VG[int(np.argmax(sc))]), z_top=r.z_top, t_apex=r.t_apex, freq=f))
O = pd.DataFrame(out); res["B_score_true_vs_fit"] = dict(n=len(O), true_gt_fit=float((O.score_true > O.score_fit).mean()), median_ratio_true_over_fit=float((O.score_true / O.score_fit).median()), mean_true=float(O.score_true.mean()), mean_fit=float(O.score_fit.mean()))
res["C_lateral_scale_mean_score"] = {str(s): float(np.mean(v)) for s, v in cs.items()}
Dv = pd.DataFrame(dv); res["D_v_only_with_true_x0_t0"] = dict(corr=float(np.corrcoef(Dv.v_scan, Dv.v_true)[0, 1]), ratio_median=float((Dv.v_scan / Dv.v_true).median()), ratio_p10=float((Dv.v_scan / Dv.v_true).quantile(.1)), ratio_p90=float((Dv.v_scan / Dv.v_true).quantile(.9)), edge=float(((Dv.v_scan <= 0.0401) | (Dv.v_scan >= 0.1399)).mean()))
# ---- E variants on a test subset ----
TZ = {}
for f in (400, 700):
    fg = f / 1000.0; tair = 2 * 0.05 / C0; TZ[f] = tair + np.sqrt(2.0) / fg + 0.37 / fg
res["E_system_tz_ns"] = TZ
sub_s = rng.choice(M[M.split == "test"].sample_id.unique(), 500, replace=False); Ms = M[M.sample_id.isin(sub_s)]; rows = []
radii = (0.0, 0.025, 0.05, 0.075, 0.1, 0.15, 0.2)
for sid, g in Ms.groupby("sample_id"):
    E = energy(sid); bx, _ = boxes(sid)
    for _, r in g.iterrows():
        f = int(r.freq); b = bx[int(r.j)]; t0a, va, _, _ = fit(E, b, f, None, (0.0,)); t0b, vb, _, _ = fit(E, b, f, TZ[f], (0.0,)); t0c, vc, Rc, _ = fit(E, b, f, TZ[f], radii)
        rows.append(dict(sample_id=sid, freq=f, z_top=r.z_top, v_true=r.v_true, e3=r.e3, type=r.type, v0=va, t0_0=t0a, v1=vb, t0_1=t0b, v2=vc, t0_2=t0c, R2=Rc, z_c=r.z_c, z_b2=r.z_b2 if "z_b2" in r else np.nan))
V = pd.DataFrame(rows); V.to_csv(f"{OUT}/b2_variants_subset.csv", index=False)
def mae(z): return float(np.abs(z - V.z_top).mean() * 100)
# V0 uses the calibrated tz from the original B2 run (as in the paper); V1/V2 use the SYSTEM tz (no fitting at all)
tz0 = json.load(open(os.path.join(os.path.dirname(B2), "b2_results.json")))["tz"]
V["z0"] = V.v0 * (V.t0_0 - V.freq.map(lambda f: tz0[str(f)])) / 2; V["z1"] = V.v1 * (V.t0_1 - V.freq.map(TZ)) / 2; V["z2"] = V.v2 * (V.t0_2 - V.freq.map(TZ)) / 2
res["E_variants_subset"] = dict(n=len(V), V0_original=mae(V.z0), V1_timezero_in_curve=mae(V.z1), V2_plus_radius=mae(V.z2),
    v_ratio_median=dict(V0=float((V.v0 / V.v_true).median()), V1=float((V.v1 / V.v_true).median()), V2=float((V.v2 / V.v_true).median())),
    v_corr=dict(V0=float(np.corrcoef(V.v0, V.v_true)[0, 1]), V1=float(np.corrcoef(V.v1, V.v_true)[0, 1]), V2=float(np.corrcoef(V.v2, V.v_true)[0, 1])),
    radius_chosen_median=float(V.R2.median()), within10=dict(V0=float((np.abs(V.z0 - V.z_top) < .1).mean()), V1=float((np.abs(V.z1 - V.z_top) < .1).mean()), V2=float((np.abs(V.z2 - V.z_top) < .1).mean())))
json.dump(res, open(f"{OUT}/b2_diag.json", "w"), indent=1, default=float); print(json.dumps(res, indent=1, default=float))
