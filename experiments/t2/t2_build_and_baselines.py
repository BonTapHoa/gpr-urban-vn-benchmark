"""T2 (depth) CPU stage: target table from the released labels, baselines B0/B1, and the label-noise floor.
Depth target = crown depth z_top (m). Time used from a box = box top + wavelet half-support (what a detector box gives).
B0: per-frequency linear calibration z = a*t + b fitted on TRAIN targets only.
B1: true layer velocities applied to the box time (inverse of the label formula): CIRCULAR, a bound only.
usage: python scripts/t2_build_and_baselines.py OUT_DIR"""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, ".")
from src.label_refraction import LabelConfig, label_scene, load_profiles, wavelet_support_ns, C0
out = sys.argv[1]; os.makedirs(out, exist_ok=True)
man = pd.read_parquet("dataset_manifest_v2_10k.parquet"); P = load_profiles("configs/dataset_physics_v2.json")
split = json.load(open("configs/splits/gpr_urban_vn_split_10k_base.json"))["splits"]
part = {s: ("val" if k == "validation" else k) for k, v in split.items() for s in v}
cfg = LabelConfig(preset="uncorrected"); cfg_sh = LabelConfig(preset="proposed", fresnel=True)
rows = []
for r in man.itertuples(index=False):
    f = int(r.frequency_mhz); p = P[f]; objs = json.loads(r.objects_json)
    meta = dict(h_asphalt_m=r.h_asphalt_m, h_base_m=r.h_base_m, er_asphalt=r.er_asphalt, er_base=r.er_base, er_subgrade=r.er_subgrade)
    sh = [bb["shadowed"] for bb in label_scene(objs, meta, p, cfg_sh)]
    for k, b in enumerate(label_scene(objs, meta, p, cfg)):
        if not b["is_inside"]: continue
        tmax = float(p["requested_time_window_ns"]); f_ghz = float(p["frequency_hz"]) / 1e9
        t_pre, t_post = wavelet_support_ns(round(f_ghz, 6), cfg.level_db)
        rows.append(dict(sample_id=r.sample_id, split=part[r.sample_id], k=k, cls=b["class_id"], type=b["type"], freq=f,
                         z_top=b["z_top_m"], z_c=float(objs[k]["depth"]), t_apex=b["t_apex_ns"],
                         t_top_box=(b["cy_norm"] - b["h_norm"] / 2) * tmax, t_pre=t_pre, f_ghz=f_ghz, shadowed=bool(sh[k]),
                         h1=r.h_asphalt_m, h2=r.h_base_m, e1=r.er_asphalt, e2=r.er_base, e3=r.er_subgrade, tier=r.depth_tier,
                         t_air=2.0 * float(p.get("source_height_above_ground_m", 0.05)) / C0))
D = pd.DataFrame(rows); D.to_csv(f"{out}/targets.csv", index=False)
print("targets with box:", len(D), D.split.value_counts().to_dict())
D["t_est"] = D.t_top_box + D.t_pre                                  # apex time recovered from the box top

def invert(s_opt, h1, h2, e1, e2, e3):
    """depth whose optical path integral(sqrt(eps) dz) equals s_opt, layers asphalt/base/subgrade."""
    a = np.sqrt(e1) * h1; b = a + np.sqrt(e2) * h2
    return np.where(s_opt <= a, s_opt / np.sqrt(e1), np.where(s_opt <= b, h1 + (s_opt - a) / np.sqrt(e2), h1 + h2 + (s_opt - b) / np.sqrt(e3)))
# B1
t0 = D.t_air + np.sqrt(2.0) / D.f_ghz + 0.37 / D.f_ghz
D["z_b1"] = invert((D.t_est - t0) * C0 / 2.0, D.h1, D.h2, D.e1, D.e2, D.e3)
# B0: per-frequency linear calibration on train
D["z_b0"] = np.nan
coef = {}
for f in (400, 700):
    tr = D[(D.split == "train") & (D.freq == f)]; A = np.c_[tr.t_est, np.ones(len(tr))]
    sol = np.linalg.lstsq(A, tr.z_top.to_numpy(), rcond=None)[0]; coef[f] = [float(sol[0]), float(sol[1])]
    m = D.freq == f; D.loc[m, "z_b0"] = D.loc[m, "t_est"] * sol[0] + sol[1]
print("B0 coefficients (m/ns, m):", coef)

def metrics(d, col):
    e = (d[col] - d.z_top).to_numpy(); ae = np.abs(e)
    return dict(n=int(len(d)), mae_cm=float(ae.mean() * 100), median_cm=float(np.median(ae) * 100), rmse_cm=float(np.sqrt((e ** 2).mean()) * 100),
                within5=float((ae <= .05).mean()), within10=float((ae <= .10).mean()), within20=float((ae <= .20).mean()),
                over_deep10=float((e > .10).mean()), over_shallow10=float((e < -.10).mean()))
D["moist"] = pd.cut(D.e3, [0, 14.8, 19.7, 99], labels=["dry", "mid", "wet"], right=False)
D["depth_bin"] = pd.cut(D.z_top, [-1, 0.6, 1.0, 9], labels=["<=0.6", "0.6-1.0", ">1.0"])
res = {"coef_B0": coef}
for col in ("z_b0", "z_b1"):
    res[col] = {}
    for sp in ("train", "val", "test"):
        d = D[D.split == sp]; res[col][sp] = metrics(d, col)
        if sp == "test":
            for fac in ("freq", "moist", "depth_bin", "type", "shadowed"):
                res[col][f"test_by_{fac}"] = {str(g): metrics(x, col) for g, x in d.groupby(fac, observed=True)}
            # bootstrap over scenes (test)
            rng = np.random.default_rng(0); sc = d.sample_id.unique(); idx = {s: np.where(d.sample_id.to_numpy() == s)[0] for s in sc}; ae = np.abs(d[col] - d.z_top).to_numpy()
            bs = [ae[np.concatenate([idx[s] for s in rng.choice(sc, len(sc))])].mean() * 100 for _ in range(1000)]
            res[col]["test_mae_cm_ci"] = [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]
# label-noise floor: invert the PHYSICAL reference apex time with the true layers and compare with the true crown depth
ref = pd.concat([pd.read_csv("../paper_aciids_2027_civ/results/validation/val400_reference.csv"), pd.read_csv("../paper_aciids_2027_civ/results/validation/val700_reference.csv")], ignore_index=True)
m = ref.merge(D[["sample_id", "k", "z_top", "h1", "h2", "e1", "e2", "e3", "t_air", "f_ghz", "shadowed", "type", "freq"]], on=["sample_id", "k"], suffixes=("_ref", ""))
t0 = m.t_air + np.sqrt(2.0) / m.f_ghz + 0.37 / m.f_ghz
m["z_ref"] = invert((m.t_apex_ns - t0) * C0 / 2.0, m.h1, m.h2, m.e1, m.e2, m.e3)
ae = (m.z_ref - m.z_top).abs() * 100
res["label_floor_val"] = dict(n=int(len(m)), mae_cm=float(ae.mean()), median_cm=float(ae.median()), within10=float(((m.z_ref - m.z_top).abs() <= .10).mean()),
                              mae_cm_unshadowed=float(ae[~m.shadowed].mean()), mae_cm_shadowed=float(ae[m.shadowed].mean()), n_shadowed=int(m.shadowed.sum()))
json.dump(res, open(f"{out}/t2_cpu_baselines.json", "w"), indent=1)
for col in ("z_b0", "z_b1"):
    print("==", col)
    for sp in ("train", "val", "test"): print("  ", sp, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in res[col][sp].items()})
    print("   test MAE CI", [round(x, 2) for x in res[col]["test_mae_cm_ci"]])
print("label floor (val, physical reference apex -> depth):", {k: round(v, 3) if isinstance(v, float) else v for k, v in res["label_floor_val"].items()})
