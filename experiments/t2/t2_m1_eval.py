"""Evaluate M1 vs B0 on TEST (ground-truth boxes) from m1/pred_s*.csv; bootstrap over scenes; writes m1/m1_eval.json"""
import pandas as pd, numpy as np, json
rng = np.random.default_rng(0); P = [pd.read_csv(f"m1/pred_s{s}.csv") for s in (0, 1, 2)]
T = [p[p.split == "test"].reset_index(drop=True) for p in P]
assert all((t.sample_id == T[0].sample_id).all() and (t.k == T[0].k).all() for t in T)
b = T[0].copy(); b["z_ens"] = np.mean([t.z_m1 for t in T], 0)
b["moist"] = pd.cut(b.e3, [0, 14.8, 19.7, 99], labels=["dry", "mid", "wet"], right=False)
b["depth"] = pd.cut(b.z_top, [-1, 0.6, 1.0, 9], labels=["<=0.6", "0.6-1.0", ">1.0"])
def met(z, d):
    e = (z - d.z_top).to_numpy(); a = np.abs(e) * 100
    return dict(n=len(d), mae=a.mean(), med=np.median(a), rmse=np.sqrt((a**2).mean()), w5=(a <= 5).mean(), w10=(a <= 10).mean(), w20=(a <= 20).mean(), over10=(e * 100 > 10).mean(), under10=(e * 100 < -10).mean())
out = {}
for name, d in [("all", b)] + [(f"moist_{k}", b[b.moist == k]) for k in ("dry", "mid", "wet")] + [(f"depth_{k}", b[b.depth == k]) for k in ("<=0.6", "0.6-1.0", ">1.0")] + [("shadowed", b[b.shadowed]), ("freq400", b[b.freq == 400]), ("freq700", b[b.freq == 700])]:
    i = d.index
    out[name] = dict(B0=met(d.z_b0, d), M1_seeds=[met(t.z_m1.loc[i], d) for t in T], M1_ens=met(d.z_ens, d))
# bootstrap over scenes: MAE difference M1(seed mean of MAE) - B0, and CIs
sc = b.sample_id.unique(); g = {s: ix.to_numpy() for s, ix in b.groupby("sample_id").groups.items()} if False else b.groupby("sample_id").indices
ab0 = np.abs(b.z_b0 - b.z_top).to_numpy() * 100; am = np.mean([np.abs(t.z_m1 - t.z_top).to_numpy() * 100 for t in T], 0)
bs = []; bs0 = []; bsd = []
for _ in range(1000):
    pick = rng.choice(sc, len(sc)); ix = np.concatenate([g[s] for s in pick]); bs.append(am[ix].mean()); bs0.append(ab0[ix].mean()); bsd.append(am[ix].mean() - ab0[ix].mean())
out["bootstrap_scene"] = dict(M1_mae_ci=np.percentile(bs, [2.5, 97.5]).tolist(), B0_mae_ci=np.percentile(bs0, [2.5, 97.5]).tolist(), diff_ci=np.percentile(bsd, [2.5, 97.5]).tolist(), diff=float(am.mean() - ab0.mean()))
json.dump(out, open("m1/m1_eval.json", "w"), indent=1, default=float)
f = lambda m: f"MAE {m['mae']:.2f} med {m['med']:.2f} RMSE {m['rmse']:.2f} w5 {m['w5']:.3f} w10 {m['w10']:.3f} w20 {m['w20']:.3f} over {m['over10']:.3f} under {m['under10']:.3f}"
for k, v in out.items():
    if k == "bootstrap_scene": print(k, v); continue
    print(f"{k:12s} n={v['B0']['n']}\n   B0  {f(v['B0'])}\n   M1s {[round(x['mae'],2) for x in v['M1_seeds']]}  M1ens {f(v['M1_ens'])}")
