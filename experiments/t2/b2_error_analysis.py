"""Error analysis of B2 from b2/b2_gt_boxes.csv + targets.csv (CPU). Decomposes depth error into apex-time error and velocity error."""
import numpy as np, pandas as pd, json
C0 = 0.2998
R = pd.read_csv("b2/b2_gt_boxes.csv"); T = pd.read_csv("targets.csv")
M = R.merge(T[["sample_id", "k", "type", "z_c", "t_apex", "t_pre", "t_top_box", "h1", "h2", "e1", "e2", "e3", "tier"]], on=["sample_id", "k"], suffixes=("", "_t"))
assert len(M) == len(R)
# true RMS velocity down to the crown (layers asphalt/base/subgrade, vertical one-way path)
def vrms(r):
    z = r.z_top; hs = [r.h1, r.h2, max(z - r.h1 - r.h2, 0.0)]; es = [r.e1, r.e2, r.e3]; z1 = min(z, r.h1); z2 = min(max(z - r.h1, 0), r.h2); zs = [z1, z2, max(z - r.h1 - r.h2, 0)]
    t = [d * np.sqrt(e) / C0 for d, e in zip(zs, es)]; v = [C0 / np.sqrt(e) for e in es]
    return np.sqrt(sum(vi**2 * ti for vi, ti in zip(v, t)) / sum(t))
M["v_true"] = M.apply(vrms, axis=1); T0 = M[M.split == "test"].copy(); out = {}
tz = json.load(open("b2/b2_results.json"))["tz"]
def mae(e): return float(np.abs(e).mean() * 100)
T0["tz"] = T0.freq.map(lambda f: tz[str(f)])
T0["z_b2"] = T0.v * (T0.t0 - T0.tz) / 2
out["n"] = len(T0); out["MAE_b2"] = mae(T0.z_b2 - T0.z_top)
# (1) apex-time error of the fit vs true apex time
dt = T0.t0 - T0.t_apex; out["t0_minus_true_apex_ns"] = dict(median=float(dt.median()), mad=float((dt - dt.median()).abs().median()), p10=float(dt.quantile(.1)), p90=float(dt.quantile(.9)), within_1ns=float((dt.abs() < 1).mean()))
# (2) velocity error
out["v_hat_vs_true"] = dict(corr=float(np.corrcoef(T0.v, T0.v_true)[0, 1]), v_hat_median=float(T0.v.median()), v_true_median=float(T0.v_true.median()), ratio_median=float((T0.v / T0.v_true).median()), ratio_p10=float((T0.v / T0.v_true).quantile(.1)), ratio_p90=float((T0.v / T0.v_true).quantile(.9)), edge_frac=float(T0.edge.mean()))
# (3) decomposition: swap in true quantities (same tz)
out["oracle_v_fit_t0_MAE"] = mae(T0.v_true * (T0.t0 - T0.tz) / 2 - T0.z_top)
out["fit_v_true_apex_MAE"] = mae(T0.v * (T0.t_apex - T0.tz) / 2 - T0.z_top)
out["true_v_true_apex_MAE"] = mae(T0.v_true * (T0.t_apex - T0.tz) / 2 - T0.z_top)
# (4) by type, by edge, by score quantile
for k, g in T0.groupby("type"): out.setdefault("by_type", {})[k] = dict(n=len(g), MAE=mae(g.z_b2 - g.z_top), t0_err_med=float((g.t0 - g.t_apex).median()), v_ratio_med=float((g.v / g.v_true).median()), edge=float(g.edge.mean()))
for e, g in T0.groupby("edge"): out.setdefault("by_edge", {})[str(e)] = dict(n=len(g), MAE=mae(g.z_b2 - g.z_top))
T0["sq"] = pd.qcut(T0.score, 4, labels=False)
for q, g in T0.groupby("sq"): out.setdefault("by_score_quartile", {})[str(q)] = dict(n=len(g), MAE=mae(g.z_b2 - g.z_top), v_ratio_med=float((g.v / g.v_true).median()))
json.dump(out, open("b2/b2_error_analysis_cpu.json", "w"), indent=1, default=float); print(json.dumps(out, indent=1, default=float))

# ---- consistent decomposition: each variant gets its OWN best time-zero constant per frequency (optimistic, fitted on the same test rows) ----
def best(vcol, tcol):
    z = pd.Series(index=T0.index, dtype=float)
    for f, g in T0.groupby("freq"):
        c = float(np.median(g[tcol] - 2 * g.z_top / g[vcol])); z[g.index] = g[vcol] * (g[tcol] - c) / 2
    return mae(z - T0.z_top)
dec = dict(fit_v_fit_t0=best("v", "t0"), true_v_fit_t0=best("v_true", "t0"), fit_v_true_apex=best("v", "t_apex"), true_v_true_apex=best("v_true", "t_apex"))
# velocity relative error vs depth error
T0["vrel"] = T0.v / T0.v_true - 1; dec["corr_abs_depth_err_vs_abs_vrel"] = float(np.corrcoef(np.abs(T0.z_b2 - T0.z_top), np.abs(T0.vrel) * T0.z_top)[0, 1])
dec["by_vrel_bin_MAE"] = {str(k): dict(n=len(g), MAE=mae(g.z_b2 - g.z_top)) for k, g in T0.groupby(pd.cut(T0.vrel, [-1, -.1, .05, .15, .3, 5]), observed=True)}
json.dump(dec, open("b2/b2_error_decomposition.json", "w"), indent=1, default=float); print(json.dumps(dec, indent=1, default=float))
