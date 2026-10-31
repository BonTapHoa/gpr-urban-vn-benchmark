"""#4: does the hyperbola-based depth carry information BEYOND B0?  Regress z_top on [z_B0, z_X] per frequency on TRAIN targets, evaluate on TEST.
usage: python b2_shrink.py B2_CSV [label]   (CSV with columns sample_id,k,split,freq,z_top and a depth column z (physical) )"""
import sys, numpy as np, pandas as pd, json
csv = sys.argv[1]; col = sys.argv[2] if len(sys.argv) > 2 else "z_b2"
R = pd.read_csv(csv); T = pd.read_csv("targets.csv"); T["t_est"] = T.t_top_box + T.t_pre
for f in (400, 700):
    tr = T[(T.split == "train") & (T.freq == f)]; s = np.linalg.lstsq(np.c_[tr.t_est, np.ones(len(tr))], tr.z_top.to_numpy(), rcond=None)[0]; m = T.freq == f; T.loc[m, "z_b0"] = T.loc[m, "t_est"] * s[0] + s[1]
M = R.merge(T[["sample_id", "k", "z_b0"]], on=["sample_id", "k"]); out = {}
for f in (400, 700):
    tr = M[(M.split == "train") & (M.freq == f)]; te = M[(M.split == "test") & (M.freq == f)]
    for name, cols in (("B0 only", ["z_b0"]), ("X only", [col]), ("B0 + X", ["z_b0", col])):
        A = np.c_[tr[cols].to_numpy(), np.ones(len(tr))]; beta = np.linalg.lstsq(A, tr.z_top.to_numpy(), rcond=None)[0]
        pr = np.c_[te[cols].to_numpy(), np.ones(len(te))] @ beta; out.setdefault(name, {})[f] = dict(n=len(te), mae=float(np.abs(pr - te.z_top).mean() * 100), coef=beta.tolist())
for name in out: out[name]["all"] = float(sum(out[name][f]["mae"] * out[name][f]["n"] for f in (400, 700)) / sum(out[name][f]["n"] for f in (400, 700)))
print(json.dumps(out, indent=1))
