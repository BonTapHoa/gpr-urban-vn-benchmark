"""T2 learned depth regressor M1: ResNet-18 (ImageNet init) predicts a correction to B0 from an image crop around the box.
Input: 256x256 grey crop centred on the (ground-truth, released) box, + scalars [cy, h, w (normalised box), freq flag, B0 depth].
Output: z_top = z_B0 + delta. No scene metadata (permittivity, layer thickness) is used. L1 loss.
Protocol: train on train, checkpoint chosen on val MAE, TEST evaluated once per seed with the frozen best checkpoint.
usage: python t2_m1_train.py TARGETS_CSV IMAGES_DIR LABELS_DIR OUT_DIR SEED [EPOCHS]"""
import sys, os, json, time
import numpy as np, pandas as pd, torch, torch.nn as nn, torchvision
from PIL import Image
T, IM, LB, OUT, SEED = sys.argv[1:6]; EP = int(sys.argv[6]) if len(sys.argv) > 6 else 15; SEED = int(SEED)
os.makedirs(OUT, exist_ok=True); torch.manual_seed(SEED); np.random.seed(SEED)
D = pd.read_csv(T)
# B0 on the same convention as t2_build_and_baselines.py (fit on train only)
D["t_est"] = D.t_top_box + D.t_pre; D["z_b0"] = np.nan
for f in (400, 700):
    tr = D[(D.split == "train") & (D.freq == f)]; A = np.c_[tr.t_est, np.ones(len(tr))]
    s = np.linalg.lstsq(A, tr.z_top.to_numpy(), rcond=None)[0]; m = D.freq == f; D.loc[m, "z_b0"] = D.loc[m, "t_est"] * s[0] + s[1]
# boxes from the released labels (line order == target order within a sample)
box = np.zeros((len(D), 4), np.float32); i = 0; cache = {}
for sid, g in D.groupby("sample_id", sort=False):
    L = [list(map(float, l.split()[1:5])) for l in open(f"{LB}/{sid}.txt") if l.strip()]
    assert len(L) == len(g), (sid, len(L), len(g))
    for ix, row in zip(g.index, L): box[ix] = row
D[["cx", "cy", "w", "h"]] = box
H = 128; crops = np.zeros((len(D), 256, 256), np.uint8)
for sid, g in D.groupby("sample_id", sort=False):
    im = np.pad(np.asarray(Image.open(f"{IM}/{sid}.png").convert("L")), H, constant_values=0)
    for ix, r in zip(g.index, g.itertuples()):
        cx = int(round(r.cx * 640)) + H; cy = int(round(r.cy * 640)) + H
        crops[ix] = im[cy - H:cy + H, cx - H:cx + H]
sc = np.c_[D.cy, D.h, D.w, (D.freq == 700).astype(float), D.z_b0].astype(np.float32)
y = (D.z_top - D.z_b0).to_numpy(np.float32) * 10.0        # decimetres of correction
sp = D.split.to_numpy(); dev = "cuda"
class Net(nn.Module):
    def __init__(s):
        super().__init__(); r = torchvision.models.resnet18(weights="IMAGENET1K_V1"); r.fc = nn.Identity(); s.r = r
        s.h = nn.Sequential(nn.Linear(512 + 5, 128), nn.ReLU(), nn.Linear(128, 1))
    def forward(s, x, z):
        x = (x.float() / 255.0 - 0.45) / 0.225; x = x.unsqueeze(1).expand(-1, 3, -1, -1)
        return s.h(torch.cat([s.r(x), z], 1)).squeeze(1)
net = Net().to(dev); opt = torch.optim.AdamW(net.parameters(), 3e-4, weight_decay=1e-4)
idx = {k: np.where(sp == k)[0] for k in ("train", "val", "test")}
sched = torch.optim.lr_scheduler.OneCycleLR(opt, 3e-4, total_steps=EP * int(np.ceil(len(idx["train"]) / 128)))
X = torch.from_numpy(crops); Z = torch.from_numpy(sc); Y = torch.from_numpy(y)
def predict(ix):
    net.eval(); out = []
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
        for b in range(0, len(ix), 256):
            j = ix[b:b + 256]; out.append(net(X[j].to(dev), Z[j].to(dev)).float().cpu())
    return torch.cat(out).numpy() / 10.0                    # metres of correction
best, hist = 1e9, []
for ep in range(EP):
    net.train(); perm = np.random.permutation(idx["train"]); t0 = time.time()
    for b in range(0, len(perm), 128):
        j = perm[b:b + 128]; x = X[j].to(dev)
        flip = torch.rand(len(j), device=dev) < 0.5; x = torch.where(flip[:, None, None], x.flip(-1), x)
        with torch.autocast("cuda", dtype=torch.bfloat16): p = net(x, Z[j].to(dev))
        loss = (p.float() - Y[j].to(dev)).abs().mean(); opt.zero_grad(); loss.backward(); opt.step(); sched.step()
    v = idx["val"]; mae = float(np.abs(predict(v) - y[v] / 10.0).mean() * 100)
    hist.append(dict(epoch=ep, val_mae_cm=mae, sec=time.time() - t0)); print(hist[-1], flush=True)
    if mae < best: best = mae; torch.save(net.state_dict(), f"{OUT}/best_s{SEED}.pt")
net.load_state_dict(torch.load(f"{OUT}/best_s{SEED}.pt"))
D["z_m1"] = D.z_b0 + np.nan
for k in ("val", "test"): D.loc[idx[k], "z_m1"] = D.z_b0.to_numpy()[idx[k]] + predict(idx[k])
D[D.split.isin(["val", "test"])][["sample_id", "k", "split", "freq", "z_top", "z_b0", "z_m1", "e3", "tier", "shadowed", "cls"]].to_csv(f"{OUT}/pred_s{SEED}.csv", index=False)
json.dump(dict(seed=SEED, epochs=EP, best_val_mae_cm=best, hist=hist), open(f"{OUT}/hist_s{SEED}.json", "w"))
print("DONE", SEED, best)
