"""Group-by-scan split of the Zenodo rebar GPR set (10.5281/zenodo.16902131, CC-BY-4.0) for the same few-shot protocol as Mojahid.
Only images whose file name carries a scan id (digits or 'info') are used (1,371 of 2,528); the others have unknown provenance and could leak.
For seed s: scans are shuffled (rng 2000+s) and assigned to test until >=30% of images, then val until >=10%, rest = train pool (>=35% required);
train subsets take 10%/25%/100% of the pool images (random, rng 3000+s). Writes assign.json and rebar_named.tgz (flat images+labels).
usage: python make_rebar_assign.py REBAR_DIR OUT_DIR"""
import glob, json, os, re, shutil, sys, tarfile
import numpy as np
R, OUT = sys.argv[1:3]; os.makedirs(OUT, exist_ok=True); imgs = {}
for sp in ("train", "valid", "test"):
    for p in glob.glob(f"{R}/{sp}/images/*.jpg"):
        b = os.path.basename(p); scan = b.split("_")[0]
        if re.fullmatch(r"\d{1,3}|info", scan): imgs[b] = (scan, p, f"{R}/{sp}/labels/{os.path.splitext(b)[0]}.txt")
groups = {}
for b, (s, _, _) in imgs.items(): groups.setdefault(s, []).append(b)
N = len(imgs); print("images", N, "scans", len(groups))
assign = {}
for seed in (0, 1, 2):
    rng = np.random.default_rng(2000 + seed)
    for _ in range(200):
        order = list(rng.permutation(sorted(groups))); test, val, pool = [], [], []
        for g in order:
            if sum(len(groups[x]) for x in test) < 0.30 * N: test.append(g)
            elif sum(len(groups[x]) for x in val) < 0.10 * N: val.append(g)
            else: pool.append(g)
        npool = sum(len(groups[x]) for x in pool)
        if npool >= 0.35 * N: break
    pool_imgs = sorted(b for g in pool for b in groups[g]); r2 = np.random.default_rng(3000 + seed); perm = list(r2.permutation(pool_imgs))
    A = dict(test_scans=test, val_scans=val, pool_scans=pool, test=sorted(b for g in test for b in groups[g]), val=sorted(b for g in val for b in groups[g]))
    for frac in (10, 25, 100): A[f"train_{frac}"] = sorted(perm[:max(1, round(frac / 100 * len(perm)))])
    assign[str(seed)] = A; print("seed", seed, "test", len(A["test"]), "val", len(A["val"]), "pool", len(pool_imgs), "train10/25/100", len(A["train_10"]), len(A["train_25"]), len(A["train_100"]), "test scans", test[:6])
json.dump(assign, open(f"{OUT}/assign.json", "w"), indent=1)
with tarfile.open(f"{OUT}/rebar_named.tgz", "w:gz") as t:
    for b, (s, p, l) in imgs.items(): t.add(p, arcname=f"images/{b}"); t.add(l, arcname=f"labels/{os.path.splitext(b)[0]}.txt")
print("wrote", f"{OUT}/rebar_named.tgz", os.path.getsize(f"{OUT}/rebar_named.tgz") // 1024, "KB")
