"""Control for the simulation pre-training: permute the label files among images of the SAME frequency (no fixed points), so image statistics and box statistics are kept but
labels no longer correspond to the images. usage: python make_shuffled_labels.py SRC_LABELS OUT_LABELS SEED"""
import glob, os, shutil, sys
import numpy as np
src, out, seed = sys.argv[1:4]; os.makedirs(out, exist_ok=True); rng = np.random.default_rng(int(seed))
ids = sorted(os.path.splitext(os.path.basename(p))[0] for p in glob.glob(f"{src}/*.txt")); fixed = 0
for f in ("400", "700"):
    g = [i for i in ids if f"_{f}_" in i]; perm = list(rng.permutation(len(g)))
    for k in range(len(g)):
        if perm[k] == k: j = (k + 1) % len(g); perm[k], perm[j] = perm[j], perm[k]
    for k, i in enumerate(g):
        if perm[k] == k: fixed += 1
        shutil.copy(f"{src}/{g[perm[k]]}.txt", f"{out}/{i}.txt")
print("shuffled", len(ids), "fixed points", fixed)
