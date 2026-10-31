"""Few-shot trees for fine-tuning on the real 'Utility' boxes (Mojahid et al., CC BY 4.0).
Splits are by SOURCE image (all augmentations of one original stay on one side).
For seed s: test = 30% of sources, val = 10%, train pool = rest; train subsets use 10%, 25%, 100% of the pool's sources.
usage: python scripts/build_real_trees.py OUT_DIR"""
import glob, os, re, shutil, sys
import numpy as np
R = "../real_gpr_mojahid/raw/GPR_data/augmented_utilities"; OUT = sys.argv[1]
imgs = sorted(glob.glob(f"{R}/*.jpg")); src = {p: re.match(r"(\d+)", os.path.basename(p)).group(1) for p in imgs}
sids = sorted(set(src.values())); print("sources", len(sids), "images", len(imgs))
def put(paths, root, split):
    os.makedirs(f"{root}/images/{split}", exist_ok=True); os.makedirs(f"{root}/labels/{split}", exist_ok=True)
    for p in paths:
        b = os.path.splitext(os.path.basename(p))[0]
        shutil.copy(p, f"{root}/images/{split}/{b}.jpg"); shutil.copy(f"{R}/annotations/YOLO_format/{b}.txt", f"{root}/labels/{split}/{b}.txt")
for seed in (0, 1, 2):
    rng = np.random.default_rng(1000 + seed); order = list(rng.permutation(sids)); nt, nv = round(0.3 * len(sids)), round(0.1 * len(sids))
    test, val, pool = order[:nt], order[nt:nt + nv], order[nt + nv:]
    for frac in (0.10, 0.25, 1.0):
        tr = pool[:max(1, round(frac * len(pool)))]; root = f"{OUT}/s{seed}_f{int(frac * 100)}"
        for split, ss in (("train", tr), ("val", val), ("test", test)): put([p for p in imgs if src[p] in ss], root, split)
        open(f"{root}/data.yaml", "w").write(f"path: /content/real/s{seed}_f{int(frac * 100)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: utility\n")
        print(seed, frac, "train sources", len(tr), "val", len(val), "test", len(test))
