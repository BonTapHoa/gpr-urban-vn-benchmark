"""On the VM: build /content/rebar/s{seed}_f{frac}/ YOLO trees (symlinks) from assign.json and the flat /content/rebar_flat/{images,labels}. usage: python build_rebar_trees_vm.py ASSIGN_JSON"""
import json, os, sys
A = json.load(open(sys.argv[1])); F = "/content/rebar_flat"
for seed, a in A.items():
    for frac in (10, 25, 100):
        root = f"/content/rebar/s{seed}_f{frac}"
        for sp, lst in (("train", a[f"train_{frac}"]), ("val", a["val"]), ("test", a["test"])):
            for kind in ("images", "labels"): os.makedirs(f"{root}/{kind}/{sp}", exist_ok=True)
            for b in lst:
                st = os.path.splitext(b)[0]
                for kind, src in (("images", f"{F}/images/{b}"), ("labels", f"{F}/labels/{st}.txt")):
                    d = f"{root}/{kind}/{sp}/{os.path.basename(src)}"
                    if not os.path.lexists(d): os.symlink(src, d)
        open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: bar\n")
        print(seed, frac, len(a[f"train_{frac}"]), len(a["val"]), len(a["test"]))
