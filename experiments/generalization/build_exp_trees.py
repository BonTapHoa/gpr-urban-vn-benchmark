"""Build YOLO trees (symlinks) for the generalisation experiments. usage: python scripts/build_exp_trees.py EXP.json IMAGES_DIR LABELS_DIR OUT_ROOT"""
import json, os, sys
E = json.load(open(sys.argv[1])); IM, LB, OUT = sys.argv[2:5]
test = json.load(open("configs/splits/gpr_urban_vn_split_10k_base.json"))["splits"]["test"]
names = ["pec", "upvc_air", "upvc_water", "upvc_triphase", "cable_bundle", "box_culvert", "sinkhole_void"]
for name, e in E.items():
    root = f"{OUT}/{name}"
    for sp, ids in (("train", e["train"]), ("val", e["val"]), ("test", test)):
        for kind in ("images", "labels"): os.makedirs(f"{root}/{kind}/{sp}", exist_ok=True)
        for s in ids:
            for kind, src, ext in (("images", IM, "png"), ("labels", LB, "txt")):
                d = f"{root}/{kind}/{sp}/{s}.{ext}"
                if not os.path.lexists(d): os.symlink(f"{src}/{s}.{ext}", d)
    open(f"{root}/data.yaml", "w").write(f"path: {root}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
    print(name, len(e["train"]), len(e["val"]), len(test), flush=True)
