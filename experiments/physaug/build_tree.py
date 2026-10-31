"""YOLO tree (symlinks) from the official split. usage: python build_tree.py SPLIT_JSON IMAGES_DIR LABELS_DIR OUT_ROOT"""
import json, os, sys
J, IM, LB, OUT = sys.argv[1:5]; S = json.load(open(J))["splits"]; names = ["pec", "upvc_air", "upvc_water", "upvc_triphase", "cable_bundle", "box_culvert", "sinkhole_void"]
for sp, key in (("train", "train"), ("val", "validation"), ("test", "test")):
    for kind in ("images", "labels"): os.makedirs(f"{OUT}/{kind}/{sp}", exist_ok=True)
    for s in S[key]:
        for kind, src, ext in (("images", IM, "png"), ("labels", LB, "txt")):
            d = f"{OUT}/{kind}/{sp}/{s}.{ext}"
            if not os.path.lexists(d): os.symlink(f"{src}/{s}.{ext}", d)
open(f"{OUT}/data.yaml", "w").write(f"path: {OUT}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n" + "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
print("tree", OUT, {k: len(v) for k, v in S.items()})
