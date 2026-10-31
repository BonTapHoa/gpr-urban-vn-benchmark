from pathlib import Path

from gpr_urban_vn_benchmark import CLASS_NAMES, load_sample, load_split, iter_split
from gpr_urban_vn_benchmark.loader import frequency_mhz

ROOT = Path(__file__).resolve().parents[1]


def test_split_sizes_and_disjoint():
    s = load_split(ROOT / "configs/splits/gpr_urban_vn_split_10k_base.json")
    assert [len(s[k]) for k in ("train", "validation", "test")] == [7000, 1000, 2000]
    assert not set(s["train"]) & set(s["test"]) and not set(s["validation"]) & set(s["test"])


def test_sample_images_and_labels():
    s = load_split(ROOT / "configs/splits/gpr_urban_vn_split_10k_base.json")
    rows = list(iter_split(s, "test", ROOT / "sample_data"))
    assert len(rows) == 12
    for sid, img, boxes in rows:
        assert img.shape == (640, 640) and frequency_mhz(sid) in (400, 700)
        for c, cx, cy, w, h in boxes:
            assert 0 <= c < len(CLASS_NAMES) and 0 <= cx <= 1 and 0 <= cy <= 1 and 0 < w <= 1 and 0 < h <= 1
