"""Minimal loader for the Tier-3 release: 640x640 grey-scale B-scans and YOLO box labels.

Image names look like ``GPRVN_v2_<freq_MHz>_<index>.png`` (freq is 400 or 700). A label file has one
line per target that has a box: ``class cx cy w h`` (normalised, x = survey axis, y = two-way time).
Targets whose echo is not recorded in the window have no box (17,758 of the 19,000 targets have one).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Tuple

import numpy as np
from PIL import Image

CLASS_NAMES = ["pec", "upvc_air", "upvc_water", "upvc_triphase", "cable_bundle", "box_culvert", "sinkhole_void"]

Box = Tuple[int, float, float, float, float]


def load_split(path: str | Path) -> Dict[str, List[str]]:
    """Return ``{"train": [...], "validation": [...], "test": [...]}`` of scene ids (grouped split, no leakage)."""
    return {k: list(v) for k, v in json.load(open(path))["splits"].items()}


def load_labels(path: str | Path) -> List[Box]:
    """Parse one YOLO label file; an empty file gives an empty list."""
    boxes: List[Box] = []
    for line in Path(path).read_text().splitlines():
        parts = line.split()
        if len(parts) == 5:
            boxes.append((int(parts[0]), *map(float, parts[1:])))
    return boxes


def frequency_mhz(scene_id: str) -> int:
    """Antenna centre frequency (MHz) encoded in the scene id."""
    return int(scene_id.split("_")[2])


def load_sample(scene_id: str, root: str | Path = "sample_data") -> Tuple[np.ndarray, List[Box]]:
    """Return the uint8 image (H, W) and its boxes for ``scene_id`` under ``root/{images,labels}``."""
    root = Path(root)
    img = np.asarray(Image.open(root / "images" / f"{scene_id}.png").convert("L"))
    return img, load_labels(root / "labels" / f"{scene_id}.txt")


def iter_split(split: Dict[str, List[str]], name: str, root: str | Path) -> Iterator[Tuple[str, np.ndarray, List[Box]]]:
    """Iterate over the scenes of one split that are present under ``root``."""
    root = Path(root)
    for sid in split[name]:
        if (root / "images" / f"{sid}.png").exists():
            yield (sid, *load_sample(sid, root))
