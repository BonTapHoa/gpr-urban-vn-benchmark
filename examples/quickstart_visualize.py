"""Quickstart example: load sample radargrams and visualize bounding box annotations.
Usage:
    python examples/quickstart_visualize.py
"""
from pathlib import Path
import sys
import numpy as np
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from gpr_urban_vn_benchmark import CLASS_NAMES, load_split, iter_split

def main():
    split_path = ROOT / "configs/splits/gpr_urban_vn_split_10k_base.json"
    data_dir = ROOT / "sample_data"
    out_dir = ROOT / "sample_data/visualized"
    out_dir.mkdir(parents=True, exist_ok=True)

    split = load_split(split_path)
    samples = list(iter_split(split, "test", data_dir))
    print(f"Loaded {len(samples)} sample radargrams from {data_dir}")

    # Visualize the first 4 samples
    for sid, img_arr, boxes in samples[:4]:
        # Convert single-channel greyscale to RGB for coloured box overlays
        img = Image.fromarray(img_arr).convert("RGB")
        draw = ImageDraw.Draw(img)
        w, h = img.size

        for cls_id, cx, cy, bw, bh in boxes:
            x1 = int((cx - bw / 2) * w)
            y1 = int((cy - bh / 2) * h)
            x2 = int((cx + bw / 2) * w)
            y2 = int((cy + bh / 2) * h)
            cls_name = CLASS_NAMES[cls_id]

            # Draw rectangle (red) and label tag
            draw.rectangle([x1, y1, x2, y2], outline="#D55E00", width=2)
            draw.text((x1 + 3, max(0, y1 - 12)), cls_name, fill="#D55E00")

        out_path = out_dir / f"{sid}_annotated.png"
        img.save(out_path)
        print(f"  [+] Saved: {out_path} ({len(boxes)} targets)")

    print(f"\nAll sample visualizations saved to: {out_dir}")

if __name__ == "__main__":
    main()
