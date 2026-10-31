# GPR-URBAN-VN benchmark: anonymous companion repository

This repository accompanies a double-blind submission. It holds the grouped split, a small loader, the label code, the
generator code and the experiment scripts behind the numbers in the paper. Authorship information is intentionally absent.

## What is here
| Path | Content |
|---|---|
| `configs/splits/gpr_urban_vn_split_10k_base.json` | Official split: 7,000 / 1,000 / 2,000 scenes (3,500 / 500 / 1,000 per frequency), grouped by scene geometry |
| `configs/dataset_physics_v2.json` | FDTD grid and source settings (400 MHz and 700 MHz) |
| `gpr_urban_vn_benchmark/` | Loader: split, YOLO labels, Tier-3 images (`pytest tests/test_loader.py`) |
| `sample_data/` | 12 test scenes (6 at 400 MHz, 6 at 700 MHz): 640x640 images and box labels |
| `src/label_refraction.py` | Box and arrival-time label code (analytic, reads scene metadata only); `tests/test_label_refraction.py` |
| `src/gpr_urban_generator.py`, `src/fdtd_fused.py` | Scene generator and the GPU FDTD solver |
| `scripts/` | Detector training (`train_detector.py`), test-set evaluation with bootstrap intervals (`eval_detectors_test.py`), field-data trees |
| `experiments/t2/` | Depth task (T2): one-velocity baseline, ResNet-18 regressor, hyperbola-fit baselines |
| `experiments/generalization/`, `physaug/`, `rebar/` | Held-out conditions, scaling, perturbations, field-data fine-tuning, joint training, label-permutation control |

## What is not here
- The full dataset (images, float32 wavefields, scene metadata) and model weights: too large for this repository. A persistent DOI will be given in the camera-ready version.
- Field data (two public sets used for transfer experiments) are not redistributed; see the paper for their sources and licences.
- The `experiments/` and `scripts/` files are the scripts **as run** on a cloud notebook VM; they contain that VM's paths (`/content/...`) and need the full dataset. They are provided for transparency, not as a one-command reproduction.

## Data conventions
- Image names: `GPRVN_v2_<freq_MHz>_<index>.png` (400 or 700). The two frequencies are independent scenario sets, not paired.
- Label line: `class cx cy w h`, normalised; x is the survey axis and y two-way travel time. Only targets whose echo is recorded in the time window have a box (17,758 of 19,000 targets).
- Classes: `pec, upvc_air, upvc_water, upvc_triphase, cable_bundle, box_culvert, sinkhole_void` (ids 0-6).
- The data are simulated (2D TM_z FDTD); see the paper for the limitations.

## Quick check
```bash
pip install -r requirements.txt
pytest tests/test_loader.py
```
