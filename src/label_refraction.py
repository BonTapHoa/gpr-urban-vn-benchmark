"""
Refraction-aware bounding-box labelling for GPR-URBAN-VN (label scheme v3).

Every box is computed from the scene description only. The B-scan image is never
read, so the labels cannot be circular with respect to the data they annotate.

Travel time to the crown of target k is the exact slowness integral along the
vertical ray above its apex through the *same* material model that the FDTD
generator paints (src/gpr_urban_generator.py): layered background, uPVC shell
(eps 3.0), stratified silt/water/air fill, concrete culvert walls (eps 8.0),
cable sheath (eps 2.5), void (eps 1.0). Nothing is fitted; overlying objects
add delay or advance simply because the ray crosses a different eps.

The box is sized from the source wavelet, not from a fixed number of ns:
  * vertical extent  = support of the wavelet envelope above `level_db`
  * horizontal extent = antenna positions where the arrival curve stays inside
    that vertical extent (finite-velocity hyperbola)
Both are analytic and are checked against full-wave references (oracle) in the
experiments (companion work, in preparation).

Presets (LabelConfig.preset) give the three schemes compared in the paper:
  uncorrected  background layers only, wavelet-based extent
  apex_centred background layers only, box centred on the apex, legacy fixed size
  proposed     ray through overlying objects + wavelet-based extent
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, replace
from functools import lru_cache
from pathlib import Path

import numpy as np

C0 = 0.299792458  # m/ns

CLASS_MAP = {
    "pec": 0,
    "upvc_air": 1,
    "upvc_water": 2,
    "upvc_triphase": 3,
    "cable_bundle": 4,
    "box_culvert": 5,
    "sinkhole_void": 6,
}

# Constants copied from the generator so the label model and the simulated
# physics cannot drift apart. Re-checked by tests/test_label_refraction.py.
EPS_UPVC_SHELL = 3.0
EPS_SILT, EPS_WATER, EPS_AIR = 24.0, 81.0, 1.0
EPS_CULVERT_WALL = 8.0
EPS_CABLE_SHEATH = 2.5
UPVC_WALL_M = 0.008
CABLE_CORE_FRAC = 0.6
DEFAULT_CULVERT_WALL_M = 0.08

# Fallbacks only when a column is missing from the manifest.
DEFAULT_ER_ASPHALT = 5.65
DEFAULT_ER_BASE = 8.30


@dataclass(frozen=True)
class LabelConfig:
    preset: str = "proposed"          # proposed | uncorrected | apex_centred
    level_db: float = -20.0           # wavelet-envelope support level
    legacy_pulse_min_ns: float = 3.8  # apex_centred only: legacy fixed height
    legacy_margin_m: float = 0.30     # apex_centred only: legacy width margin
    max_width_frac: float = 0.60      # cap on box width, fraction of scan line
    fresnel: bool = False             # average each object's delay over its first Fresnel zone
    clip_to_image: bool = True        # False reproduces the first v3 (boxes could extend past the image)
    pulse_lag_periods: float = 0.37   # received pulse peak lags the crown arrival (oracle, train scenes)
    min_w_norm: float = 0.04
    min_h_norm: float = 0.04
    # box_mode "hyperbola": box around the whole visible diffraction hyperbola (physics reference at -30 dB, fitted on train
    # scenes, loop_hyp/): half-width = a * crown depth, arm drop scaled by hyp_time_scale. "first_lobe" is the released v3.
    box_mode: str = "first_lobe"
    hyp_halfwidth_per_depth: float = 1.3
    hyp_time_scale: float = 0.8

    @property
    def use_overlying(self) -> bool:
        return self.preset == "proposed"

    @property
    def wavelet_extent(self) -> bool:
        return self.preset in ("proposed", "uncorrected")

    @property
    def centre_on_apex(self) -> bool:
        return self.preset == "apex_centred"


# --------------------------------------------------------------------------
# Source wavelet
# --------------------------------------------------------------------------
@lru_cache(maxsize=16)
def wavelet_support_ns(f_ghz: float, level_db: float = -20.0):
    """(t_before, t_after) in ns around the main-lobe peak of the Ricker source
    where the analytic envelope stays above level_db of its maximum."""
    f = f_ghz  # 1/ns
    dt = 0.002
    t = np.arange(0.0, 12.0 / f, dt)
    chi = np.sqrt(2.0) / f
    zeta = (np.pi * f) ** 2
    w = (1.0 - 2.0 * zeta * (t - chi) ** 2) * np.exp(-zeta * (t - chi) ** 2)
    n = len(w)
    spec = np.fft.fft(w)
    h = np.zeros(n)
    h[0] = 1.0
    h[1 : (n + 1) // 2] = 2.0
    if n % 2 == 0:
        h[n // 2] = 1.0
    env = np.abs(np.fft.ifft(spec * h))
    thr = env.max() * 10 ** (level_db / 20.0)
    above = np.where(env >= thr)[0]
    ip = int(np.argmin(np.abs(t - chi)))
    lo = ip
    while lo > 0 and env[lo - 1] >= thr:
        lo -= 1
    hi = ip
    while hi < n - 1 and env[hi + 1] >= thr:
        hi += 1
    return float(t[ip] - t[lo]), float(t[hi] - t[ip])


# --------------------------------------------------------------------------
# Material model of one object along a vertical ray
# --------------------------------------------------------------------------
def _fill_levels(obj, bottom, span):
    """Absolute heights where silt ends and water ends, given the interior
    spanning [bottom, bottom+span] (same semantics as the generator)."""
    silt = float(obj.get("silt_ratio", 0.0))
    water = float(obj.get("water_ratio", 0.0)) * (1.0 - silt)
    silt_top = bottom + silt * span
    water_top = silt_top + water * span
    return silt_top, water_top


def _stack(lo, hi, silt_top, water_top):
    """Split [lo, hi] into silt / water / air segments (z up)."""
    out = []
    for a, b, eps in (
        (lo, min(hi, silt_top), EPS_SILT),
        (max(lo, silt_top), min(hi, water_top), EPS_WATER),
        (max(lo, water_top), hi, EPS_AIR),
    ):
        if b > a:
            out.append((a, b, eps))
    return out


def object_segments(obj, x):
    """Return (segments, blocked) for a vertical ray at lateral position x.

    segments: list of (z_top_depth, z_bottom_depth, eps) with depth measured
    downward from the ground surface. blocked: the ray hits a perfect
    conductor, so no energy passes through (shadow zone).
    """
    kind = str(obj["type"]).lower()
    xc = float(obj["xc"])
    depth = float(obj["depth"])
    dx = abs(x - xc)

    if kind == "box_culvert":
        w = float(obj["width"])
        hgt = float(obj["height"])
        wall = float(obj.get("wall_thickness", DEFAULT_CULVERT_WALL_M))
        if dx > w / 2.0:
            return [], False
        top, bot = depth - hgt / 2.0, depth + hgt / 2.0
        if dx >= w / 2.0 - wall:
            return [(top, bot, EPS_CULVERT_WALL)], False
        segs = [(top, top + wall, EPS_CULVERT_WALL), (bot - wall, bot, EPS_CULVERT_WALL)]
        # heights measured upward from the interior floor
        floor, span = 0.0, hgt - 2.0 * wall
        silt_top, water_top = _fill_levels(obj, floor, span)
        for a, b, eps in _stack(0.0, span, silt_top, water_top):
            segs.append((bot - wall - b, bot - wall - a, eps))
        return segs, False

    r = float(obj["radius"])
    if dx >= r:
        return [], False
    half = np.sqrt(r * r - dx * dx)
    top, bot = depth - half, depth + half

    if kind == "pec":
        return [], True
    if kind == "sinkhole_void":
        return [(top, bot, EPS_AIR)], False
    if kind == "cable_bundle":
        rc = CABLE_CORE_FRAC * r
        if dx < rc:
            return [], True
        return [(top, bot, EPS_CABLE_SHEATH)], False

    # uPVC family
    ri = r - UPVC_WALL_M
    if dx >= ri:
        return [(top, bot, EPS_UPVC_SHELL)], False
    half_in = np.sqrt(ri * ri - dx * dx)
    segs = [(top, depth - half_in, EPS_UPVC_SHELL), (depth + half_in, bot, EPS_UPVC_SHELL)]
    if kind == "upvc_air":
        segs.append((depth - half_in, depth + half_in, EPS_AIR))
    elif kind == "upvc_water":
        segs.append((depth - half_in, depth + half_in, EPS_WATER))
    else:  # upvc_triphase: heights measured upward from the pipe floor
        floor, span = 0.0, 2.0 * ri
        silt_top, water_top = _fill_levels(obj, floor, span)
        lo, hi = ri - half_in, ri + half_in  # chord in floor-up coordinates
        for a, b, eps in _stack(lo, hi, silt_top, water_top):
            segs.append((depth + ri - b, depth + ri - a, eps))
    return segs, False


# --------------------------------------------------------------------------
# Travel time
# --------------------------------------------------------------------------
def _background_eps(z, h1, h2, er1, er2, er3):
    if z < h1:
        return er1
    if z < h1 + h2:
        return er2
    return er3


def target_geometry(obj):
    kind = str(obj["type"]).lower()
    depth = float(obj["depth"])
    if kind == "box_culvert":
        half_h = float(obj["height"]) / 2.0
        width = float(obj["width"])
    else:
        half_h = float(obj["radius"])
        width = 2.0 * half_h
    return max(0.01, depth - half_h), width


def slowness_integral(z_end, x, objs, skip, meta, use_overlying):
    """Integral of sqrt(eps) dz from the surface to depth z_end along x.
    Returns (value_m, blocked, n_crossed)."""
    h1 = float(meta["h_asphalt_m"])
    h2 = float(meta["h_base_m"])
    er1 = float(meta.get("er_asphalt", DEFAULT_ER_ASPHALT))
    er2 = float(meta.get("er_base", DEFAULT_ER_BASE))
    er3 = float(meta["er_subgrade"])

    segs, blocked, crossed = [], False, 0
    if use_overlying:
        for o in objs:
            if o is skip or float(o["depth"]) >= float(skip["depth"]):
                continue
            s, b = object_segments(o, x)
            if s or b:
                crossed += 1
            blocked = blocked or b
            segs.extend(s)

    cuts = {0.0, z_end, h1, h1 + h2}
    for a, b, _ in segs:
        cuts.update((a, b))
    pts = sorted(c for c in cuts if 0.0 <= c <= z_end)
    total = 0.0
    for a, b in zip(pts[:-1], pts[1:]):
        mid = 0.5 * (a + b)
        eps = None
        for sa, sb, se in segs:
            if sa <= mid < sb:
                eps = se
                break
        if eps is None:
            eps = _background_eps(mid, h1, h2, er1, er2, er3)
        total += np.sqrt(eps) * (b - a)
    return float(total), blocked, crossed


def object_delay_ns(obj, x, meta):
    """Two-way excess delay (ns) of one object for a vertical ray at x, relative to
    the layered background; (delay, blocked)."""
    segs, blocked = object_segments(obj, x)
    h1, h2 = float(meta["h_asphalt_m"]), float(meta["h_base_m"])
    er1 = float(meta.get("er_asphalt", DEFAULT_ER_ASPHALT))
    er2 = float(meta.get("er_base", DEFAULT_ER_BASE))
    er3 = float(meta["er_subgrade"])
    total = 0.0
    for a, b, eps in segs:
        pts = sorted({a, b} | {c for c in (h1, h1 + h2) if a < c < b})
        for u, v in zip(pts[:-1], pts[1:]):
            n_b = np.sqrt(_background_eps(0.5 * (u + v), h1, h2, er1, er2, er3))
            total += (np.sqrt(eps) - n_b) * (v - u)
    return 2.0 * total / C0, blocked


def fresnel_radius_m(f_ghz, v_m_ns, z_obj, z_tgt):
    """First Fresnel zone radius at the overlying object (m)."""
    lam = v_m_ns / f_ghz
    d1, d2 = max(z_obj, 1e-3), max(z_tgt - z_obj, 1e-3)
    return float(np.sqrt(lam * d1 * d2 / (d1 + d2)))


def smoothed_extra_delay(obj, objs, x, z_top, meta, f_ghz, n=21):
    """Sum over overlying objects of the delay averaged over +-R_F around x."""
    total, blocked, crossed = 0.0, False, 0
    n_sub = np.sqrt(float(meta["er_subgrade"]))
    for o in objs:
        if o is obj or float(o["depth"]) >= float(obj["depth"]):
            continue
        rf = fresnel_radius_m(f_ghz, C0 / n_sub, float(o["depth"]), z_top)
        vals = []
        for s in np.linspace(-rf, rf, n):
            d, b = object_delay_ns(o, x + s, meta)
            vals.append(d)
            if abs(s) < 1e-12 and b:
                blocked = True
        mean = float(np.mean(vals))
        if abs(mean) > 0 or any(v != 0 for v in vals):
            crossed += 1
        total += mean
    return total, blocked, crossed


def label_scene(objs, meta, profile, cfg=LabelConfig()):
    """Boxes for all targets of one scene. Returns a list of dicts."""
    f_ghz = float(profile["frequency_hz"]) / 1e9
    t_max = float(profile["requested_time_window_ns"])
    t_air = 2.0 * float(profile.get("source_height_above_ground_m", 0.05)) / C0
    chi = np.sqrt(2.0) / f_ghz

    scan0 = float(profile["scan_x_start_m"])
    scan1 = float(profile["scan_x_end_m"])
    off = float(profile["tx_rx_offset_m"])
    cmp0, cmp1 = scan0 + off / 2.0, scan1 + off / 2.0
    cmp_w = cmp1 - cmp0

    t_pre, t_post = wavelet_support_ns(round(f_ghz, 6), cfg.level_db)

    out = []
    for obj in objs:
        z_top, width_m = target_geometry(obj)
        xc = float(obj["xc"])
        if cfg.use_overlying and cfg.fresnel:
            sl, _, _ = slowness_integral(z_top, xc, objs, obj, meta, False)
            extra, blocked, crossed = smoothed_extra_delay(obj, objs, xc, z_top, meta, f_ghz)
            t_apex = t_air + chi + 2.0 * sl / C0 + extra
            v_eff = C0 * z_top / (sl + extra * C0 / 2.0)
        else:
            sl, blocked, crossed = slowness_integral(
                z_top, xc, objs, obj, meta, cfg.use_overlying
            )
            t_apex = t_air + chi + 2.0 * sl / C0
            v_eff = C0 * z_top / sl  # m/ns, mean velocity to the crown

        if not cfg.centre_on_apex:
            t_apex += cfg.pulse_lag_periods / f_ghz   # ns; one period is 1/f_GHz
        x_norm = (xc - cmp0) / cmp_w
        y_apex = t_apex / t_max
        inside = 0.0 <= x_norm <= 1.0 and 0.0 <= y_apex <= 1.0

        hyp_box = None
        if cfg.box_mode == "hyperbola":
            hw = cfg.hyp_halfwidth_per_depth * z_top
            x0m, x1m = max(cmp0, xc - hw), min(cmp1, xc + hw)
            drop = 2.0 * (np.sqrt(z_top * z_top + hw * hw) - z_top) / v_eff * cfg.hyp_time_scale
            t0m, t1m = max(0.0, t_apex - t_pre), min(t_max, t_apex + drop + t_post)
            hyp_box = (0.5 * (x0m + x1m) - cmp0) / cmp_w, max(x1m - x0m, 0.0) / cmp_w, 0.5 * (t0m + t1m) / t_max, max(t1m - t0m, 0.0) / t_max
        if cfg.wavelet_extent:
            h_ns = t_pre + t_post
            # arrival curve t(x) = t_apex + 2(sqrt(d^2+x^2)-d)/v stays in the box
            d = z_top
            reach = np.sqrt((d + v_eff * h_ns / 2.0) ** 2 - d * d)
            w_m = min(2.0 * reach + width_m, cfg.max_width_frac * cmp_w)
            t_top = t_apex - t_pre
            cy = (t_top + h_ns / 2.0) / t_max
        else:  # apex_centred: legacy fixed size, centre on the apex
            h_ns = max(1.45 / f_ghz, cfg.legacy_pulse_min_ns)
            w_m = width_m + cfg.legacy_margin_m
            cy = y_apex

        h_norm = min(1.0, max(cfg.min_h_norm, h_ns / t_max))
        w_norm = min(1.0, max(cfg.min_w_norm, w_m / cmp_w))
        cx = float(np.clip(x_norm, 0.0, 1.0))
        cy = float(np.clip(cy, 0.0, 1.0))
        if cfg.clip_to_image:   # a label cannot extend past the recorded survey line or time window
            x0, x1 = max(cx - w_norm / 2, 0.0), min(cx + w_norm / 2, 1.0)
            y0, y1 = max(cy - h_norm / 2, 0.0), min(cy + h_norm / 2, 1.0)
            cx, w_norm, cy, h_norm = 0.5 * (x0 + x1), x1 - x0, 0.5 * (y0 + y1), y1 - y0

        if hyp_box is not None:
            cx, w_norm, cy, h_norm = hyp_box
        out.append(
            {
                "type": str(obj["type"]).lower(),
                "class_id": CLASS_MAP[str(obj["type"]).lower()],
                "cx_norm": cx,
                "cy_norm": cy,
                "w_norm": float(w_norm),
                "h_norm": float(h_norm),
                "t_apex_ns": float(t_apex),
                "z_top_m": float(z_top),
                "v_eff_m_ns": float(v_eff),
                "is_inside": bool(inside),
                "shadowed": crossed > 0,
                "blocked": bool(blocked),
            }
        )
    return out


def to_yolo_lines(boxes):
    return [
        f"{b['class_id']} {b['cx_norm']:.6f} {b['cy_norm']:.6f} {b['w_norm']:.6f} {b['h_norm']:.6f}"
        for b in boxes
        if b["is_inside"]
    ]


# --------------------------------------------------------------------------
# Manifest driver
# --------------------------------------------------------------------------
def load_profiles(path):
    cfg = json.loads(Path(path).read_text())
    return {int(k): v for k, v in cfg["profiles"].items()}


def label_manifest(df, profiles, cfg=LabelConfig()):
    """Yield (sample_id, boxes) for each row of the manifest dataframe."""
    for row in df.itertuples(index=False):
        meta = {
            "h_asphalt_m": row.h_asphalt_m,
            "h_base_m": row.h_base_m,
            "er_asphalt": row.er_asphalt,
            "er_base": row.er_base,
            "er_subgrade": row.er_subgrade,
        }
        objs = json.loads(row.objects_json)
        yield row.sample_id, label_scene(objs, meta, profiles[int(row.frequency_mhz)], cfg)


def write_labels(manifest_path, profiles_path, out_dir, cfg=LabelConfig()):
    import pandas as pd

    df = pd.read_parquet(manifest_path)
    profiles = load_profiles(profiles_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n_obj = n_kept = n_shadow = n_blocked = 0
    for sid, boxes in label_manifest(df, profiles, cfg):
        lines = to_yolo_lines(boxes)
        (out / f"{sid}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))
        n_obj += len(boxes)
        n_kept += len(lines)
        n_shadow += sum(b["shadowed"] for b in boxes)
        n_blocked += sum(b["blocked"] for b in boxes)
    return {
        "preset": cfg.preset,
        "scenes": len(df),
        "objects": n_obj,
        "kept_inside": n_kept,
        "shadowed": n_shadow,
        "blocked_by_conductor": n_blocked,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--profiles", default="configs/dataset_physics_v2.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--preset", default="proposed", choices=["proposed", "uncorrected", "apex_centred"])
    a = ap.parse_args()
    print(json.dumps(write_labels(a.manifest, a.profiles, a.out, LabelConfig(preset=a.preset)), indent=2))
