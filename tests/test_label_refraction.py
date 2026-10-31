import json
import numpy as np
import pytest

from src.label_refraction import (
    C0, LabelConfig, label_scene, object_segments, slowness_integral,
    to_yolo_lines, wavelet_support_ns, load_profiles,
)

PROFILES = load_profiles("configs/dataset_physics_v2.json")
P400 = PROFILES[400]
META = dict(h_asphalt_m=0.10, h_base_m=0.30, er_asphalt=5.0, er_base=8.0, er_subgrade=16.0)


def pipe(kind, xc, depth, r=0.1, **kw):
    return dict(type=kind, xc=xc, depth=depth, radius=r, water_ratio=kw.get("water_ratio", 0.0),
                silt_ratio=kw.get("silt_ratio", 0.0))


def test_no_overlying_object_equals_layered_formula():
    t = pipe("upvc_air", 1.5, 1.0)
    z_top = 1.0 - 0.1
    sl, blocked, crossed = slowness_integral(z_top, 1.5, [t], t, META, True)
    ref = 0.10 * np.sqrt(5.0) + 0.30 * np.sqrt(8.0) + (z_top - 0.40) * np.sqrt(16.0)
    assert sl == pytest.approx(ref, rel=1e-12)
    assert crossed == 0 and not blocked


def test_presets_agree_without_overlying_objects():
    t = pipe("upvc_water", 1.5, 1.0, water_ratio=1.0)
    a = label_scene([t], META, P400, LabelConfig(preset="proposed"))[0]
    b = label_scene([t], META, P400, LabelConfig(preset="uncorrected"))[0]
    assert a == b


def test_water_pipe_above_delays_deeper_target():
    top = pipe("upvc_water", 1.5, 0.7, r=0.10)
    low = pipe("upvc_air", 1.5, 1.2, r=0.10)
    prop = label_scene([top, low], META, P400, LabelConfig(preset="proposed"))[1]
    unc = label_scene([top, low], META, P400, LabelConfig(preset="uncorrected"))[1]
    # vertical ray through centre: shell 2*0.008 at eps 3, water 2*0.092 at eps 81
    expect = 2 * (0.016 * (np.sqrt(3) - 4) + 0.184 * (9 - 4)) / C0
    assert prop["t_apex_ns"] - unc["t_apex_ns"] == pytest.approx(expect, rel=1e-9)
    assert prop["shadowed"] and not unc["shadowed"]


def test_air_pipe_above_advances_deeper_target():
    top = pipe("upvc_air", 1.5, 0.7)
    low = pipe("upvc_air", 1.5, 1.2)
    prop = label_scene([top, low], META, P400, LabelConfig())[1]
    unc = label_scene([top, low], META, P400, LabelConfig(preset="uncorrected"))[1]
    assert prop["t_apex_ns"] < unc["t_apex_ns"]


def test_object_off_the_ray_has_no_effect():
    top = pipe("upvc_water", 2.3, 0.7, water_ratio=1.0)
    low = pipe("upvc_air", 1.5, 1.2)
    a = label_scene([top, low], META, P400, LabelConfig())[1]
    b = label_scene([top, low], META, P400, LabelConfig(preset="uncorrected"))[1]
    assert a["t_apex_ns"] == b["t_apex_ns"] and not a["shadowed"]


def test_pec_above_blocks():
    top = pipe("pec", 1.5, 0.7)
    low = pipe("upvc_air", 1.5, 1.2)
    assert label_scene([top, low], META, P400)[1]["blocked"]


def test_triphase_layers_sum_to_interior():
    o = pipe("upvc_triphase", 1.0, 1.0, r=0.1, water_ratio=0.5, silt_ratio=0.25)
    segs, _ = object_segments(o, 1.0)
    inner = sum(b - a for a, b, e in segs if e != 3.0)
    assert inner == pytest.approx(2 * (0.1 - 0.008), rel=1e-12)
    epsv = {e: sum(b - a for a, b, ee in segs if ee == e) for e in (24.0, 81.0, 1.0)}
    d = 2 * 0.092
    assert epsv[24.0] == pytest.approx(0.25 * d)
    assert epsv[81.0] == pytest.approx(0.5 * 0.75 * d)


def test_culvert_vertical_chord_is_full_height():
    c = dict(type="box_culvert", xc=1.0, depth=0.9, width=0.6, height=0.5, wall_thickness=0.08,
             water_ratio=0.3, silt_ratio=0.1)
    segs, blocked = object_segments(c, 1.0)
    assert sum(b - a for a, b, _ in segs) == pytest.approx(0.5, rel=1e-12)
    assert not blocked


def test_wavelet_support_is_positive_and_shrinks_with_frequency():
    a = wavelet_support_ns(0.4, -20.0)
    b = wavelet_support_ns(0.7, -20.0)
    assert all(v > 0 for v in a + b)
    assert sum(b) < sum(a)


def test_yolo_lines_normalised():
    t = pipe("upvc_air", 1.5, 1.0)
    for ln in to_yolo_lines(label_scene([t], META, P400)):
        v = ln.split()
        assert int(v[0]) == 3 - 2 and all(0 <= float(x) <= 1 for x in v[1:])


def test_fresnel_off_by_default_and_matches_chord_at_zero_radius():
    top = pipe("upvc_water", 1.5, 0.7, water_ratio=1.0)
    low = pipe("upvc_air", 1.5, 1.2)
    a = label_scene([top, low], META, P400, LabelConfig())[1]
    b = label_scene([top, low], META, P400, LabelConfig(fresnel=True))[1]
    assert a["t_apex_ns"] != b["t_apex_ns"]          # smoothing changes the answer
    assert b["t_apex_ns"] < a["t_apex_ns"]           # centre delay is the maximum, so the mean is smaller
    far = pipe("upvc_water", 2.5, 0.7, water_ratio=1.0)
    c = label_scene([far, low], META, P400, LabelConfig(fresnel=True))[1]
    assert c["t_apex_ns"] == label_scene([far, low], META, P400, LabelConfig(preset="uncorrected"))[1]["t_apex_ns"]


def test_object_delay_matches_closed_form():
    from src.label_refraction import object_delay_ns
    o = pipe("upvc_water", 1.5, 0.7, r=0.10, water_ratio=1.0)
    meta = dict(META, er_subgrade=16.0)
    d, _ = object_delay_ns(o, 1.5, meta)
    assert d == pytest.approx(2 * (0.016 * (np.sqrt(3) - 4) + 0.184 * (9 - 4)) / C0, rel=1e-9)


def test_pulse_lag_shifts_peak_by_fraction_of_period_and_not_apex_centred():
    t = pipe("upvc_air", 1.5, 1.0)
    no_lag = label_scene([t], META, P400, LabelConfig(pulse_lag_periods=0.0))[0]["t_apex_ns"]
    lag = label_scene([t], META, P400, LabelConfig())[0]["t_apex_ns"]
    assert lag - no_lag == pytest.approx(0.37 / 0.4, rel=1e-9)      # 400 MHz: period 2.5 ns
    a0 = label_scene([t], META, P400, LabelConfig(preset="apex_centred", pulse_lag_periods=0.0))[0]["t_apex_ns"]
    a1 = label_scene([t], META, P400, LabelConfig(preset="apex_centred"))[0]["t_apex_ns"]
    assert a0 == a1


def test_boxes_stay_inside_the_image_and_unclipped_flag_reproduces_old_behaviour():
    t = pipe("upvc_air", 0.35, 1.0)   # close to the left edge of the survey line
    for preset in ("uncorrected", "proposed", "apex_centred"):
        b = label_scene([t], META, P400, LabelConfig(preset=preset))[0]
        assert b["cx_norm"] - b["w_norm"] / 2 >= -1e-9 and b["cx_norm"] + b["w_norm"] / 2 <= 1 + 1e-9
        assert b["cy_norm"] - b["h_norm"] / 2 >= -1e-9 and b["cy_norm"] + b["h_norm"] / 2 <= 1 + 1e-9
    raw = label_scene([t], META, P400, LabelConfig(clip_to_image=False))[0]
    assert raw["cx_norm"] - raw["w_norm"] / 2 < 0


def test_hyperbola_mode_is_opt_in_and_default_is_unchanged():
    t = pipe("upvc_air", 1.5, 1.0)
    assert LabelConfig().box_mode == "first_lobe"
    a = label_scene([t], META, P400, LabelConfig(preset="uncorrected"))[0]
    b = label_scene([t], META, P400, LabelConfig(preset="uncorrected", box_mode="first_lobe"))[0]
    assert a == b


def test_hyperbola_box_matches_closed_form_and_contains_the_first_lobe_in_time():
    t = pipe("upvc_air", 1.5, 0.8)
    cfg_h = LabelConfig(preset="uncorrected", box_mode="hyperbola")
    h = label_scene([t], META, P400, cfg_h)[0]
    f = label_scene([t], META, P400, LabelConfig(preset="uncorrected"))[0]
    tmax = P400["requested_time_window_ns"]; off = P400["tx_rx_offset_m"]
    cmp_w = P400["scan_x_end_m"] - P400["scan_x_start_m"]
    z = h["z_top_m"]; hw = cfg_h.hyp_halfwidth_per_depth * z
    assert h["w_norm"] * cmp_w == pytest.approx(2 * hw, rel=1e-9)          # unclipped, central target
    drop = 2 * (np.sqrt(z * z + hw * hw) - z) / h["v_eff_m_ns"] * cfg_h.hyp_time_scale
    t_pre, t_post = wavelet_support_ns(round(P400["frequency_hz"] / 1e9, 6), -20.0)
    assert h["h_norm"] * tmax == pytest.approx(t_pre + drop + t_post, rel=1e-9)
    top_h, top_f = (h["cy_norm"] - h["h_norm"] / 2) * tmax, (f["cy_norm"] - f["h_norm"] / 2) * tmax
    bot_h, bot_f = (h["cy_norm"] + h["h_norm"] / 2) * tmax, (f["cy_norm"] + f["h_norm"] / 2) * tmax
    assert top_h == pytest.approx(top_f, abs=1e-9) and bot_h >= bot_f - 1e-9
    assert h["t_apex_ns"] == f["t_apex_ns"] and h["is_inside"] == f["is_inside"]


def test_hyperbola_boxes_stay_inside_the_image_for_deep_and_edge_targets():
    culvert = dict(type="box_culvert", xc=0.15, depth=1.3, width=0.6, height=0.4, wall_thickness=0.08, water_ratio=0.0, silt_ratio=0.0)
    objs = [culvert, pipe("upvc_air", 3.0, 1.2), pipe("cable_bundle", 1.5, 0.3, r=0.03)]
    for b in label_scene(objs, META, P400, LabelConfig(preset="uncorrected", box_mode="hyperbola")):
        assert -1e-9 <= b["cx_norm"] - b["w_norm"] / 2 and b["cx_norm"] + b["w_norm"] / 2 <= 1 + 1e-9
        assert -1e-9 <= b["cy_norm"] - b["h_norm"] / 2 and b["cy_norm"] + b["h_norm"] / 2 <= 1 + 1e-9
