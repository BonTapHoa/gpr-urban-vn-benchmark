"""Fused-kernel variant of GPRUrbanVNGenerator.solve_super_batch (HORIPML only).

The reference solver runs ~25 separate elementwise kernels per time step on
(B, traces, ny, nx) tensors, so it is limited by memory bandwidth rather than
arithmetic. Here the H update and the interior E update are each compiled into a
single kernel. The thin PML strips, the receiver gather and the source injection
are unchanged, so results agree with the reference to float32 round-off.
"""
import numpy as np
import torch

from src.gpr_urban_generator import (
    apply_4cell_interface_smoothing, compute_boundary_face_permittivity,
    prepare_inhomogeneous_horipml_tensors, apply_horipml_h_corrections, apply_horipml_e_corrections,
)


def _h_update(Ez, Hx, Hy, c_hx, c_hy):
    Hx[:, :, :-1, :] -= c_hx * (Ez[:, :, 1:, :] - Ez[:, :, :-1, :])
    Hy[:, :, :, :-1] += c_hy * (Ez[:, :, :, 1:] - Ez[:, :, :, :-1])


def _e_update(Ez, Hx, Hy, Ca, Cb):
    Ez[:, :, 1:-1, 1:-1] = Ca[:, :, 1:-1, 1:-1] * Ez[:, :, 1:-1, 1:-1] + Cb[:, :, 1:-1, 1:-1] * (
        (Hy[:, :, 1:-1, 1:-1] - Hy[:, :, 1:-1, :-2]) - (Hx[:, :, 1:-1, 1:-1] - Hx[:, :, :-2, 1:-1]))


import torch._dynamo
torch._dynamo.config.cache_size_limit = 256      # many crop shapes: past the default limit of 8 PyTorch silently falls back to eager
_compiled = {}


def _get(name, compile_):
    key = (name, bool(compile_))          # the cache must distinguish compiled from eager kernels
    if key not in _compiled:
        fn = {"h": _h_update, "e": _e_update}[name]
        _compiled[key] = torch.compile(fn, dynamic=False) if compile_ else fn
    return _compiled[key]


@torch.no_grad()
def solve_super_batch_fused(gen, eps_r, sigma, pec_mask, num_traces=None, num_steps=None, compile_=True, use_graph=True,
                            field_dtype=torch.float32, trace_stride=1):
    assert gen.pml_formulation == "HORIPML"
    dev = gen.device
    eps_r, sigma, pec_mask = eps_r.to(dev), sigma.to(dev), pec_mask.to(dev)
    batch_b = eps_r.shape[0]
    n_full = num_traces if num_traces is not None else gen.num_traces
    steps = gen.num_steps if num_steps is None else num_steps

    ant_r = int(round((gen.ground_y + gen.source_height_above_ground) / gen.dy))
    xs = np.linspace(gen.scan_x_start, gen.scan_x_end, n_full)[::trace_stride]   # every k-th antenna position
    n_traces = len(xs)
    tx_c = torch.tensor([int(round(x / gen.dx)) for x in xs], dtype=torch.long, device=dev)
    rx_c = torch.tensor([int(round((x + gen.ant_offset) / gen.dx)) for x in xs], dtype=torch.long, device=dev)

    if gen.interface_smoothing:
        eps_r, sigma = apply_4cell_interface_smoothing(eps_r, sigma)
    eps = eps_r * gen.EPS_0
    term = (sigma * gen.dt) / (2.0 * eps)
    Ca = (1.0 - term) / (1.0 + term)
    Cb = gen.dt / (eps * gen.dx) / (1.0 + term)
    Ca[pec_mask] = 0.0
    Cb[pec_mask] = 0.0
    Ca, Cb = Ca.unsqueeze(1).contiguous(), Cb.unsqueeze(1).contiguous()
    c_hx = gen.dt / (gen.MU_0 * gen.dy)
    c_hy = gen.dt / (gen.MU_0 * gen.dx)

    er_faces = compute_boundary_face_permittivity(eps_r, pml_w=gen.pml_w)
    coeffs = prepare_inhomogeneous_horipml_tensors(
        pml_cells=gen.pml_w, d=gen.dx, dt=gen.dt, er_faces=er_faces, device=dev, dtype=torch.float32, is_4d=True)

    shp = (batch_b, n_traces, gen.ny, gen.nx)
    Ez, Hx, Hy = (torch.zeros(shp, device=dev, dtype=field_dtype) for _ in range(3))
    w = gen.pml_w
    psi_Hx_b = torch.zeros((batch_b, n_traces, w, gen.nx), device=dev)
    psi_Hx_t = torch.zeros_like(psi_Hx_b)
    psi_Hy_l = torch.zeros((batch_b, n_traces, gen.ny, w), device=dev)
    psi_Hy_r = torch.zeros_like(psi_Hy_l)
    psi_Ez_xl = torch.zeros((batch_b, n_traces, gen.ny, w), device=dev)
    psi_Ez_xr = torch.zeros_like(psi_Ez_xl)
    psi_Ez_yb = torch.zeros((batch_b, n_traces, w, gen.nx), device=dev)
    psi_Ez_yt = torch.zeros_like(psi_Ez_yb)
    bscan = torch.zeros((steps, batch_b, n_traces), device=dev)

    b_mesh, tr_mesh = torch.meshgrid(torch.arange(batch_b, device=dev), torch.arange(n_traces, device=dev), indexing="ij")
    tx_2d, rx_2d = tx_c.unsqueeze(0).expand(batch_b, -1), rx_c.unsqueeze(0).expand(batch_b, -1)
    t = (torch.arange(gen.num_steps, device=dev, dtype=torch.float32) + 0.5) * gen.dt
    chi = np.sqrt(2.0) / gen.f_center
    zeta = (np.pi * gen.f_center) ** 2
    ricker = (1.0 - 2.0 * zeta * (t - chi) ** 2) * torch.exp(-zeta * (t - chi) ** 2)
    eps_tx = eps_r[:, 0, 0].new_tensor(0.0)  # placeholder to keep dtype/device
    eps_r_tx = float(eps_r[..., ant_r, tx_c].float().mean().item())
    src_scale = gen.dt / (gen.EPS_0 * eps_r_tx * gen.dx * gen.dy)

    h_up, e_up = _get("h", compile_), _get("e", compile_)
    n_t = torch.zeros(1, dtype=torch.long, device=dev)          # step counter lives on the device
    ant_idx = torch.full_like(rx_2d, ant_r)

    def step():
        bscan.index_copy_(0, n_t, Ez[b_mesh, tr_mesh, ant_r, rx_2d].float().unsqueeze(0))
        h_up(Ez, Hx, Hy, c_hx, c_hy)
        apply_horipml_h_corrections(Hx, Hy, Ez, psi_Hx_b, psi_Hx_t, psi_Hy_l, psi_Hy_r,
                                    coeffs, c_hx, c_hy, gen.pml_w, gen.top_pml_enabled)
        e_up(Ez, Hx, Hy, Ca, Cb)
        apply_horipml_e_corrections(Ez, Hx, Hy, psi_Ez_xl, psi_Ez_xr, psi_Ez_yb, psi_Ez_yt,
                                    coeffs, Cb, gen.pml_w, gen.top_pml_enabled)
        Ez[b_mesh, tr_mesh, ant_r, tx_2d] -= src_scale * ricker.index_select(0, n_t)
        n_t.add_(1)

    def reset():
        for a in (Ez, Hx, Hy, psi_Hx_b, psi_Hx_t, psi_Hy_l, psi_Hy_r, psi_Ez_xl, psi_Ez_xr, psi_Ez_yb, psi_Ez_yt, bscan):
            a.zero_()
        n_t.zero_()

    if use_graph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):
                step()
        torch.cuda.current_stream().wait_stream(side)
        reset()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            step()
        reset()
        for _ in range(steps):
            graph.replay()
    else:
        for _ in range(steps):
            step()
    torch.cuda.synchronize()
    return bscan.cpu().numpy()
