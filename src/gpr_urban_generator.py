"""
================================================================================
GPR-URBAN-VN DATASET GENERATOR ENGINE (PyTorch Batch GPU FDTD)
Authors: Dinh-Nguyen Tu, Hoang-Thien-Nga Nguyen, Hoai-Trung Dang
Platform: PyTorch 2.x + CUDA (Tesla T4 or higher)
Standard: 400 MHz & 700 MHz Urban Vietnam Pavement & Multiphase Utilities
================================================================================
"""

import os
import json
import time
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

try:
    from src.horipml_engine import (
        compute_horipml_coefficients,
        prepare_horipml_tensors,
        compute_boundary_face_permittivity,
        prepare_inhomogeneous_horipml_tensors,
        apply_horipml_h_corrections,
        apply_horipml_e_corrections,
        apply_4cell_interface_smoothing,
    )
except ImportError:
    from horipml_engine import (
        compute_horipml_coefficients,
        prepare_horipml_tensors,
        compute_boundary_face_permittivity,
        prepare_inhomogeneous_horipml_tensors,
        apply_horipml_h_corrections,
        apply_horipml_e_corrections,
        apply_4cell_interface_smoothing,
    )


class GPRUrbanVNGenerator:
    def __init__(
        self,
        domain_x=2.2,       # Total survey length (meters)
        domain_y=1.3,       # Total domain depth (meters)
        dx=0.005,           # Spatial grid resolution (5mm)
        dy=0.005,
        dt=1.179327e-11,    # Time step (11.79 ps, Courant stable)
        num_steps=2200,     # Time steps (~26 ns window)
        f_center=700e6,     # Central frequency (Hz)
        pml_w=10,           # PML absorbing boundary thickness
        ant_offset=0.10,    # Bistatic Tx-Rx offset (meters)
        scan_x_start=0.35,  # First trace position
        scan_x_end=1.85,    # Last trace position
        num_traces=50,      # Number of A-scans in B-scan
        ground_y=1.00,      # Ground surface height in model (meters)
        device='cuda' if torch.cuda.is_available() else 'cpu',
        source_height_above_ground=.010,
        top_pml_enabled=True,
        pml_formulation="HORIPML",
        interface_smoothing=False,
    ):
        if pml_formulation not in ("HORIPML", "CPML"):
            raise ValueError(f"Unsupported pml_formulation: {pml_formulation}. Must be 'HORIPML' or 'CPML'.")
        if not isinstance(interface_smoothing, bool):
            raise TypeError(f"interface_smoothing must be a bool, got {type(interface_smoothing)}")

        self.domain_x = domain_x
        self.domain_y = domain_y
        self.dx = dx
        self.dy = dy
        self.nx = int(round(domain_x / dx))
        self.ny = int(round(domain_y / dy))
        self.dt = dt
        self.num_steps = num_steps
        self.f_center = f_center
        self.pml_w = pml_w
        self.ant_offset = ant_offset
        self.scan_x_start = scan_x_start
        self.scan_x_end = scan_x_end
        self.num_traces = num_traces
        self.ground_y = ground_y
        self.device = device
        self.source_height_above_ground = source_height_above_ground
        self.top_pml_enabled = top_pml_enabled
        self.pml_formulation = pml_formulation
        self.interface_smoothing = interface_smoothing

        # Physical constants
        self.C0 = 299792458.0
        self.MU_0 = 4.0 * np.pi * 1e-7
        self.EPS_0 = 8.8541878128e-12

        # Coordinate meshes
        self.x_coords = np.arange(self.nx) * self.dx + self.dx / 2.0
        self.y_coords = np.arange(self.ny) * self.dy + self.dy / 2.0
        self.X, self.Y = np.meshgrid(self.x_coords, self.y_coords)

    def build_pavement_scene(
        self,
        asphalt_thickness=0.09,   # 9 cm AC
        asphalt_er=5.0,
        asphalt_sigma=0.001,
        base_thickness=0.25,      # 25 cm Base
        base_er=7.0,
        base_sigma=0.004,
        subgrade_er=16.0,         # Tropical clay subgrade
        subgrade_sigma=0.035,
        add_gravel=True,
        num_gravels=40
    ):
        """Constructs realistic 3-layer urban pavement with rough interfaces and clutter."""
        eps_r = np.ones((self.ny, self.nx), dtype=np.float32)
        sigma = np.zeros((self.ny, self.nx), dtype=np.float32)

        # Ground surface elevation with gentle roughness
        surf_y = self.ground_y + 0.002 * np.sin(2.0 * np.pi * self.X[0, :] / 0.3)
        asphalt_bottom = surf_y - asphalt_thickness + 0.005 * np.sin(2.0 * np.pi * self.X[0, :] / 0.5)
        base_bottom = asphalt_bottom - base_thickness + 0.008 * np.sin(2.0 * np.pi * self.X[0, :] / 0.7)

        # Layer 3: Subgrade (Bottom up to base_bottom)
        mask_subgrade = self.Y <= base_bottom[np.newaxis, :]
        eps_r[mask_subgrade] = subgrade_er
        sigma[mask_subgrade] = subgrade_sigma

        # Layer 2: Crushed Stone Base
        mask_base = (self.Y > base_bottom[np.newaxis, :]) & (self.Y <= asphalt_bottom[np.newaxis, :])
        eps_r[mask_base] = base_er
        sigma[mask_base] = base_sigma

        # Gravel clutter in base layer
        if add_gravel:
            np.random.seed(42)
            for _ in range(num_gravels):
                gx = np.random.uniform(self.scan_x_start - 0.2, self.scan_x_end + 0.2)
                gy = np.random.uniform(self.ground_y - asphalt_thickness - base_thickness + 0.03, self.ground_y - asphalt_thickness - 0.03)
                gr = np.random.uniform(0.015, 0.03)
                g_mask = ((self.X - gx)**2 + (self.Y - gy)**2) <= gr**2
                eps_r[g_mask & mask_base] = np.random.uniform(8.0, 9.5)

        # Layer 1: Asphalt Concrete
        mask_asphalt = (self.Y > asphalt_bottom[np.newaxis, :]) & (self.Y <= surf_y[np.newaxis, :])
        eps_r[mask_asphalt] = asphalt_er
        sigma[mask_asphalt] = asphalt_sigma

        # Above ground: Air (eps_r = 1.0, sigma = 0.0)
        mask_air = self.Y > surf_y[np.newaxis, :]
        eps_r[mask_air] = 1.0
        sigma[mask_air] = 0.0

        return eps_r, sigma

    def insert_pec_pipe(self, eps_r, sigma, pec_mask, x_center, depth, radius):
        """Inserts a perfect electric conductor (metal pipe)."""
        y_center = self.ground_y - depth
        pipe_mask = ((self.X - x_center)**2 + (self.Y - y_center)**2) <= radius**2
        pec_mask[pipe_mask] = True
        return pec_mask

    def insert_upvc_pipe(
        self,
        eps_r,
        sigma,
        x_center,
        depth,
        outer_radius,
        wall_thickness=0.008,
        filling_mode="air", # "air", "water", "tri_phase"
        silt_ratio=0.25,    # fraction of pipe height from bottom
        water_ratio=0.45    # fraction of pipe height above silt
    ):
        """Inserts a uPVC pipe with customizable multiphase contents."""
        y_center = self.ground_y - depth
        inner_radius = outer_radius - wall_thickness

        dist_sq = (self.X - x_center)**2 + (self.Y - y_center)**2
        shell_mask = (dist_sq <= outer_radius**2) & (dist_sq > inner_radius**2)
        inner_mask = dist_sq <= inner_radius**2

        # uPVC Shell: er = 3.0, sigma = 0.0001
        eps_r[shell_mask] = 3.0
        sigma[shell_mask] = 0.0001

        # Internal contents
        if filling_mode == "air":
            eps_r[inner_mask] = 1.0
            sigma[inner_mask] = 0.0

        elif filling_mode == "water":
            eps_r[inner_mask] = 81.0
            sigma[inner_mask] = 0.015

        elif filling_mode == "tri_phase":
            # Total height ranges from y_center - inner_radius to y_center + inner_radius
            pipe_bottom = y_center - inner_radius
            pipe_diam = 2.0 * inner_radius

            silt_top = pipe_bottom + silt_ratio * pipe_diam
            water_top = silt_top + water_ratio * pipe_diam

            silt_mask = inner_mask & (self.Y <= silt_top)
            water_mask = inner_mask & (self.Y > silt_top) & (self.Y <= water_top)
            air_mask = inner_mask & (self.Y > water_top)

            # Silt layer: er = 24.0, sigma = 0.04
            eps_r[silt_mask] = 24.0
            sigma[silt_mask] = 0.040

            # Water layer: er = 81.0, sigma = 0.015
            eps_r[water_mask] = 81.0
            sigma[water_mask] = 0.015

            # Free air: er = 1.0, sigma = 0.0
            eps_r[air_mask] = 1.0
            sigma[air_mask] = 0.0

    def insert_cable_bundle(self, eps_r, sigma, pec_mask, x_center, depth, radius=0.03):
        """Inserts a small telecom/power cable bundle."""
        y_center = self.ground_y - depth
        dist_sq = (self.X - x_center)**2 + (self.Y - y_center)**2
        sheath_mask = (dist_sq <= radius**2) & (dist_sq > (radius*0.6)**2)
        core_mask = dist_sq <= (radius*0.6)**2

        eps_r[sheath_mask] = 2.5
        sigma[sheath_mask] = 0.0001
        pec_mask[core_mask] = True

    def insert_concrete_box_culvert(
        self,
        eps_r,
        sigma,
        x_center,
        depth,
        width=0.60,
        height=0.50,
        wall_thickness=0.08,
        water_height=0.15
    ):
        """Inserts a concrete box culvert with water and air inside."""
        y_top = self.ground_y - depth
        y_bottom = y_top - height
        x_left = x_center - width / 2.0
        x_right = x_center + width / 2.0

        outer_mask = (self.X >= x_left) & (self.X <= x_right) & (self.Y >= y_bottom) & (self.Y <= y_top)

        in_left = x_left + wall_thickness
        in_right = x_right - wall_thickness
        in_bottom = y_bottom + wall_thickness
        in_top = y_top - wall_thickness
        inner_mask = (self.X >= in_left) & (self.X <= in_right) & (self.Y >= in_bottom) & (self.Y <= in_top)

        wall_mask = outer_mask & (~inner_mask)
        # Concrete walls: er = 8.0, sigma = 0.01
        eps_r[wall_mask] = 8.0
        sigma[wall_mask] = 0.010

        # Interior: bottom water, top air
        water_level = in_bottom + water_height
        water_mask = inner_mask & (self.Y <= water_level)
        air_mask = inner_mask & (self.Y > water_level)

        eps_r[water_mask] = 81.0
        sigma[water_mask] = 0.015
        eps_r[air_mask] = 1.0
        sigma[air_mask] = 0.0

    def build_pavement_scenes_gpu(
        self,
        batch_b=1,
        asphalt_h_range=(0.06, 0.12),
        asphalt_er_range=(4.5, 6.0),
        asphalt_sig_range=(0.0005, 0.002),
        base_h_range=(0.18, 0.32),
        base_er_range=(6.0, 8.5),
        base_sig_range=(0.002, 0.008),
        subgrade_er_range=(10.0, 24.0),
        subgrade_sig_range=(0.015, 0.085),
        add_gravel=True,
        num_gravels=30,
        seed=None
    ):
        """
        100% Vectorized GPU Scene Synthesis.
        Synthesizes batch_b distinct geological pavement profiles directly in VRAM.
        Returns: eps_r: (batch_b, ny, nx), sigma: (batch_b, ny, nx), pec_mask: (batch_b, ny, nx)
        """
        if seed is not None:
            torch.manual_seed(seed)
            
        dev = self.device
        
        # 1. Coordinate grid on GPU: (1, ny, nx)
        y_1d = torch.arange(self.ny, device=dev, dtype=torch.float32) * self.dy + self.dy / 2.0
        x_1d = torch.arange(self.nx, device=dev, dtype=torch.float32) * self.dx + self.dx / 2.0
        Y, X = torch.meshgrid(y_1d, x_1d, indexing='ij')
        X = X.unsqueeze(0)  # (1, ny, nx)
        Y = Y.unsqueeze(0)  # (1, ny, nx)
        
        # 2. Random properties for each batch element: shape (batch_b, 1, 1)
        rand = lambda low, high: (torch.rand((batch_b, 1, 1), device=dev) * (high - low) + low)
        
        h_asphalt = rand(*asphalt_h_range)
        er_asphalt = rand(*asphalt_er_range)
        sig_asphalt = rand(*asphalt_sig_range)
        
        h_base = rand(*base_h_range)
        er_base = rand(*base_er_range)
        sig_base = rand(*base_sig_range)
        
        er_subgrade = rand(*subgrade_er_range)
        sig_subgrade = rand(*subgrade_sig_range)
        
        # Random roughness phase shifts per batch: shape (batch_b, 1, 1)
        phi_surf = torch.rand((batch_b, 1, 1), device=dev) * 2.0 * np.pi
        phi_asph = torch.rand((batch_b, 1, 1), device=dev) * 2.0 * np.pi
        phi_base = torch.rand((batch_b, 1, 1), device=dev) * 2.0 * np.pi
        
        surf_y = self.ground_y + 0.002 * torch.sin(2.0 * np.pi * X / 0.3 + phi_surf)
        asphalt_bottom = surf_y - h_asphalt + 0.005 * torch.sin(2.0 * np.pi * X / 0.5 + phi_asph)
        base_bottom = asphalt_bottom - h_base + 0.008 * torch.sin(2.0 * np.pi * X / 0.7 + phi_base)
        
        # Allocate tensors in VRAM: shape (batch_b, ny, nx)
        eps_r = torch.ones((batch_b, self.ny, self.nx), device=dev, dtype=torch.float32)
        sigma = torch.zeros((batch_b, self.ny, self.nx), device=dev, dtype=torch.float32)
        pec_mask = torch.zeros((batch_b, self.ny, self.nx), device=dev, dtype=torch.bool)
        
        # Subgrade
        mask_sub = Y <= base_bottom
        eps_r = torch.where(mask_sub, er_subgrade, eps_r)
        sigma = torch.where(mask_sub, sig_subgrade, sigma)
        
        # Base
        mask_base = (Y > base_bottom) & (Y <= asphalt_bottom)
        eps_r = torch.where(mask_base, er_base, eps_r)
        sigma = torch.where(mask_base, sig_base, sigma)
        
        # Asphalt
        mask_asph = (Y > asphalt_bottom) & (Y <= surf_y)
        eps_r = torch.where(mask_asph, er_asphalt, eps_r)
        sigma = torch.where(mask_asph, sig_asphalt, sigma)
        
        # Gravel clutter in base
        if add_gravel:
            for _ in range(num_gravels):
                gx = rand(self.scan_x_start - 0.2, self.scan_x_end + 0.2)
                gy = rand(self.ground_y - 0.35, self.ground_y - 0.12)
                gr = rand(0.015, 0.030)
                g_mask = (((X - gx)**2 + (Y - gy)**2) <= gr**2) & mask_base
                er_g = rand(8.0, 9.5)
                eps_r = torch.where(g_mask, er_g, eps_r)
                
        return eps_r, sigma, pec_mask

    def insert_object_gpu(self, eps_r, sigma, pec_mask, obj):
        """V2 explicit cross-sections; center depth, binary extended-type rasterization.

        Cable bundle is a concentric core/sheath effective cross-section, not
        resolved individual wires. New wavefields require reference validation.
        """
        from src.gpr_batch_metadata import TYPES
        kind=obj["type"]
        if kind not in TYPES:
            raise ValueError(f"Unsupported object type: {kind}")
        xc=float(obj["xc"]);yc=self.ground_y-float(obj["depth"])
        water=float(obj.get("water_ratio",0));silt=float(obj.get("silt_ratio",0))
        if not all(np.isfinite(v) for v in (xc,yc,water,silt)) or not 0<=water<=1 or not 0<=silt<1:
            raise ValueError("Invalid object coordinates/fill ratios")
        if kind in ("pec","upvc_air","upvc_water","upvc_triphase"):
            return self.insert_pipe_gpu(eps_r,sigma,pec_mask,xc=xc,yc=yc,
                radius=float(obj["radius"]),pipe_type=kind,conformal_smoothing=True,
                silt_ratio=silt,water_ratio=water*(1-silt))
        y=torch.arange(self.ny,device=eps_r.device)*self.dy+self.dy/2
        x=torch.arange(self.nx,device=eps_r.device)*self.dx+self.dx/2
        Y,X=torch.meshgrid(y,x,indexing="ij")
        if kind=="box_culvert":
            width=float(obj["width"]);height=float(obj["height"]);wall=float(obj.get("wall_thickness",.08))
            if not all(np.isfinite(v) and v>0 for v in (width,height,wall)) or wall>=min(width,height)/2:
                raise ValueError("Invalid culvert geometry")
            outer=(torch.abs(X-xc)<=width/2)&(torch.abs(Y-yc)<=height/2)
            inner=(torch.abs(X-xc)<width/2-wall)&(torch.abs(Y-yc)<height/2-wall)
            eps_r=torch.where(outer,8.,eps_r);sigma=torch.where(outer,.01,sigma)
            bottom=yc-height/2+wall;available=height-2*wall
            silt_top=bottom+silt*available;water_top=silt_top+water*(1-silt)*available
            eps_r=torch.where(inner,torch.where(Y<=silt_top,24.,torch.where(Y<=water_top,81.,1.)),eps_r)
            sigma=torch.where(inner,torch.where(Y<=silt_top,.045,torch.where(Y<=water_top,.015,0.)),sigma)
        else:
            radius=float(obj["radius"])
            if not np.isfinite(radius) or radius<=0:raise ValueError("Invalid radius")
            dist=torch.sqrt((X-xc)**2+(Y-yc)**2);outer=dist<=radius
            if kind=="sinkhole_void":
                eps_r=torch.where(outer,1.,eps_r);sigma=torch.where(outer,0.,sigma)
            else:
                core=dist<=.6*radius
                eps_r=torch.where(outer,2.5,eps_r);sigma=torch.where(outer,.0001,sigma)
                eps_r=torch.where(core,1.,eps_r);sigma=torch.where(core,1e5,sigma)
        pec_mask=pec_mask & ~outer
        return eps_r,sigma,pec_mask

    def insert_pipe_gpu(
        self,
        eps_r,
        sigma,
        pec_mask,
        xc,
        yc,
        radius,
        pipe_type="pec",         # "pec", "upvc_air", "upvc_water", "upvc_triphase"
        wall_thickness=0.008,
        conformal_smoothing=True,
        silt_ratio=0.25,
        water_ratio=0.45
    ):
        """
        Inserts underground pipes with optional Conformal Sub-cell Smoothing directly on GPU.
        Accepts 3D tensors: eps_r, sigma, pec_mask: (batch_b, ny, nx) or 2D (ny, nx)
        """
        if pipe_type not in ("pec","upvc_air","upvc_water","upvc_triphase"):
            raise ValueError(f"Unsupported pipe_type: {pipe_type}; use explicit v2 object routing")
        dev = eps_r.device
        orig_2d = (eps_r.dim() == 2)
        if orig_2d:
            eps_r = eps_r.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            pec_mask = pec_mask.unsqueeze(0)
            
        batch_b = eps_r.shape[0]
        y_1d = torch.arange(self.ny, device=dev, dtype=torch.float32) * self.dy + self.dy / 2.0
        x_1d = torch.arange(self.nx, device=dev, dtype=torch.float32) * self.dx + self.dx / 2.0
        Y, X = torch.meshgrid(y_1d, x_1d, indexing='ij')
        X = X.unsqueeze(0)  # (1, ny, nx)
        Y = Y.unsqueeze(0)  # (1, ny, nx)
        
        dist = torch.sqrt((X - xc)**2 + (Y - yc)**2)
        
        if pipe_type == "pec":
            if conformal_smoothing:
                f_pec = torch.clamp((radius + self.dx / 2.0 - dist) / self.dx, 0.0, 1.0)
                pec_mask |= (f_pec >= 0.5)
            else:
                pec_mask |= (dist <= radius)
                
        elif pipe_type.startswith("upvc"):
            inner_r = radius - wall_thickness
            if conformal_smoothing:
                f_out = torch.clamp((radius + self.dx / 2.0 - dist) / self.dx, 0.0, 1.0)
                f_in = torch.clamp((inner_r + self.dx / 2.0 - dist) / self.dx, 0.0, 1.0)
                f_shell = torch.clamp(f_out - f_in, 0.0, 1.0)
            else:
                f_out = (dist <= radius).float()
                f_in = (dist <= inner_r).float()
                f_shell = f_out - f_in
                
            # Shell: er = 3.0, sig = 0.0001
            eps_r = (1.0 - f_out) * eps_r + f_shell * 3.0
            sigma = (1.0 - f_out) * sigma + f_shell * 0.0001
            
            # Interior contents
            if pipe_type == "upvc_air":
                eps_r = eps_r + f_in * 1.0
                sigma = sigma + f_in * 0.0
            elif pipe_type == "upvc_water":
                eps_r = eps_r + f_in * 81.0
                sigma = sigma + f_in * 0.015
            elif pipe_type == "upvc_triphase":
                pipe_bottom = yc - inner_r
                pipe_diam = 2.0 * inner_r
                silt_top = pipe_bottom + silt_ratio * pipe_diam
                water_top = silt_top + water_ratio * pipe_diam
                
                m_silt = (Y <= silt_top).float()
                m_water = ((Y > silt_top) & (Y <= water_top)).float()
                m_air = (Y > water_top).float()
                
                eps_r = eps_r + f_in * (m_silt * 24.0 + m_water * 81.0 + m_air * 1.0)
                sigma = sigma + f_in * (m_silt * 0.040 + m_water * 0.015 + m_air * 0.0)

        if orig_2d:
            return eps_r.squeeze(0), sigma.squeeze(0), pec_mask.squeeze(0)
        return eps_r, sigma, pec_mask

    def solve_super_batch(
        self,
        eps_r_batch,
        sigma_batch,
        pec_mask_batch,
        num_traces=None,
        reuse_buffers=True
    ):
        """
        Executes 4D Super-Batch FDTD simulation for B independent B-scans simultaneously.
        Field tensors shape: (B, N_traces, ny, nx)
        Update coefficients Ca, Cb: (B, 1, ny, nx) broadcast across N_traces.
        """
        if isinstance(eps_r_batch, np.ndarray):
            eps_r = torch.from_numpy(eps_r_batch).to(self.device)
        else:
            eps_r = eps_r_batch.to(self.device)
            
        if isinstance(sigma_batch, np.ndarray):
            sigma = torch.from_numpy(sigma_batch).to(self.device)
        else:
            sigma = sigma_batch.to(self.device)
            
        if isinstance(pec_mask_batch, np.ndarray):
            pec_mask = torch.from_numpy(pec_mask_batch).to(self.device)
        else:
            pec_mask = pec_mask_batch.to(self.device)
            
        if eps_r.dim() == 2:
            eps_r = eps_r.unsqueeze(0)
            sigma = sigma.unsqueeze(0)
            pec_mask = pec_mask.unsqueeze(0)
            
        batch_b = eps_r.shape[0]
        n_traces = num_traces if num_traces is not None else self.num_traces
        
        # Antenna geometry
        ant_y_m = self.ground_y + self.source_height_above_ground
        ant_r = int(round(ant_y_m / self.dy))
        scan_xs = np.linspace(self.scan_x_start, self.scan_x_end, n_traces)
        tx_c = torch.tensor([int(round(x / self.dx)) for x in scan_xs], dtype=torch.long, device=self.device)
        rx_c = torch.tensor([int(round((x + self.ant_offset) / self.dx)) for x in scan_xs], dtype=torch.long, device=self.device)
        
        if not self.pml_w<=ant_r<self.ny-self.pml_w or torch.any(tx_c<self.pml_w) or torch.any(tx_c>=self.nx-self.pml_w) or torch.any(rx_c<self.pml_w) or torch.any(rx_c>=self.nx-self.pml_w):
            raise ValueError("Source/receiver antenna index outside non-PML domain")

        # 4-Cell interface smoothing if requested
        if self.interface_smoothing:
            eps_r, sigma = apply_4cell_interface_smoothing(eps_r, sigma)

        eps = eps_r * self.EPS_0
        term = (sigma * self.dt) / (2.0 * eps)
        Ca = (1.0 - term) / (1.0 + term)
        Cb = self.dt / (eps * self.dx) / (1.0 + term)
        
        # Bake PEC mask into Ca and Cb
        Ca[pec_mask] = 0.0
        Cb[pec_mask] = 0.0
        
        # Shape: (B, 1, ny, nx) broadcasts over N_traces
        Ca = Ca.unsqueeze(1)
        Cb = Cb.unsqueeze(1)
        
        c_hx = self.dt / (self.MU_0 * self.dy)
        c_hy = self.dt / (self.MU_0 * self.dx)
        
        if self.pml_formulation == "HORIPML":
            er_faces = compute_boundary_face_permittivity(eps_r, pml_w=self.pml_w)
            tensor_coeffs = prepare_inhomogeneous_horipml_tensors(
                pml_cells=self.pml_w,
                d=self.dx,
                dt=self.dt,
                er_faces=er_faces,
                device=self.device,
                dtype=torch.float32,
                is_4d=True,
            )
        else:
            # CPML parameters in 4D: (1, 1, ...)
            m_ord = 3.0
            sig_max = 0.8 * (m_ord + 1) / (self.dx * 377.0)
            p_rng = torch.arange(self.pml_w, device=self.device, dtype=torch.float32)
            dist = (self.pml_w - p_rng) / self.pml_w
            b_pml = torch.exp(-(sig_max * (dist**m_ord) + 0.05 * (1.0 - dist)) * self.dt / self.EPS_0)
            a_pml = (b_pml - 1.0) * (sig_max * (dist**m_ord)) / ((sig_max * (dist**m_ord)) + 0.05 * (1.0 - dist) + 1e-9)
            b_pml_rev, a_pml_rev = torch.flip(b_pml, [0]), torch.flip(a_pml, [0])

            b_x_l, a_x_l = b_pml.view(1, 1, 1, -1), a_pml.view(1, 1, 1, -1)
            b_x_r, a_x_r = b_pml_rev.view(1, 1, 1, -1), a_pml_rev.view(1, 1, 1, -1)
            b_y_b, a_y_b = b_pml.view(1, 1, -1, 1), a_pml.view(1, 1, -1, 1)
            b_y_t, a_y_t = b_pml_rev.view(1, 1, -1, 1), a_pml_rev.view(1, 1, -1, 1)

            b_x_le, a_x_le = b_pml[1:].view(1, 1, 1, -1), a_pml[1:].view(1, 1, 1, -1)
            b_x_re, a_x_re = b_pml_rev[:-1].view(1, 1, 1, -1), a_pml_rev[:-1].view(1, 1, 1, -1)
            b_y_be, a_y_be = b_pml[1:].view(1, 1, -1, 1), a_pml[1:].view(1, 1, -1, 1)
            b_y_te, a_y_te = b_pml_rev[:-1].view(1, 1, -1, 1), a_pml_rev[:-1].view(1, 1, -1, 1)

        # 4D Field allocations
        shape_4d = (batch_b, n_traces, self.ny, self.nx)
        buf = getattr(self, '_super_buffer_cache', None)
        if reuse_buffers and buf is not None and buf.get('shape') == shape_4d and buf.get('pml') == self.pml_formulation:
            Ez = buf['Ez'].zero_()
            Hx = buf['Hx'].zero_()
            Hy = buf['Hy'].zero_()
            psi_Hx_b = buf['psi_Hx_b'].zero_()
            psi_Hx_t = buf['psi_Hx_t'].zero_()
            psi_Hy_l = buf['psi_Hy_l'].zero_()
            psi_Hy_r = buf['psi_Hy_r'].zero_()
            psi_Ez_xl = buf['psi_Ez_xl'].zero_()
            psi_Ez_xr = buf['psi_Ez_xr'].zero_()
            psi_Ez_yb = buf['psi_Ez_yb'].zero_()
            psi_Ez_yt = buf['psi_Ez_yt'].zero_()
            bscan = buf['bscan'].zero_()
        else:
            Ez = torch.zeros(shape_4d, device=self.device)
            Hx = torch.zeros(shape_4d, device=self.device)
            Hy = torch.zeros(shape_4d, device=self.device)
            psi_Hx_b = torch.zeros((batch_b, n_traces, self.pml_w, self.nx), device=self.device)
            psi_Hx_t = torch.zeros((batch_b, n_traces, self.pml_w, self.nx), device=self.device)
            psi_Hy_l = torch.zeros((batch_b, n_traces, self.ny, self.pml_w), device=self.device)
            psi_Hy_r = torch.zeros((batch_b, n_traces, self.ny, self.pml_w), device=self.device)
            ez_pml_w = self.pml_w if self.pml_formulation == "HORIPML" else self.pml_w - 1
            psi_Ez_xl = torch.zeros((batch_b, n_traces, self.ny, ez_pml_w), device=self.device)
            psi_Ez_xr = torch.zeros((batch_b, n_traces, self.ny, ez_pml_w), device=self.device)
            psi_Ez_yb = torch.zeros((batch_b, n_traces, ez_pml_w, self.nx), device=self.device)
            psi_Ez_yt = torch.zeros((batch_b, n_traces, ez_pml_w, self.nx), device=self.device)
            bscan = torch.zeros((self.num_steps, batch_b, n_traces), device=self.device)
            if reuse_buffers:
                self._super_buffer_cache = {
                    'shape': shape_4d,
                    'pml': self.pml_formulation,
                    'Ez': Ez, 'Hx': Hx, 'Hy': Hy,
                    'psi_Hx_b': psi_Hx_b, 'psi_Hx_t': psi_Hx_t,
                    'psi_Hy_l': psi_Hy_l, 'psi_Hy_r': psi_Hy_r,
                    'psi_Ez_xl': psi_Ez_xl, 'psi_Ez_xr': psi_Ez_xr,
                    'psi_Ez_yb': psi_Ez_yb, 'psi_Ez_yt': psi_Ez_yt,
                    'bscan': bscan
                }

        b_mesh, tr_mesh = torch.meshgrid(
            torch.arange(batch_b, device=self.device),
            torch.arange(n_traces, device=self.device),
            indexing='ij'
        )
        tx_2d = tx_c.unsqueeze(0).expand(batch_b, -1)
        rx_2d = rx_c.unsqueeze(0).expand(batch_b, -1)

        # Ricker source wavelet evaluated at (n + 0.5) * dt (gprMax sources.py exact timing)
        t = (torch.arange(self.num_steps, device=self.device, dtype=torch.float32) + 0.5) * self.dt
        chi = np.sqrt(2.0) / self.f_center
        zeta = (np.pi * self.f_center)**2
        delay = t - chi
        ricker = (1.0 - 2.0 * zeta * delay**2) * torch.exp(-zeta * delay**2)
        eps_r_tx = float(eps_r[..., ant_r, tx_c].float().mean().item())
        src_scale = self.dt / (self.EPS_0 * eps_r_tx * self.dx * self.dy)

        # 4D Time stepping loop
        for n in range(self.num_steps):
            # Receiver recording before field update and source injection (gprMax solvers.py store_outputs policy)
            bscan[n, :, :] = Ez[b_mesh, tr_mesh, ant_r, rx_2d]

            Hx[:, :, :-1, :] -= c_hx * (Ez[:, :, 1:, :] - Ez[:, :, :-1, :])
            Hy[:, :, :, :-1] += c_hy * (Ez[:, :, :, 1:] - Ez[:, :, :, :-1])

            if self.pml_formulation == "HORIPML":
                apply_horipml_h_corrections(
                    Hx, Hy, Ez,
                    psi_Hx_b, psi_Hx_t, psi_Hy_l, psi_Hy_r,
                    tensor_coeffs, c_hx, c_hy, self.pml_w, self.top_pml_enabled
                )
            else:
                # CPML Hx
                psi_Hx_b = b_y_b * psi_Hx_b + a_y_b * (Ez[:, :, 1:self.pml_w+1, :] - Ez[:, :, :self.pml_w, :])
                Hx[:, :, :self.pml_w, :] -= c_hx * psi_Hx_b
                if self.top_pml_enabled:
                    psi_Hx_t = b_y_t * psi_Hx_t + a_y_t * (Ez[:, :, self.ny-self.pml_w:, :] - Ez[:, :, self.ny-self.pml_w-1:-1, :])
                    Hx[:, :, self.ny-self.pml_w:, :] -= c_hx * psi_Hx_t

                # CPML Hy
                psi_Hy_l = b_x_l * psi_Hy_l + a_x_l * (Ez[:, :, :, 1:self.pml_w+1] - Ez[:, :, :, :self.pml_w])
                Hy[:, :, :, :self.pml_w] += c_hy * psi_Hy_l
                psi_Hy_r = b_x_r * psi_Hy_r + a_x_r * (Ez[:, :, :, self.nx-self.pml_w:] - Ez[:, :, :, self.nx-self.pml_w-1:-1])
                Hy[:, :, :, self.nx-self.pml_w:] += c_hy * psi_Hy_r

            # Update Ez
            Ez[:, :, 1:-1, 1:-1] = Ca[:, :, 1:-1, 1:-1] * Ez[:, :, 1:-1, 1:-1] + Cb[:, :, 1:-1, 1:-1] * (
                (Hy[:, :, 1:-1, 1:-1] - Hy[:, :, 1:-1, :-2]) - (Hx[:, :, 1:-1, 1:-1] - Hx[:, :, :-2, 1:-1])
            )

            if self.pml_formulation == "HORIPML":
                apply_horipml_e_corrections(
                    Ez, Hx, Hy,
                    psi_Ez_xl, psi_Ez_xr, psi_Ez_yb, psi_Ez_yt,
                    tensor_coeffs, Cb, self.pml_w, self.top_pml_enabled
                )
            else:
                # CPML Ez
                psi_Ez_xl = b_x_le * psi_Ez_xl + a_x_le * (Hy[:, :, :, 1:self.pml_w] - Hy[:, :, :, :self.pml_w-1])
                Ez[:, :, :, 1:self.pml_w] += Cb[:, :, :, 1:self.pml_w] * psi_Ez_xl
                psi_Ez_xr = b_x_re * psi_Ez_xr + a_x_re * (Hy[:, :, :, self.nx-self.pml_w:-1] - Hy[:, :, :, self.nx-self.pml_w-1:-2])
                Ez[:, :, :, self.nx-self.pml_w:-1] += Cb[:, :, :, self.nx-self.pml_w:-1] * psi_Ez_xr
                psi_Ez_yb = b_y_be * psi_Ez_yb + a_y_be * (Hx[:, :, 1:self.pml_w, :] - Hx[:, :, :self.pml_w-1, :])
                Ez[:, :, 1:self.pml_w, :] -= Cb[:, :, 1:self.pml_w, :] * psi_Ez_yb
                if self.top_pml_enabled:
                    psi_Ez_yt = b_y_te * psi_Ez_yt + a_y_te * (Hx[:, :, self.ny-self.pml_w:-1, :] - Hx[:, :, self.ny-self.pml_w-1:-2, :])
                    Ez[:, :, self.ny-self.pml_w:-1, :] -= Cb[:, :, self.ny-self.pml_w:-1, :] * psi_Ez_yt

            # Source injection
            Ez[b_mesh, tr_mesh, ant_r, tx_2d] -= src_scale * ricker[n]

        if str(self.device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.synchronize()
        if batch_b == 1:
            return bscan[:, 0, :].cpu().numpy()
        return bscan.cpu().numpy()

    def solve_bscan(self, eps_r_cpu, sigma_cpu, pec_mask_cpu, reuse_buffers=True):
        """Backward-compatible single B-scan solver delegating to solve_super_batch."""
        return self.solve_super_batch(eps_r_cpu, sigma_cpu, pec_mask_cpu, reuse_buffers=reuse_buffers)

    @staticmethod
    def apply_agc_gpu(bscan_tensor, window_len=150):
        """
        Applies 1D Automatic Gain Control (AGC) fully vectorized on GPU.
        Input: bscan_tensor: (num_steps, num_traces) or (num_steps, batch_b, num_traces) on CUDA/CPU
        Output: normalized bscan tensor [-1, 1] on same device
        """
        orig_shape = bscan_tensor.shape
        num_steps = orig_shape[0]
        # Reshape to (M, 1, num_steps) where M is total number of A-scan traces across all batches
        flat_tr = bscan_tensor.reshape(num_steps, -1).transpose(0, 1).unsqueeze(1)  # (M, 1, num_steps)
        abs_tr = torch.abs(flat_tr)
        pad = window_len // 2
        padded = torch.nn.functional.pad(abs_tr, (pad, pad), mode='replicate')
        rms = torch.nn.functional.avg_pool1d(padded, kernel_size=window_len, stride=1)
        rms = rms[:, 0, :num_steps].transpose(0, 1) + 1e-8  # (num_steps, M)
        rms = rms.reshape(orig_shape)
        bscan_gain = bscan_tensor / rms
        vmax = torch.quantile(torch.abs(bscan_gain), 0.98)
        return torch.clamp(bscan_gain / (vmax + 1e-9), -1.0, 1.0)

    @classmethod
    def apply_agc_gain(cls, bscan, window_len=150, device=None):
        """
        Applies Automatic Gain Control (AGC) to enhance deep reflections for visualization.
        Automatically leverages GPU acceleration via PyTorch if available, eliminating CPU bottlenecks.
        """
        if isinstance(bscan, torch.Tensor):
            return cls.apply_agc_gpu(bscan, window_len)
        elif isinstance(bscan, np.ndarray):
            dev = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
            t_in = torch.from_numpy(bscan).to(dev)
            t_out = cls.apply_agc_gpu(t_in, window_len)
            return t_out.cpu().numpy()
        else:
            raise TypeError(f"Unsupported bscan type: {type(bscan)}")
