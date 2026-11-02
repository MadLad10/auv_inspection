"""Predicted image quality of a camera pose against the analytic hull.

The planner and the onboard software have no renderer and no ground truth. They evaluate the same
optics (optics.py) on the analytic hull for a camera pose they choose or estimate, with the water they
assume or measure, and the coverage limits the referee applies. Surfaces are taken to carry the
reference albedo (coverage.reference_albedo): the question is whether a faint defect on hull paint
would be seen there.
"""
from __future__ import annotations

import math

import numpy as np

from . import optics
from .camera import focal_px, footprint_mm, pixel_rays
from .coverage import within_limits
from .hull import HullGeometry


def view(hull: HullGeometry, cfg: dict, cam_pos: np.ndarray, cam_axes: np.ndarray, rays_cam: np.ndarray, water: dict,
         bounded: bool = True) -> dict:
    """Geometry and light for camera rays (q = (x, y, -1) in the camera frame) with the lamps mounted on
    this camera. Rays are a regular image grid (H, W, 3) or an ordered sequence (M, 3) of neighbouring
    rays; either way the backscatter is integrated on every optics.backscatter_grid_px-th ray and
    interpolated in between (it varies slowly across the image). bounded=False extends the hull section
    past its ends (for footprint geometry beyond them)."""
    q = np.asarray(rays_cam, float)
    qn = np.linalg.norm(q, axis=-1)
    dirs = (q / qn[..., None]) @ cam_axes.T
    t = hull.raycast(cam_pos, dirs, bounded=bounded)
    hit = np.isfinite(t)
    pts = cam_pos + np.where(hit, t, 0.0)[..., None] * dirs
    x, s, _ = hull.inverse(pts)
    n = hull.normal(s)
    cos_inc = np.clip(-(dirs * n).sum(-1), 1e-3, 1.0)
    depth = np.where(hit, t / qn, np.inf)
    gsd = footprint_mm(np.where(hit, depth, 0.0), q, n @ cam_axes, focal_px(cfg['cameras']))
    o = cfg['optics']
    lamps = optics.Lamps.mounted(cam_pos, cam_axes, cfg['cameras']['lamps'])
    _, _, c = optics.iops(water)
    E = optics.irradiance(pts, n, lamps, water, optics.forward_eta(water, o))
    s1 = np.where(hit[..., None], E / math.pi * np.exp(-c * np.where(hit, t, 0.0)[..., None]), 0.0)
    if q.ndim == 3:
        back = optics.backscatter_image(cam_pos, dirs, t, lamps, water, o)
    else:
        s_grid = optics.path_grid(o)
        idx = np.unique(np.r_[0:len(dirs):o['backscatter_grid_px'], len(dirs) - 1])
        sub = optics.backscatter_cumulative(cam_pos, dirs[idx], lamps, water, s_grid)
        pos = np.interp(np.arange(len(dirs)), idx, np.arange(len(idx)))
        i0 = np.minimum(pos.astype(int), max(len(idx) - 2, 0))
        w = (pos - i0)[:, None, None]
        table = sub[i0] if len(idx) == 1 else (1 - w) * sub[i0] + w * sub[np.minimum(i0 + 1, len(idx) - 1)]
        back = optics.interpolate_along(table, s_grid, t)
    back = back + optics.daylight_veil(water, -float(cam_pos[2]), t)
    k = optics.electrons_per_radiance(cfg['cameras']['sensor'], 1.0 / qn)
    return {'hit': hit, 'x': x, 's': s, 'range': t, 'incidence_deg': np.degrees(np.arccos(cos_inc)), 'gsd_mm': gsd,
            's1_rate': s1 * k, 'back_rate': back * k}


def background(v: dict, cfg: dict, water: dict) -> np.ndarray:
    return optics.background_rate(v['s1_rate'], v['back_rate'], v['range'], water, cfg)


def quality(v: dict, cfg: dict, water: dict, exposure_s: float, gain: float) -> np.ndarray:
    """Which rays meet every coverage limit for this exposure."""
    cnr = optics.cell_cnr(v['s1_rate'], background(v, cfg, water), exposure_s, gain, v['gsd_mm'], cfg)
    return v['hit'] & within_limits(cfg['coverage'], v['range'], v['incidence_deg'], v['gsd_mm'], cnr)


def metering_rays(cfg: dict, n: int = 12) -> np.ndarray:
    """A coarse grid of the pixels the auto exposure meters (rays in the camera frame)."""
    cams = cfg['cameras']
    ae = cams['sensor']['auto_exposure']
    rays = pixel_rays(cams)
    H, W = rays.shape[:2]
    fh, fw = int(H * ae['metering_fraction'] / 2), int(W * ae['metering_fraction'] / 2)
    rr = np.linspace(H // 2 - fh, H // 2 + fh - 1, n).astype(int)
    cc = np.linspace(W // 2 - fw, W // 2 + fw - 1, n).astype(int)
    return rays[np.ix_(rr, cc)]


def exposure(hull: HullGeometry, cfg: dict, cam_pos: np.ndarray, cam_axes: np.ndarray, water: dict) -> tuple[float, float]:
    """Exposure time and gain the auto exposure would choose for this view of hull paint."""
    v = view(hull, cfg, cam_pos, cam_axes, metering_rays(cfg), water)
    return optics.auto_exposure(background(v, cfg, water)[None], cfg['cameras']['sensor'], optics.white_balance(cfg['cameras']))
