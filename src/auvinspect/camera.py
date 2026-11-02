"""Camera rig geometry: viewing directions, tilt, intrinsics.

MuJoCo cameras look along their -z axis with +x to the image right and +y up. Axes here are
expressed in the vehicle frame (x forward, y port, z up).
"""
from __future__ import annotations

import math

import mujoco
import numpy as np

# viewing direction and image-up vector in the vehicle frame
LOOK = {
    'forward': ((1, 0, 0), (0, 0, 1)),
    'port': ((0, 1, 0), (0, 0, 1)),
    'starboard': ((0, -1, 0), (0, 0, 1)),
    'down': ((0, 0, -1), (1, 0, 0)),
}


def axes(look: str, tilt_rad: float = 0.0) -> np.ndarray:
    """Camera axes as columns [x y z] (3x3) after pitching the view up by tilt_rad about the image x axis."""
    f, u = (np.asarray(v, float) for v in LOOK[look])
    x = np.cross(f, u)
    f, u = f * math.cos(tilt_rad) + np.cross(x, f) * math.sin(tilt_rad), u * math.cos(tilt_rad) + np.cross(x, u) * math.sin(tilt_rad)
    return np.stack([x, u, -f], 1)


def quat(mat: np.ndarray) -> np.ndarray:
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.ascontiguousarray(mat, float).reshape(-1))
    return q


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def fovy_deg(cams: dict) -> float:
    return math.degrees(2 * math.atan(math.tan(math.radians(cams['hfov_deg']) / 2) * cams['height'] / cams['width']))


def focal_px(cams: dict) -> float:
    return cams['height'] / 2 / math.tan(math.radians(fovy_deg(cams)) / 2)


def pixel_rays(cams: dict) -> np.ndarray:
    """Per-pixel ray in camera coordinates with unit -z component, shape (H, W, 3)."""
    W, H, f = cams['width'], cams['height'], focal_px(cams)
    u = (np.arange(W) + 0.5 - W / 2) / f
    v = -(np.arange(H) + 0.5 - H / 2) / f
    uu, vv = np.meshgrid(u, v)
    return np.stack([uu, vv, -np.ones_like(uu)], -1)


def footprint_mm(depth: np.ndarray, rays_cam: np.ndarray, normals_cam: np.ndarray, focal: float) -> np.ndarray:
    """Worst-direction size on the surface of one pixel (mm), exact for a locally flat surface.

    A pixel's ray is q = (x, y, -1) in the camera frame (x = u / f, y = v / f) and meets the surface at
    depth Z along the optical axis. Moving one pixel along the image x (or y) axis moves the surface
    point by (Z / f) (e - q (n.e) / (n.q)) for e = (1, 0, 0) (or (0, 1, 0)), n the surface normal. The
    footprint is the larger singular value of those two vectors. A surface parallel to the image plane
    gives Z / f in every pixel; tilting it by i stretches the centre pixel to Z / (f cos i).
    """
    q = np.asarray(rays_cam, float)
    n = np.asarray(normals_cam, float)
    nq = (n * q).sum(-1)
    nq = np.where(np.abs(nq) < 1e-9, -1e-9, nq)
    j1 = -q * (n[..., 0] / nq)[..., None]
    j1[..., 0] += 1.0
    j2 = -q * (n[..., 1] / nq)[..., None]
    j2[..., 1] += 1.0
    a, b, c = (j1 * j1).sum(-1), (j1 * j2).sum(-1), (j2 * j2).sum(-1)
    smax = np.sqrt((a + c) / 2 + np.sqrt(((a - c) / 2) ** 2 + b * b))
    return np.asarray(depth, float) / focal * smax * 1000.0
