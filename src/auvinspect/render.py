"""Render the camera rig and compute exact per-pixel ground truth.

MuJoCo rasterises the geometry: depth, segmentation and unlit albedo. Light, water and sensor come
from optics.py: the lamps on the inspection camera's tilt unit light the hull through the water
(Jaffe-McGlamery single scattering: direct, forward scatter, backscatter, optional daylight) and a
photon-counting sensor with auto exposure and motion blur records the image.

Each frame holds the image, the albedo, depth and range, and per pixel: the hull surface coordinates
(x, s), the incidence angle, the exact pixel footprint (gsd), the contrast-to-noise ratio of a faint
defect over one coverage cell, and the item label. Labels come from back-projecting the depth onto
the analytic hull and reading the scenario's label raster, so they do not depend on how the mesh is
tessellated or textured.
"""
from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

import mujoco
import numpy as np

from . import optics
from .camera import axes, focal_px, footprint_mm, pixel_rays, quat, yaw_quat
from .scenario import Scenario
from .scene import build_mjcf

LABEL_SOUND = -1        # hull paint without a defect
LABEL_WATER = -2        # no geometry along the ray
LABEL_SEABED = -3


@dataclass
class Frame:
    camera: str
    rgb: np.ndarray            # (H, W, 3) uint8, the camera's image
    albedo: np.ndarray         # (H, W, 3) uint8, surface albedo seen by each pixel (0 with no geometry)
    depth: np.ndarray          # (H, W) float32, distance along the optical axis (m)
    range: np.ndarray          # (H, W) float32, distance along the ray (m); inf with no geometry
    label: np.ndarray          # (H, W) int32 item id or LABEL_*
    on_hull: np.ndarray        # (H, W) bool, pixel sees the hull surface
    on_anode: np.ndarray       # (H, W) bool, pixel sees an anode (x, s are its projection on the hull)
    x: np.ndarray              # (H, W) float32 hull x (valid where on_hull)
    s: np.ndarray              # (H, W) float32 hull girth
    incidence_deg: np.ndarray  # (H, W) float32 angle between the ray and the hull normal
    gsd_mm: np.ndarray         # (H, W) float32 surface size of one pixel along its worst direction
    cnr: np.ndarray            # (H, W) float32 contrast-to-noise ratio of the reference defect over one cell
    exposure_s: float
    gain: float
    blur_px: float             # motion blur during the exposure (pixels, at the image centre)
    cam_pos: np.ndarray
    cam_mat: np.ndarray        # camera axes as columns (world frame)
    signal: bool = True        # False when the camera delivered no image (blackout fault)
    radiance: dict = field(default_factory=dict)   # 'direct', 'forward', 'backscatter' (W/m^2/sr, per channel)


class InspectionRenderer:
    def __init__(self, scn: Scenario):
        if os.environ.get('MUJOCO_GL') is None:
            os.environ['MUJOCO_GL'] = 'egl'
        self.scn, self.cfg = scn, scn.cfg
        xml, assets = build_mjcf(scn)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)
        cams = self.cfg['cameras']
        self.W, self.H = cams['width'], cams['height']
        self.renderer = mujoco.Renderer(self.model, self.H, self.W)
        self.cam_id = {n: self.model.camera(n).id for n in cams['rig']}
        self.lamp_cam = self.cam_id[self.cfg['survey']['camera']]
        self.look = {n: c['look'] for n, c in cams['rig'].items()}
        self.hull_geom = self.model.geom('hull').id
        self.seabed_geom = self.model.geom('seabed').id
        self.geom_item = {self.model.geom(f'anode_{a["id"]}').id: a['id'] for a in scn.anodes()}
        self.focal = focal_px(cams)
        self.rays = pixel_rays(cams)
        self.ray_norm = np.linalg.norm(self.rays, axis=-1).astype(np.float32)
        self.unit_rays = self.rays / self.ray_norm[..., None]
        self.per_radiance = optics.electrons_per_radiance(cams['sensor'], 1.0 / self.ray_norm)   # e/s per W/m^2/sr
        self.zfar = cams['zfar_m']
        self.px = scn.mm_per_px / 1000.0
        mujoco.mj_forward(self.model, self.data)

    def close(self):
        self.renderer.close()

    def set_pose(self, position, yaw: float = 0.0, tilt_deg: dict | None = None, attitude=None):
        """Place the rig (vehicle frame origin) with a yaw, or a full `attitude` quaternion (w, x, y, z),
        and set per-camera tilt angles (degrees, view pitched up)."""
        self.data.mocap_pos[0] = position
        self.data.mocap_quat[0] = yaw_quat(yaw) if attitude is None else attitude
        for name, cid in self.cam_id.items():
            t = math.radians((tilt_deg or {}).get(name, 0.0))
            self.model.cam_quat[cid] = quat(axes(self.look[name], t))
        mujoco.mj_forward(self.model, self.data)

    def _render(self, camera: str, mode: str) -> np.ndarray:
        r = self.renderer
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        if mode == 'depth':
            r.enable_depth_rendering()
        elif mode == 'seg':
            r.enable_segmentation_rendering()
        r.update_scene(self.data, camera=self.cam_id[camera])
        out = r.render()
        r.disable_depth_rendering()
        r.disable_segmentation_rendering()
        return out

    def capture(self, camera: str, rng: np.random.Generator, water: dict | None = None,
                blackout: bool = False, motion: tuple | None = None) -> Frame:
        """Render one camera. `water` replaces the scenario's water (a turbidity event); `motion` is the
        camera's (velocity, angular velocity) in the world frame during the exposure. A blackout returns
        a black image whose ground truth still records what was in view."""
        cfg, hull = self.cfg, self.scn.hull
        water = water or cfg['water']
        albedo = self._render(camera, 'rgb').copy()
        depth = self._render(camera, 'depth').astype(np.float32)
        seg = self._render(camera, 'seg')
        cid = self.cam_id[camera]
        cpos = self.data.cam_xpos[cid].copy()
        cmat = self.data.cam_xmat[cid].reshape(3, 3).copy()
        geom = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        has_geom = (geom >= 0) & (depth < self.zfar * 0.999)
        rng_m = np.where(has_geom, depth * self.ray_norm, np.inf).astype(np.float32)
        pts = cpos + (self.rays * depth[..., None]) @ cmat.T
        x, s, _ = hull.inverse(pts)
        on_hull = has_geom & (geom == self.hull_geom)
        on_seabed = has_geom & (geom == self.seabed_geom)
        label = np.full(depth.shape, LABEL_WATER, np.int32)
        label[on_seabed] = LABEL_SEABED
        rows = np.clip((s / self.px).astype(np.int64), 0, self.scn.labels.shape[0] - 1)
        cols = np.clip((x / self.px).astype(np.int64), 0, self.scn.labels.shape[1] - 1)
        label[on_hull] = self.scn.labels[rows[on_hull], cols[on_hull]]
        on_anode = np.zeros(depth.shape, bool)
        for gid, iid in self.geom_item.items():
            m = has_geom & (geom == gid)
            label[m] = iid
            on_anode |= m
        # surface normals: the analytic hull (anode faces are parallel to it), the flat seabed
        normal = hull.normal(s)
        normal[on_seabed] = (0.0, 0.0, 1.0)
        dirs = self.unit_rays @ cmat.T
        cos_inc = np.clip(-(dirs * normal).sum(-1), 1e-3, 1.0)
        incidence = np.degrees(np.arccos(cos_inc)).astype(np.float32)
        gsd = footprint_mm(depth, self.rays, normal @ cmat, self.focal).astype(np.float32)

        # light through the water
        o = cfg['optics']
        lamps = optics.Lamps.mounted(self.data.cam_xpos[self.lamp_cam], self.data.cam_xmat[self.lamp_cam].reshape(3, 3),
                                     cfg['cameras']['lamps'])
        _, _, c = optics.iops(water)
        s1 = np.zeros(depth.shape + (3,))                       # direct radiance per unit albedo
        E = optics.irradiance(pts[has_geom], normal[has_geom], lamps, water, optics.forward_eta(water, o))
        s1[has_geom] = E / math.pi * np.exp(-c * rng_m[has_geom][:, None])
        direct = s1 * (albedo / 255.0)
        forward = optics.forward_scatter(direct, rng_m, has_geom, water, self.focal, o)
        back = optics.backscatter_image(cpos, dirs, rng_m, lamps, water, o) + optics.daylight_veil(water, -cpos[2], rng_m)

        # the sensor
        sensor = cfg['cameras']['sensor']
        rate = (direct + forward + back) * self.per_radiance
        wb = optics.white_balance(cfg['cameras'])
        t, gain = optics.auto_exposure(rate, sensor, wb)
        blur = self._blur(cmat, depth, has_geom, motion, t)
        rgb = optics.expose(optics.motion_blur(rate, blur), t, gain, rng, sensor, wb)
        if blackout:
            rgb = np.zeros_like(rgb)
        s1_rate = s1 * self.per_radiance
        bg = optics.background_rate(s1_rate, back * self.per_radiance, rng_m, water, cfg)
        cnr = np.where(has_geom, optics.cell_cnr(s1_rate, bg, t, gain, gsd, cfg), 0.0).astype(np.float32)
        return Frame(camera, rgb, albedo, depth, rng_m, label, on_hull, on_anode, x.astype(np.float32), s.astype(np.float32),
                     incidence, gsd, cnr, t, gain, float(np.hypot(*blur)), cpos, cmat, signal=not blackout,
                     radiance={'direct': direct.astype(np.float32), 'forward': forward.astype(np.float32),
                               'backscatter': back.astype(np.float32)})

    def _blur(self, cmat: np.ndarray, depth: np.ndarray, has_geom: np.ndarray, motion, t: float) -> np.ndarray:
        """Image motion of the centre pixel over the exposure, (columns, rows) in pixels."""
        if motion is None:
            return np.zeros(2)
        v, w = (np.asarray(m, float) @ cmat for m in motion)    # camera frame
        H, W = depth.shape
        Z = float(depth[H // 2, W // 2]) if has_geom[H // 2, W // 2] else math.inf
        du = self.focal * (w[1] - v[0] / Z)
        dv = self.focal * (-w[0] - v[1] / Z)
        return np.array([du, -dv]) * t
