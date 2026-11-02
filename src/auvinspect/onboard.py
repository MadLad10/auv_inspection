"""The vehicle's own software: navigation, frame checks, claimed coverage and revisit planning.

It reads sensor readings, its own commands and the hull model (the ship's drawings). It never reads
the scenario's items, labels or water, or the simulator's truth.

Navigation is auv_navigation_fusion's EKF on the IMU, DVL, depth and compass, with two hull-relative
updates added through its scalar-update interface (Jacobians by finite differences of the analytic
ray cast):
  standoff  the sounder's range to the hull, which pins the distance from the hull (and, with depth,
            the girth position);
  hull end  lanes run past both hull ends, so the sounder's echo stops or starts where its beam
            crosses x = 0 or x = L: an along-ship position fix, twice per lane. Between fixes the
            along-ship uncertainty grows with dead reckoning.

A sensor health check excludes the depth sensor from navigation while its readings are frozen
(onboard.frozen_samples identical readings in a row: a live sensor always shows noise). A gate
cannot see a sensor stuck at a plausible value; while it is excluded, depth comes from the DVL and
the standoff sounder, and the growing uncertainty reaches the claim logic.

Each frame is checked (a constant, noiseless frame means a dead camera and is rejected) and projected onto the hull model from the
estimated camera pose (navigation estimate, AHRS roll and pitch, tilt encoder). The pixels that would
pass the coverage limits, judged with the image physics (predict.py) for the water the water-quality
sensor measures and the exposure and gain the camera used, form the frame's claimed footprint. When the position uncertainty along the
ship or along the girth exceeds onboard.claim_max_std_m, the footprint is kept as seen but unverified
rather than claimed. Below that, the footprint is shrunk by onboard.claim_sigma_k times those
uncertainties before it is claimed, so a claim survives the pose error the vehicle expects. Holes are cells not claimed (never seen, or seen only while uncertain); each
cluster gets revisit lanes placed with the planner's own strip rules. Revisits are not flown while
the vehicle is itself too uncertain of its position to claim what it would see.
"""
from __future__ import annotations

import math
from collections import deque

import numpy as np
from auv_navigation_fusion.ekf import rot
from PIL import Image, ImageFilter

from . import predict
from .camera import axes, pixel_rays
from .config import ConfigError
from .coverage import CellGrid
from .guidance import Leg
from .hull import HullGeometry
from .planner import band_lanes, capture_step, lane_xs, strips

NAV_SENSORS = ('imu', 'dvl', 'pressure', 'compass')


class Onboard:
    def __init__(self, hull: HullGeometry, cfg: dict, x0, yaw0: float):
        from auv_navigation_fusion import NavigationFusion, load_config
        self.hull, self.cfg = hull, cfg
        n = cfg['navigation']
        fix = n['initial_fix_std']
        nav_cfg = load_config(overrides=n['ekf_overrides'])
        nav_cfg['initial_std'].update(pos=fix['pos_m'], yaw_deg=fix['yaw_deg'])
        self.nav = NavigationFusion(nav_cfg, x0=tuple(float(v) for v in x0), yaw0=float(yaw0))
        cams, cam = cfg['cameras'], cfg['survey']['camera']
        self.look = cams['rig'][cam]['look']
        self.offset = np.asarray(cams['rig'][cam]['offset'], float)
        k = cfg['onboard']['claim_pixel_stride']
        self.rays_cam = pixel_rays(cams)[k // 2::k, k // 2::k]
        self.claimed = CellGrid(hull, cfg)
        self.unverified = CellGrid(hull, cfg)     # seen while too uncertain of position to claim
        self.water: dict | None = None            # the water as measured
        self.tilt = 0.0                           # camera tilt from its encoder (deg)
        self.imu: dict = {}
        self._last_standoff = -math.inf
        self._prev_beam: tuple[bool, float] | None = None
        self.echo = False                         # the standoff sounder currently sees the hull
        self.battery = 100.0                      # state of charge (%) from the battery monitor
        self.blocked: set[str] = set()            # sensors the health check excludes from navigation
        self.health_log: list[dict] = []
        self._depths: list[float] = []
        self._last_depth_t = -math.inf
        self.standoff_stats = {'accepted': 0, 'rejected': 0}
        self.hull_end_stats = {'accepted': 0, 'rejected': 0}
        self._strips = None

    # ------------------------------------------------------------------ navigation
    def update(self, readings: dict, t: float):
        self._check_depth(readings.get('pressure'), t)
        self.nav.update({k: readings[k] for k in NAV_SENSORS if k in readings}, time_s=t, blocked=self.blocked)
        self.imu = readings.get('imu', self.imu)
        self.tilt = readings['tilt']['deg']
        self.battery = readings.get('battery', {}).get('percent', self.battery)
        tilt_deg = self.tilt
        wq = readings.get('water_quality')
        if wq is not None:
            c, b = np.asarray(wq['attenuation_per_m']), np.asarray(wq['scattering_per_m'])
            o = self.cfg['onboard']
            self.water = {'absorption_per_m': np.maximum(c - b, 0.0).tolist(), 'scattering_per_m': b.tolist(),
                          'phase_g': o['assumed_phase_g'], 'daylight': o['assumed_daylight']}
        so = readings.get('standoff')
        n = self.cfg['navigation']
        if so is not None and so['t'] > self._last_standoff:
            self._last_standoff = so['t']
            x_beam = self._beam_x(None, tilt_deg)
            prev, self._prev_beam = self._prev_beam, (so['valid'], x_beam)
            self.echo = bool(so['valid'])
            if n['hull_end_update'] and prev is not None and prev[0] != so['valid'] and math.isfinite(prev[1]) and math.isfinite(x_beam):
                self._hull_end_update(prev[1], x_beam, lost=prev[0], tilt_deg=tilt_deg)
            if so['valid'] and n['standoff_update']:
                self._standoff_update(so['range'], tilt_deg)

    def _check_depth(self, p: dict | None, t: float):
        """Exclude the depth sensor while its last onboard.frozen_samples readings are identical."""
        if p is None or p['t'] <= self._last_depth_t:
            return
        self._last_depth_t = p['t']
        n = self.cfg['onboard']['frozen_samples']
        self._depths = (self._depths + [p['depth']])[-n:]
        frozen = len(self._depths) == n and len(set(self._depths)) == 1
        if frozen and 'pressure' not in self.blocked:
            self.blocked.add('pressure')
            self.health_log.append({'t': round(t, 3), 'sensor': 'pressure', 'event': 'frozen: excluded'})
        elif 'pressure' in self.blocked and self._depths[-1] != self._depths[-2]:
            self.blocked.discard('pressure')
            self.health_log.append({'t': round(t, 3), 'sensor': 'pressure', 'event': 'live again: restored'})

    def state(self) -> dict:
        st = self.nav.state()
        f = self.nav.filter
        gyro = self.imu.get('gyro', [0.0, 0.0, 0.0])
        return {**st, 'roll': float(f.roll), 'pitch': float(f.pitch), 'q': float(gyro[1])}

    def position(self) -> np.ndarray:
        st = self.nav.state()
        return np.array([st['x'], st['y'], st['z']])

    def camera_pose(self, tilt_deg: float, state=None) -> tuple[np.ndarray, np.ndarray]:
        """Estimated inspection-camera position and axes (columns, world); state = (x, y, z, yaw)."""
        f = self.nav.filter
        x, y, z, yaw = state if state is not None else (*f.s[:3], f.s[6])
        R = rot(f.roll, f.pitch, yaw)
        return np.array([x, y, z]) + R @ self.offset, R @ axes(self.look, math.radians(tilt_deg))

    def _range(self, state, tilt_deg: float) -> float:
        c, A = self.camera_pose(tilt_deg, state)
        return float(self.hull.raycast(c, -A[:, 2]))

    def _beam_x(self, state, tilt_deg: float) -> float:
        """Along-ship position where the sounder beam meets the hull section extended past its ends."""
        c, A = self.camera_pose(tilt_deg, state)
        t = float(self.hull.raycast(c, -A[:, 2], bounded=False))
        return float(c[0] - t * A[0, 2]) if math.isfinite(t) else math.inf

    def _jacobian(self, fn, tilt_deg: float) -> tuple[float, list[float]]:
        """Value and finite-difference derivatives of fn(state) for state = (x, y, z, yaw)."""
        f, h = self.nav.filter, self.cfg['navigation']['jacobian_step']
        base = np.array([*f.s[:3], f.s[6]])
        v0 = fn(base, tilt_deg)
        d = []
        for i in range(4):
            p = base.copy()
            p[i] += h
            v = fn(p, tilt_deg)
            d.append((v - v0) / h if math.isfinite(v) and math.isfinite(v0) else 0.0)
        return v0, d

    def _hull_end_update(self, x_prev: float, x_now: float, lost: bool, tilt_deg: float):
        """The echo vanished (lost) or appeared between two soundings: the beam crossed a hull end.
        Which end follows from the direction of travel (losing the echo moving +x means x = L);
        when the vehicle barely moved, the nearer end."""
        n, f = self.cfg['navigation'], self.nav.filter
        x_mid = (x_prev + x_now) / 2
        L = self.hull.length
        if abs(x_now - x_prev) > n['jacobian_step']:
            edge = L if (x_now > x_prev) == lost else 0.0
        else:
            edge = 0.0 if abs(x_mid) <= abs(x_mid - L) else L
        _, d = self._jacobian(self._beam_x, tilt_deg)
        var = (x_now - x_prev) ** 2 / 12 + n['hull_end_std_m'] ** 2 + d[2] ** 2 * float(f.P[2, 2])
        ok = f.update_scalar('hull_end', edge - x_mid, (d[0], d[1], d[3]), var, n['hull_end_gate'])
        self.hull_end_stats['accepted' if ok else 'rejected'] += 1

    def _standoff_update(self, r_meas: float, tilt_deg: float):
        f, n = self.nav.filter, self.cfg['navigation']
        r0, d = self._jacobian(self._range, tilt_deg)
        if not math.isfinite(r0):
            return
        # depth is not in the scalar update's state slice; its (small) uncertainty becomes measurement noise
        var = self.cfg['sensors']['standoff']['noise_m'] ** 2 + d[2] ** 2 * float(f.P[2, 2])
        ok = f.update_scalar('standoff', r_meas - r0, (d[0], d[1], d[3]), var, n['standoff_gate'])
        self.standoff_stats['accepted' if ok else 'rejected'] += 1

    def hull_std(self, s: float) -> tuple[float, float]:
        """1-sigma position uncertainty along the ship and along the girth at girth s."""
        P = self.nav.filter.P[:3, :3]
        phi = float(self.hull.turn_angle(s))
        t = np.array([0.0, math.sin(phi), -math.cos(phi)])
        return math.sqrt(max(P[0, 0], 0.0)), math.sqrt(max(float(t @ P @ t), 0.0))

    # ------------------------------------------------------------------ frames
    def footprint(self, c: np.ndarray, A: np.ndarray, exposure_s: float, gain: float) -> tuple[np.ndarray, np.ndarray]:
        """Hull (x, s) of the sampled pixels that would pass the coverage limits, seen from pose (c, A),
        on the hull section extended past its ends (the grid keeps only the real hull)."""
        v = predict.view(self.hull, self.cfg, c, A, self.rays_cam, self.water, bounded=False)
        ok = predict.quality(v, self.cfg, self.water, exposure_s, gain)
        return v['x'][ok], v['s'][ok]

    def process_frame(self, rgb: np.ndarray, exposure_s: float, gain: float) -> dict:
        """Check a frame and add its footprint to the claimed (or unverified) coverage."""
        c, A = self.camera_pose(self.tilt)
        rec = {'cam_pos_est': [round(float(v), 4) for v in c]}
        if float(rgb.std()) < self.cfg['onboard']['dead_frame_std_dn']:
            return {**rec, 'claim': 'black', 'claimed_new': 0}
        r = float(self.hull.raycast(c, -A[:, 2]))
        s_axis = float(self.hull.inverse(c + r * -A[:, 2])[1]) if math.isfinite(r) else float(self.hull.inverse(c)[1])
        sa, sg = self.hull_std(s_axis)
        rec['std_along_girth_m'] = [round(sa, 4), round(sg, 4)]
        x, s = self.footprint(c, A, exposure_s, gain)
        if x.size == 0:
            return {**rec, 'claim': 'below_limits', 'claimed_new': 0}
        lim = self.cfg['onboard']['claim_max_std_m']
        if sa > lim[0] or sg > lim[1]:
            self.unverified.add_points(x, s)
            return {**rec, 'claim': 'uncertain', 'claimed_new': 0}
        k = self.cfg['onboard']['claim_sigma_k']           # claim only what stays covered if the pose is off by k sigma
        self.unverified.add_points(x, s)
        return {**rec, 'claim': 'claimed', 'claimed_new': self.claimed.add_cells(self.claimed.eroded_cells(x, s, k * sa, k * sg))}

    def battery_low(self) -> bool:
        return self.battery < self.cfg['onboard']['battery_reserve_pct']

    def camera_hull_xs(self) -> tuple[float, float]:
        """Hull coordinates where the camera's optical axis meets the hull (its own position's projection
        when it does not)."""
        c, A = self.camera_pose(self.tilt)
        t = float(np.nan_to_num(self.hull.raycast(c, -A[:, 2]), posinf=0.0))
        x, s, _ = self.hull.inverse(c + t * -A[:, 2])
        return float(x), float(s)

    # ------------------------------------------------------------------ holes and revisits
    def can_claim(self) -> bool:
        """Whether the current position uncertainty is within the claim limits."""
        c, A = self.camera_pose(self.tilt)
        sa, sg = self.hull_std(float(self.hull.inverse(c)[1]))
        lim = self.cfg['onboard']['claim_max_std_m']
        return sa <= lim[0] and sg <= lim[1]

    def holes(self) -> list[dict]:
        """Clusters of unclaimed cells (holes closer than onboard.hole_merge_m merge), with hull bounds."""
        unseen = ~self.claimed.covered()
        if not unseen.any():
            return []
        g = self.claimed
        k = math.ceil(self.cfg['onboard']['hole_merge_m'] / g.cell / 2)
        grown = unseen
        if k > 0:
            grown = np.asarray(Image.fromarray(unseen.astype(np.uint8) * 255).filter(ImageFilter.MaxFilter(2 * k + 1))) > 0
        out = []
        for comp in _components(grown):
            rr, cc = np.nonzero(comp & unseen)
            out.append({'x': [round(cc.min() * g.cell, 4), round((cc.max() + 1) * g.cell, 4)],
                        's': [round(g.g0 + rr.min() * g.cell, 4), round(g.g0 + (rr.max() + 1) * g.cell, 4)],
                        'cells': int(rr.size)})
        return out

    def revisit_legs(self, round_k: int) -> tuple[list[Leg], list[dict]]:
        """Revisit lanes for every hole, nearest first, joined by transits at the standoff."""
        holes = self.holes()
        if not holes:
            return [], []
        if not self.can_claim():
            for h in holes:
                h['skipped'] = 'position uncertainty above onboard.claim_max_std_m'
            return [], holes
        cfg, hull = self.cfg, self.hull
        sv, v, o = cfg['survey'], cfg['vehicle'], cfg['onboard']
        if self._strips is None:
            self._strips = strips(hull, cfg)
        cx, cs = self.camera_hull_xs()
        phase = f'revisit_{round_k}'
        legs, todo = [], list(holes)
        while todo:
            h = min(todo, key=lambda h: math.hypot(h['x'][0] - cx, (h['s'][0] + h['s'][1]) / 2 - cs))
            todo.remove(h)
            try:
                girths = band_lanes(hull, cfg, self._strips, *h['s'])
            except ConfigError as e:
                h['skipped'] = str(e)
                continue
            xa = max(h['x'][0] - o['revisit_margin_m'], -sv['end_margin_m'])
            xb = min(h['x'][1] + o['revisit_margin_m'], hull.length + sv['end_margin_m'])
            for g in girths:
                x0, x1 = (xa, xb) if abs(cx - xa) <= abs(cx - xb) else (xb, xa)
                legs.append(Leg('transit', phase, [cx, x0], [cs, g], v['transit_speed_mps']))
                legs.append(Leg('lane', phase, [x0, x1], [g, g], v['speed_mps'], lane=None,
                                capture_x=[float(x) for x in lane_xs(x0, x1, capture_step(cfg))],
                                edge_search=not 0.0 <= x1 <= hull.length))
                cx, cs = x1, g
        return legs, holes


def _components(mask: np.ndarray) -> list[np.ndarray]:
    """4-connected components of a boolean grid."""
    lab = np.zeros(mask.shape, np.int32)
    n = 0
    H, W = mask.shape
    for r0, c0 in zip(*np.nonzero(mask)):
        if lab[r0, c0]:
            continue
        n += 1
        lab[r0, c0] = n
        q = deque([(r0, c0)])
        while q:
            r, c = q.popleft()
            for rr, cc in ((r + 1, c), (r - 1, c), (r, c + 1), (r, c - 1)):
                if 0 <= rr < H and 0 <= cc < W and mask[rr, cc] and not lab[rr, cc]:
                    lab[rr, cc] = n
                    q.append((rr, cc))
    return [lab == k for k in range(1, n + 1)]
