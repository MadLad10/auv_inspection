"""Coverage survey plan: back-and-forth lanes at constant girth and a fixed standoff from the hull.

The inspection camera sits on a tilt unit and faces the local hull normal, so it pitches up as the
lanes go round the bilge while the vehicle stays level. Spacing follows the *quality footprint*: the
part of the image whose pixels would pass the referee's limits (range, incidence, pixel footprint, and
the contrast-to-noise ratio of a faint defect under the vehicle's lamps in the scenario's water) when
facing the flat side at the standoff, reduced by the requested overlaps. Each candidate lane's
covered strip is then evaluated on the curved hull with the same physics (predict.py), midway
between frames.
"""
from __future__ import annotations

import functools
import json
import math

import numpy as np

from . import predict
from .camera import axes, focal_px, fovy_deg, pixel_rays
from .config import ConfigError
from .hull import HullGeometry


def camera_pose(hull: HullGeometry, cfg: dict, x: float, s: float) -> tuple[np.ndarray, np.ndarray]:
    """Inspection camera on the hull normal at (x, s), at the standoff, tilted to face the hull."""
    C = hull.point(x, s) + cfg['survey']['standoff_m'] * hull.normal(s)
    return C, axes(cfg['cameras']['rig'][cfg['survey']['camera']]['look'], float(hull.turn_angle(s)))


PLAN_SECTIONS = ('schema_version', 'hull', 'water', 'cameras', 'optics', 'coverage', 'survey')


def _key(cfg: dict) -> str:
    """The configuration a plan depends on (caches stay valid across run settings like faults)."""
    return json.dumps({k: cfg[k] for k in PLAN_SECTIONS}, sort_keys=True)


def footprint(cfg: dict) -> tuple[float, float]:
    """Quality footprint (along the image width, along the image height) at the standoff (m): the
    largest centred rectangle with the image's aspect whose pixels all pass, facing the flat side."""
    return _footprint(_key(cfg))


@functools.lru_cache(maxsize=32)
def _footprint(cfg_json: str) -> tuple[float, float]:
    cfg = json.loads(cfg_json)
    hull, cams, d = HullGeometry.from_config(cfg), cfg['cameras'], cfg['survey']['standoff_m']
    C, A = camera_pose(hull, cfg, hull.length / 2, hull.side_depth / 2)
    step = cfg['optics']['backscatter_grid_px']
    rays = pixel_rays(cams)[step // 2::step, step // 2::step]
    t, gain = predict.exposure(hull, cfg, C, A, cfg['water'])
    ok = predict.quality(predict.view(hull, cfg, C, A, rays, cfg['water']), cfg, cfg['water'], t, gain)
    th, tv = math.tan(math.radians(cams['hfov_deg']) / 2), math.tan(math.radians(fovy_deg(cams)) / 2)
    reach = np.maximum(np.abs(rays[..., 0]) / th, np.abs(rays[..., 1]) / tv)   # rectangle scale through each pixel
    if not ok[reach <= reach.min() + 1e-9].all():
        raise ConfigError(f'survey.standoff_m={d}: the image centre fails the coverage limits')
    scale = float(min(1.0, reach[~ok].min())) if (~ok).any() else 1.0
    return 2 * d * scale * th, 2 * d * scale * tv


def passes(hull: HullGeometry, cfg: dict, s_lane: float, dx: float, s_pts: np.ndarray) -> np.ndarray:
    """Whether surface points (dx along the hull from the camera, girth s_pts) pass every coverage limit
    for the inspection camera of a lane at girth s_lane (exact on the curved hull, in the scenario's
    water, with the exposure the camera would choose there)."""
    cams = cfg['cameras']
    f = focal_px(cams)
    x0 = hull.length / 2
    C, A = camera_pose(hull, cfg, x0, s_lane)
    pc = (hull.point(np.full_like(s_pts, x0 + dx), s_pts) - C) @ A
    depth = -pc[:, 2]
    in_view = depth > 1e-6
    q = np.stack([pc[:, 0], pc[:, 1], -np.ones_like(depth)], 1) / np.where(in_view, depth, 1.0)[:, None]
    q[:, 2] = -1.0
    in_view &= (np.abs(f * q[:, 0]) <= cams['width'] / 2) & (np.abs(f * q[:, 1]) <= cams['height'] / 2)
    t, gain = predict.exposure(hull, cfg, C, A, cfg['water'])
    ok = np.zeros(len(s_pts), bool)
    if in_view.any():
        v = predict.view(hull, cfg, C, A, q[in_view], cfg['water'])
        # the ray must reach the intended point (not another part of the hull first)
        same = np.abs(v['s'] - s_pts[in_view]) < cfg['survey']['plan_resolution_m']
        ok[in_view] = predict.quality(v, cfg, cfg['water'], t, gain) & same
    return ok


def lane_interval(hull: HullGeometry, cfg: dict, s_lane: float, half_step: float) -> tuple[float, float] | None:
    """Contiguous girth strip round s_lane covered at full quality, also midway between frames."""
    res = cfg['survey']['plan_resolution_m']
    s = np.arange(0.0, hull.girth + res / 2, res)
    ok = passes(hull, cfg, s_lane, half_step, s)
    k = int(round(s_lane / res))
    if not (0 <= k < s.size and ok[k]):
        return None
    lo = k
    while lo > 0 and ok[lo - 1]:
        lo -= 1
    hi = k
    while hi < s.size - 1 and ok[hi + 1]:
        hi += 1
    return float(s[lo]), float(s[hi])


def strips(hull: HullGeometry, cfg: dict) -> list[tuple[float, float, float]]:
    """(lane girth, strip start, strip end) for every candidate lane girth that has a strip."""
    return _strips(json.dumps(hull.to_dict(), sort_keys=True), _key(cfg))


@functools.lru_cache(maxsize=16)
def _strips(hull_json: str, cfg_json: str) -> list[tuple[float, float, float]]:
    hull, cfg = HullGeometry(**json.loads(hull_json)), json.loads(cfg_json)
    sv = cfg['survey']
    half = capture_step(cfg) / 2
    res = sv['plan_resolution_m']
    # only lanes whose image can reach the inspection band (plus its margin) are candidates
    g0, g1 = cfg['hull']['inspect_girth_m']
    reach = footprint(cfg)[1] + sv['band_margin_m']
    cand = np.arange(max(0.0, g0 - reach), min(hull.girth, g1 + reach) + res / 2, res)
    return [(float(s), i[0], i[1]) for s in cand for i in [lane_interval(hull, cfg, float(s), half)] if i is not None]


def cover_band(cands: list[tuple[float, float, float]], g0: float, g1: float, overlap: float) -> list[float]:
    """Greedy lanes over the girth band [g0, g1]: each lane is the deepest one whose strip still overlaps
    the previous strip by `overlap` of that strip's width; the first strip must reach g0."""
    lanes, reach = [], g0
    while True:
        need = reach if not lanes else reach - overlap * (lanes[-1][2] - lanes[-1][1])
        ok = [c for c in cands if c[1] <= need + 1e-9 and c[2] > reach]
        if not ok:
            raise ConfigError(f'cannot cover the inspection band beyond girth {reach:.3f} m with these limits')
        lanes.append(max(ok))
        reach = lanes[-1][2]
        if reach >= g1 - 1e-9:
            return [round(float(l[0]), 6) for l in lanes]


def band_lanes(hull: HullGeometry, cfg: dict, cands, g0: float, g1: float) -> list[float]:
    """Lanes over [g0, g1] widened by survey.band_margin_m on both sides (within the hull's girth)."""
    m = cfg['survey']['band_margin_m']
    return cover_band(cands, max(0.0, g0 - m), min(hull.girth, g1 + m), cfg['survey']['overlap_across'])


def lane_girths(hull: HullGeometry, cfg: dict) -> np.ndarray:
    """Lanes over the whole inspection band (see cover_band and band_lanes)."""
    return np.array(band_lanes(hull, cfg, strips(hull, cfg), *cfg['hull']['inspect_girth_m']))


def capture_step(cfg: dict) -> float:
    """Distance between frames along a lane."""
    return footprint(cfg)[0] * (1 - cfg['survey']['overlap_along'])


def lane_xs(x0: float, x1: float, step: float) -> np.ndarray:
    """Evenly spaced frame positions from x0 to x1 at most `step` apart."""
    return np.linspace(x0, x1, max(2, math.ceil(abs(x1 - x0) / step) + 1))


def lane_plan(hull: HullGeometry, cfg: dict) -> list[dict]:
    sv = cfg['survey']
    cam = sv['camera']
    offset = np.asarray(cfg['cameras']['rig'][cam]['offset'], float)
    xs = lane_xs(-sv['end_margin_m'], hull.length + sv['end_margin_m'], capture_step(cfg))
    plan = []
    for lane, s in enumerate(lane_girths(hull, cfg)):
        tilt = math.degrees(float(hull.turn_angle(s)))
        n = hull.normal(s)
        for x in (xs if lane % 2 == 0 else xs[::-1]):
            cam_pos = hull.point(x, s) + sv['standoff_m'] * n
            plan.append({
                'index': len(plan), 'lane': lane, 'x': round(float(x), 4), 's': round(float(s), 4),
                'rig_position': [round(float(v), 6) for v in cam_pos - offset],
                'yaw': 0.0, 'tilt_deg': {cam: round(tilt, 4)},
            })
    return plan
