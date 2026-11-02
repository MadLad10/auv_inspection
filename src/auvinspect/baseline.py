"""A classical baseline detector: colour and brightness anomalies against the hull paint.

No learning and no downloads. It gives learned detectors a floor to beat, scored by the same referee
(perception.py). Per frame:
  1. hull pixels: brighter than baseline.dark_level (open water and the distant seabed are dark);
  2. flat field: a 2-D polynomial in log brightness per channel, refitted without outliers, models the
     lamp hot spot, vignetting and slow colour changes; defects, seams and fittings do not bend it.
     The residual is close to each surface's log albedo relative to the paint, since the water
     attenuates both alike;
  3. anomaly: the (lightly smoothed) residual in units of its median absolute deviation over the
     hull, whose root-mean-square over the channels exceeds baseline.threshold_mad;
  4. clean-up: opening and closing, then connected components of at least baseline.min_area_px,
     not lying mostly along the hull's edge against open water. Dark components that are thin lines
     along an image axis or span baseline.seam_length_share of the image are weld seams; dark compact
     ones are gratings; neither is reported. A detection lying mostly inside a larger one (barnacles
     in a fouling patch, fragments) is folded into it;
  5. class from the mean residual (R, G, B, in median deviations) and the shape:
       dark and thin                     crack
       less red, greener                 biofouling
       redder, greener, blue unchanged   corrosion
       brighter, a filled rectangle      anode (very bright) or missing-anode footprint
       brighter otherwise                coating damage
The score is the component's mean anomaly strength squashed into (0, 1).
"""
from __future__ import annotations

import json
import os

import numpy as np
from PIL import Image
from scipy import ndimage


def illumination(logc: np.ndarray, hull: np.ndarray, b: dict) -> np.ndarray:
    """Robust 2-D polynomial fit of log brightness per channel (the lamps, vignetting, slow changes)."""
    H, W = hull.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    deg = b['flat_field_degree']
    terms = [u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)]
    st = b['fit_step_px']
    A = np.stack([t[::st, ::st].ravel() for t in terms], 1)
    keep = hull[::st, ::st].ravel()
    out = np.empty_like(logc)
    for c in range(3):
        y = logc[::st, ::st, c].ravel()
        m = keep.copy()
        for _ in range(b['fit_iterations']):
            coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            r = y - A @ coef
            m = keep & (np.abs(r) < b['fit_reject_mad'] * 1.4826 * np.median(np.abs(r[m])))
        out[..., c] = sum(cf * t for cf, t in zip(coef, terms))
    return out


def detect(rgb: np.ndarray, cfg: dict) -> list[dict]:
    b = cfg['baseline']
    img = rgb.astype(np.float32) / 255.0
    H, W = img.shape[:2]
    hull = img.mean(-1) > b['dark_level']
    if hull.mean() < 0.05:
        return []
    logc = np.log(np.maximum(img, 1e-3))
    dev = logc - illumination(logc, hull, b)
    if b['smooth_px']:
        dev = np.stack([ndimage.gaussian_filter(dev[..., c], b['smooth_px']) for c in range(3)], -1)
    mad = 1.4826 * np.median(np.abs(dev[hull] - np.median(dev[hull], 0)), axis=0) + 1e-6
    dev = (dev - np.median(dev[hull], 0)) / mad                    # in median deviations from the paint
    z = np.sqrt((dev ** 2).mean(-1))
    mask = hull & (z > b['threshold_mad'])
    if b['open_px']:
        mask = ndimage.binary_opening(mask, iterations=b['open_px'])
    if b['close_px']:
        mask = ndimage.binary_closing(mask, iterations=b['close_px'])
    lab, n = ndimage.label(mask)
    near_edge = ndimage.distance_transform_edt(hull) < b['edge_px']
    out = []
    for k, sl in enumerate(ndimage.find_objects(lab), start=1):
        comp = lab[sl] == k
        area = int(comp.sum())
        if area < b['min_area_px'] or near_edge[sl][comp].mean() > 0.5:
            continue
        h, wd = sl[0].stop - sl[0].start, sl[1].stop - sl[1].start
        dr, dg, db = dev[sl][comp].mean(0)
        bright = (dr + dg + db) / 3
        fill = area / (h * wd)
        thin = min(h, wd) <= 8 or fill < 0.35
        if bright < 0:                                            # darker than the paint
            line = min(h, wd) <= b['seam_width_px'] and max(h, wd) >= b['seam_aspect'] * min(h, wd)
            if line or max(h, wd) >= b['seam_length_share'] * max(H, W) or not thin:
                continue                                          # weld seam or grating
            cls = 'crack'
        elif dr < 0 and dg > 0:
            cls = 'biofouling'
        elif dr > 0 and dg > 0 and db < 0.4 * dg:
            cls = 'corrosion'
        elif fill > b['rectangle_fill']:
            cls = 'anode' if dg > b['anode_green_mad'] else 'missing_anode'
        else:
            cls = 'coating_damage'
        score = float(1 - np.exp(-z[sl][comp].mean() / b['threshold_mad']))
        out.append({'class': cls, 'score': round(score, 4), 'box': [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)],
                    'area_px': area})
    return _fold_contained(out, b['containment'])


def _fold_contained(dets: list[dict], share: float) -> list[dict]:
    """Drop detections lying mostly inside a larger kept detection."""
    def area(d):
        return (d['box'][2] - d['box'][0]) * (d['box'][3] - d['box'][1])
    kept = []
    for d in sorted(dets, key=area, reverse=True):
        x0, y0, x1, y1 = d['box']
        inside = any(max(0, min(x1, k['box'][2]) - max(x0, k['box'][0])) * max(0, min(y1, k['box'][3]) - max(y0, k['box'][1]))
                     >= share * area(d) for k in kept)
        if not inside:
            kept.append(d)
    return kept


def run(dataset_dir: str, cfg: dict, split: str | None = None) -> dict:
    """Detections for every frame of a dataset (or one split)."""
    with open(os.path.join(dataset_dir, 'annotations.json')) as f:
        ann = json.load(f)
    fids = ann['splits'][split] if split else list(ann['frames'])
    return {fid: detect(np.asarray(Image.open(os.path.join(dataset_dir, ann['frames'][fid]['image'])).convert('RGB')), cfg)
            for fid in fids}
