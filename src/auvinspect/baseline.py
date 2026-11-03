"""A classical baseline detector: colour and brightness anomalies against the hull paint.

No learning and no downloads. It gives learned detectors a floor to beat, scored by the same referee
(perception.py). Per frame:
  1. hull pixels: brighter than baseline.dark_level (open water and the distant seabed are dark);
  2. flat field: a 2-D polynomial in log brightness per channel, refitted without outliers, models the
     lamp hot spot, vignetting and slow colour changes; defects, seams and fittings do not bend it.
     Pixels whose colour lies off the run's paint reference (the median paint colour over the run's
     frames, after the brightness falloff) get their own constant offset in the fit: a frame filled by
     a large defect is still measured against paint, not against its majority, and the defect still
     shapes the falloff where no paint shows. The residual is close to each
     surface's log albedo relative to the paint, since the water attenuates both alike;
  3. anomaly: the (lightly smoothed) residual in units of its median absolute deviation over the
     hull, whose root-mean-square over the channels exceeds baseline.threshold_mad;
  4. clean-up: opening and closing, then connected components of at least baseline.min_area_px,
     not lying mostly along the hull's edge against open water. Dark components that are thin lines
     along an image axis or span baseline.seam_length_share of the image are weld seams; dark compact
     ones are gratings; neither is reported. A detection lying mostly inside a larger one (barnacles
     in a fouling patch, fragments) is folded into it;
  5. class from the median residual (R, G, B, in median deviations) and the shape:
       dark and thin                     crack
       less red, greener                 biofouling
       brighter, ringed by fouling       a barnacle, part of the fouling (not reported)
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


def illumination(logc: np.ndarray, hull: np.ndarray, b: dict, other: np.ndarray | None = None) -> np.ndarray:
    """Robust 2-D polynomial fit of log brightness per channel (the lamps, vignetting, slow changes).
    Pixels in `other` (hull pixels off the paint colour) shape the fit through their own constant offset."""
    H, W = hull.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    u, v = (xx - W / 2) / (W / 2), (yy - H / 2) / (H / 2)
    deg = b['flat_field_degree']
    terms = [u ** i * v ** j for i in range(deg + 1) for j in range(deg + 1 - i)]
    st = b['fit_step_px']
    A = np.stack([t[::st, ::st].ravel() for t in terms], 1)
    keep = hull[::st, ::st].ravel()
    if other is not None and other.any():
        A = np.hstack([A, other[::st, ::st].ravel()[:, None].astype(np.float32)])
    out = np.empty_like(logc)
    for c in range(logc.shape[-1]):
        y = logc[::st, ::st, c].ravel()
        m = keep.copy()
        for _ in range(b['fit_iterations']):
            coef, *_ = np.linalg.lstsq(A[m], y[m], rcond=None)
            r = y - A @ coef
            m = keep & (np.abs(r) < b['fit_reject_mad'] * 1.4826 * np.median(np.abs(r[m])))
        out[..., c] = sum(cf * t for cf, t in zip(coef, terms))         # the paint's falloff (no offset term)
    return out


def _colour(img: np.ndarray, hull: np.ndarray, b: dict) -> np.ndarray:
    """Log colour with the brightness falloff (fitted on log luminance) removed."""
    logc = np.log(np.maximum(img, 1e-3))
    return logc - illumination(logc.mean(-1, keepdims=True), hull, b)


def paint_reference(images: list[np.ndarray], cfg: dict) -> dict:
    """The run's paint colour: medians over frames of each frame's median colour and spread."""
    b = cfg['baseline']
    meds, mads = [], []
    for rgb in images:
        img = rgb.astype(np.float32) / 255.0
        hull = img.mean(-1) > b['dark_level']
        if hull.mean() < 0.2:
            continue
        col = _colour(img, hull, b)[hull]
        med = np.median(col, 0)
        meds.append(med)
        mads.append(1.4826 * np.median(np.abs(col - med), 0))
    if not meds:
        return {}
    return {'median': np.median(meds, 0).tolist(), 'mad': np.median(mads, 0).tolist()}


def residual(rgb: np.ndarray, cfg: dict, reference: dict | None = None) -> dict | None:
    """Log colour relative to the paint after the flat field (steps 1 and 2): 'log' per channel (the
    log albedo ratio to the paint, roughly), 'dev' the same in median deviations of the paint (lightly
    smoothed), 'z' their root-mean-square over the channels, and the 'hull' and 'paint' masks. None
    when the frame shows too little hull."""
    b = cfg['baseline']
    img = rgb.astype(np.float32) / 255.0
    hull = img.mean(-1) > b['dark_level']
    if hull.mean() < 0.05:
        return None
    paint, other = hull, None
    if reference:
        near = np.abs(_colour(img, hull, b) - reference['median']) < b['reference_band_mad'] * np.asarray(reference['mad'])
        if (hull & near.all(-1)).mean() > 0.02:
            paint, other = hull & near.all(-1), hull & ~near.all(-1)
    logc = np.log(np.maximum(img, 1e-3))
    dev = logc - illumination(logc, hull, b, other)
    if b['smooth_px']:
        dev = np.stack([ndimage.gaussian_filter(dev[..., c], b['smooth_px']) for c in range(3)], -1)
    centre = np.median(dev[paint], 0)
    mad = 1.4826 * np.median(np.abs(dev[paint] - centre), axis=0) + 1e-6
    log = dev - centre
    dev = log / mad                                               # in median deviations from the paint
    return {'log': log, 'dev': dev, 'z': np.sqrt((dev ** 2).mean(-1)), 'hull': hull, 'paint': paint}


def components(rgb: np.ndarray, cfg: dict, reference: dict | None = None, exclude: np.ndarray | None = None,
               context_px: int = 0) -> list[dict]:
    """Candidate regions of one frame (steps 1 to 4 before the shape rules): connected anomalies of at
    least baseline.min_area_px not lying mostly along the hull's edge, each with its box, pixel count,
    fill, median colour (in median deviations and in log relative to the paint), mean anomaly, share
    of its fouling-coloured ring (for the barnacle rule) and, when context_px is given, the median log
    colour of the hull in a ring that wide around it. Pixels in `exclude` never join a region."""
    b = cfg['baseline']
    r = residual(rgb, cfg, reference)
    if r is None:
        return []
    dev, z, hull, log = r['dev'], r['z'], r['hull'], r['log']
    H, W = hull.shape
    mask = hull & (z > b['threshold_mad'])
    if exclude is not None:
        mask &= ~exclude
    fouling_colour = (dev[..., 0] < 0) & (dev[..., 1] > 1.0)
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
        edge_share = float(near_edge[sl][comp].mean())
        if area < b['min_area_px'] or edge_share > 0.5:
            continue
        c = {'box': [int(sl[1].start), int(sl[0].start), int(sl[1].stop), int(sl[0].stop)], 'area_px': area,
             'h': sl[0].stop - sl[0].start, 'w': sl[1].stop - sl[1].start, 'edge_share': edge_share,
             'dev': np.median(dev[sl][comp], axis=0), 'log': np.median(log[sl][comp], axis=0),
             'z': float(z[sl][comp].mean()), 'fouling_ring': _ringed_by(comp, fouling_colour[sl], b['barnacle_ring_px']),
             'image': (H, W), 'mask': comp}
        c['fill'] = area / (c['h'] * c['w'])
        if context_px:
            big = (slice(max(0, sl[0].start - context_px), min(H, sl[0].stop + context_px)),
                   slice(max(0, sl[1].start - context_px), min(W, sl[1].stop + context_px)))
            own = lab[big] == k
            ring = ndimage.binary_dilation(own, iterations=context_px) & ~own & hull[big] & ~mask[big]
            c['context_log'] = np.median(log[big][ring], axis=0) if ring.any() else np.zeros(3)
        out.append(c)
    return out


def rule_class(c: dict, b: dict) -> str | None:
    """The hand-written class of a candidate region (step 5), or None for seams, gratings, barnacles."""
    H, W = c['image']
    h, wd, fill = c['h'], c['w'], c['fill']
    dr, dg, db = c['dev']
    bright = (dr + dg + db) / 3
    thin = min(h, wd) <= 8 or fill < 0.35
    if bright < 0:                                                # darker than the paint
        line = min(h, wd) <= b['seam_width_px'] and max(h, wd) >= b['seam_aspect'] * min(h, wd)
        if line or max(h, wd) >= b['seam_length_share'] * max(H, W) or not thin:
            return None                                           # weld seam or grating
        return 'crack'
    if dr < 0 and dg > 0:
        return 'biofouling'
    if c['fouling_ring'] > 0.5:
        return None                                               # a barnacle: part of the fouling around it
    if dr > 0 and dg > 0 and db < 0.4 * dg:
        return 'corrosion'
    if fill > b['rectangle_fill']:
        return 'anode' if dg > b['anode_green_mad'] else 'missing_anode'
    return 'coating_damage'


def detect(rgb: np.ndarray, cfg: dict, reference: dict | None = None, exclude: np.ndarray | None = None,
           classifier=None) -> list[dict]:
    """Detections in one frame; pixels in `exclude` (planned fittings) never join a detection. Classes
    come from the hand-written rules, or from `classifier` (detector.Classifier: a class or none, and
    a score, per region)."""
    b = cfg['baseline']
    comps = components(rgb, cfg, reference, exclude, classifier.context_px if classifier else 0)
    if classifier is not None:
        labelled = classifier.classify(comps)
    else:
        labelled = [(rule_class(c, b), float(1 - np.exp(-c['z'] / b['threshold_mad']))) for c in comps]
    out = [{'class': cls, 'score': round(score, 4), 'box': c['box'], 'area_px': c['area_px']}
           for c, (cls, score) in zip(comps, labelled) if cls is not None]
    return _fold_contained(out, b['containment'])


def _ringed_by(comp: np.ndarray, other: np.ndarray, width: int) -> float:
    """Share of the ring of pixels just outside a component that belong to `other`."""
    pad = width + 1
    c = np.pad(comp, pad)
    ring = ndimage.binary_dilation(c, iterations=width) & ~c
    return float(np.pad(other, pad)[ring].mean()) if ring.any() else 0.0


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


def run(dataset_dir: str, cfg: dict, split: str | None = None, model: dict | None = None) -> dict:
    """Detections for every frame of a dataset (or one split), each run against its own paint reference,
    classed by the region classifier (detector.model, or `model`) for its hull's paint when there is one."""
    from . import detector
    with open(os.path.join(dataset_dir, 'annotations.json')) as f:
        ann = json.load(f)
    fids = ann['splits'][split] if split else list(ann['frames'])
    if any('run' not in ann['frames'][fid] for fid in fids):
        raise ValueError('the classical baseline needs recorded runs (a paint reference from the run, the hull paint '
                         'from its scenario); this dataset has frames without one (real images)')
    by_run: dict[str, list[str]] = {}
    for fid in fids:
        by_run.setdefault(ann['frames'][fid]['run'], []).append(fid)
    model = model or detector.load_model(cfg)
    out = {}
    for run, run_fids in by_run.items():
        images = {fid: np.asarray(Image.open(os.path.join(dataset_dir, ann['frames'][fid]['image'])).convert('RGB')) for fid in run_fids}
        ref = paint_reference([images[f] for f in run_fids[::cfg['baseline']['reference_stride']]], cfg)
        clf = detector.for_run(run, cfg, model) if model else None
        out.update({fid: detect(img, cfg, ref, classifier=clf) for fid, img in images.items()})
    return out


def run_frames(run_dir: str, cfg: dict, exclude: dict | None = None, model: dict | None = None) -> dict:
    """Detections for a recorded run's inspection-camera frames, keyed by frame name; `exclude` maps
    frame names to masks of pixels left out (stations.window_masks). Classes come from the region
    classifier (detector.model, or `model`) when there is one, else from the rules."""
    from . import detector
    clf = detector.for_run(run_dir, cfg, model)
    recs = [r for r in (json.loads(l) for l in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera'] and r.get('signal', True)]
    images = {r['frame']: np.asarray(Image.open(os.path.join(run_dir, 'frames', r['frame'] + '.png')).convert('RGB')) for r in recs}
    ref = paint_reference(list(images.values())[::cfg['baseline']['reference_stride']], cfg)
    return {name: detect(img, cfg, ref, (exclude or {}).get(name), clf) for name, img in images.items()}
