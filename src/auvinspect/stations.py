"""Anode stations: the hull plan's anode positions, each checked in the frames that see it.

The hull plan (hull.anode_stations: the vessel's drawings) says where every anode should be, so the
question at a station is not whether something is there but what is fitted now: intact, depleted or
missing. A station no frame saw is `not_inspected`, never missing.

Views: an inspection frame views a station when, from the camera pose the vehicle estimated, the
station centre is within the coverage referee's range and incidence limits and the station window
(the plan footprint grown by stations.window_margin_m) lies wholly in the image. On the detector's
paint-relative residual (baseline.residual, with the run's paint reference) a view measures
  the area of the window's off-paint pixels (z above baseline.threshold_mad), from their share of
  the window's pixels;
  the median log colour, relative to the paint, of those pixels and of the plan footprint.
Anode metal is grey and bright (an intact anode saturates the camera); a depleted anode is smaller
and duller; a missing anode leaves a footprint of unfouled undercoat and two bolts.

Classifier: per paint scheme (appearance.paint_rgb, part of the hull plan), a linear discriminant (a
mean per status, one pooled covariance) fitted on the views of training hulls with that paint, whose
station status the scenarios give (simulation only; `train`). Colours relative to the paint do not
carry over between paints (the slime film over a dark paint changes its colour far more than over a
light one), so a hull whose paint has no model gets `unclear` stations, never a guess. A station's
status is the one with the highest log likelihood summed over its views; its confidence is the
posterior from the mean log likelihood (views of one station are not independent). A station whose
views lie far from the status they chose (median squared Mahalanobis distance above
stations.max_distance2) looks like nothing the model has seen: it is `unclear`, for an inspector,
rather than given a confident guess. The model records the settings its features depend on and the
scenarios it was trained on, and a model trained with other settings is refused.
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files

import numpy as np
from PIL import Image, ImageDraw

from . import baseline
from .camera import focal_px
from .hull import HullGeometry, anode_stations
from .mapping import camera_pose

STATUSES = ('intact', 'depleted', 'missing')
FEATURES = ('off_paint_area_m2', 'off_paint_log_r', 'off_paint_log_g', 'off_paint_log_b',
            'footprint_log_r', 'footprint_log_g', 'footprint_log_b')


def window(st: dict, cfg: dict, glare: bool = False) -> list[float]:
    """A station's window on the hull: its plan footprint grown by stations.window_margin_m (and by
    stations.glare_margin_m for the zone the detector leaves to the station check)."""
    m = cfg['stations']['window_margin_m'] + (cfg['stations']['glare_margin_m'] if glare else 0.0)
    x0, s0, x1, s1 = st['box']
    return [x0 - m, s0 - m, x1 + m, s1 + m]


def _outline(box: list[float], n: int = 8) -> np.ndarray:
    x0, s0, x1, s1 = box
    t = np.linspace(0, 1, n, endpoint=False)
    return np.concatenate([np.stack([x0 + (x1 - x0) * t, np.full(n, s0)], 1), np.stack([np.full(n, x1), s0 + (s1 - s0) * t], 1),
                           np.stack([x1 - (x1 - x0) * t, np.full(n, s1)], 1), np.stack([np.full(n, x0), s1 - (s1 - s0) * t], 1)])


def _project(box: list[float], hull: HullGeometry, C: np.ndarray, A: np.ndarray, f: float, W: int, H: int) -> np.ndarray | None:
    """Image outline (u, v) of a box on the hull; None when any of it is behind the camera."""
    pts = _outline(box)
    pc = (hull.point(pts[:, 0], pts[:, 1]) - C) @ A
    if np.any(pc[:, 2] >= 0):
        return None
    return np.stack([f * pc[:, 0] / -pc[:, 2] + W / 2, -f * pc[:, 1] / -pc[:, 2] + H / 2], 1)


def _fill(uv: np.ndarray, W: int, H: int) -> np.ndarray:
    im = Image.new('L', (W, H), 0)
    ImageDraw.Draw(im).polygon([tuple(p) for p in uv], fill=1)
    return np.asarray(im, bool)


def _spec(run_dir: str) -> dict:
    with open(os.path.join(os.path.dirname(os.path.normpath(run_dir)), 'scenario.json')) as f:
        return json.load(f)


def measure(run_dir: str, cfg: dict) -> list[dict]:
    """Every view of every station in a recorded run (inside its scenario directory)."""
    spec = _spec(run_dir)
    hull = HullGeometry(**spec['hull'])
    plan = anode_stations(spec['config'])
    cams, lim, b = cfg['cameras'], cfg['coverage'], cfg['baseline']
    f, W, H = focal_px(cams), cams['width'], cams['height']
    recs = [r for r in (json.loads(line) for line in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera'] and r.get('signal', True)]

    def image(r):
        return np.asarray(Image.open(os.path.join(run_dir, 'frames', r['frame'] + '.png')).convert('RGB'))
    ref = baseline.paint_reference([image(r) for r in recs[::b['reference_stride']]], cfg)
    views = []
    for r in recs:
        C, A = camera_pose(r)
        res = None
        for st in plan:
            ray = hull.point(st['x'], st['s']) - C
            rng = float(np.linalg.norm(ray))
            inc = float(np.degrees(np.arccos(min(1.0, abs(float(ray @ hull.normal(st['s']))) / rng))))
            if rng > lim['max_range_m'] or inc > lim['max_incidence_deg']:
                continue
            uv = _project(window(st, cfg), hull, C, A, f, W, H)
            if uv is None or uv.min() < 0 or uv[:, 0].max() > W - 1 or uv[:, 1].max() > H - 1:
                continue
            if res is None:
                res = baseline.residual(image(r), cfg, ref)
                if res is None:
                    break
            win = _fill(uv, W, H)
            foot = _fill(_project(st['box'], hull, C, A, f, W, H), W, H)
            off = win & res['hull'] & (res['z'] > b['threshold_mad'])
            x0, s0, x1, s1 = window(st, cfg)
            area = (x1 - x0) * (s1 - s0) * off.sum() / max(int(win.sum()), 1)
            box = [int(uv[:, 0].min()), int(uv[:, 1].min()), int(np.ceil(uv[:, 0].max())), int(np.ceil(uv[:, 1].max()))]
            feats = [area, *(np.median(res['log'][off], 0) if off.any() else np.zeros(3)), *np.median(res['log'][foot], 0)]
            views.append({'station': st['id'], 'frame': r['frame'], 'range_m': round(rng, 3), 'incidence_deg': round(inc, 1),
                          'box': box, 'offcentre_px': round(float(np.hypot((box[0] + box[2]) / 2 - W / 2, (box[1] + box[3]) / 2 - H / 2)), 1),
                          'features': [round(float(v), 6) for v in feats]})
    return views


def window_masks(run_dir: str, cfg: dict) -> dict[str, np.ndarray]:
    """Per frame of a recorded run, the pixels of the planned stations' glare zones (from the
    estimated camera pose), for the detector to leave out: the fitting is the station check's business."""
    spec = _spec(run_dir)
    hull = HullGeometry(**spec['hull'])
    plan = anode_stations(spec['config'])
    cams = cfg['cameras']
    f, W, H = focal_px(cams), cams['width'], cams['height']
    out = {}
    for r in (json.loads(line) for line in open(os.path.join(run_dir, 'frames.jsonl'))):
        if r['camera'] != cfg['survey']['camera']:
            continue
        C, A = camera_pose(r)
        m = np.zeros((H, W), bool)
        for st in plan:
            uv = _project(window(st, cfg, glare=True), hull, C, A, f, W, H)
            if uv is not None and uv[:, 0].max() >= 0 and uv[:, 0].min() < W and uv[:, 1].max() >= 0 and uv[:, 1].min() < H:
                m |= _fill(uv, W, H)
        if m.any():
            out[r['frame']] = m
    return out


# ---------------------------------------------------------------------------------- the classifier
def settings(cfg: dict) -> dict:
    """The settings the features depend on: a model is used only with the same."""
    return {'baseline': cfg['baseline'], 'window_margin_m': cfg['stations']['window_margin_m'], 'features': list(FEATURES),
            'limits': {k: cfg['coverage'][k] for k in ('max_range_m', 'max_incidence_deg')}}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def fit(X: np.ndarray, y: np.ndarray) -> dict:
    """Linear discriminant: a mean per status (y indexes STATUSES) and the pooled covariance."""
    X, y = np.asarray(X, float), np.asarray(y)
    absent = [s for k, s in enumerate(STATUSES) if not (y == k).any()]
    if absent:
        raise ValueError(f'no training views with status {absent}')
    means = np.array([X[y == k].mean(0) for k in range(len(STATUSES))])
    R = X - means[y]
    cov = R.T @ R / (len(X) - len(STATUSES))
    return {'means': means.tolist(), 'covariance': cov.tolist(), 'views': {s: int((y == k).sum()) for k, s in enumerate(STATUSES)}}


def paint_key(paint_rgb) -> str:
    return json.dumps([round(float(c), 4) for c in paint_rgb])


def truth(spec: dict) -> dict[str, dict]:
    """Simulation only: each station's true status and item, from the scenario's items."""
    out = {}
    for st in anode_stations(spec['config']):
        x0, s0, x1, s1 = st['box']
        for it in spec['items']:
            if it['class'] == 'anode':
                xc, sc = it['centroid_xs']
            elif it['class'] == 'missing_anode':
                xc, sc = (it['bbox_xs'][0] + it['bbox_xs'][2]) / 2, (it['bbox_xs'][1] + it['bbox_xs'][3]) / 2
            else:
                continue
            if x0 <= xc <= x1 and s0 <= sc <= s1:
                out[st['id']] = {'status': it.get('status', 'missing'), 'item': it['id']}
    return out


def train(runs: list[str], cfg: dict) -> dict:
    """A station model (a discriminant per paint scheme) from recorded runs of training hulls, with
    its provenance."""
    data: dict[str, tuple[list, list]] = {}
    sources = []
    for run in runs:
        spec = _spec(run)
        true = truth(spec)
        key = paint_key(spec['config']['appearance']['paint_rgb'])
        X, y = data.setdefault(key, ([], []))
        for v in measure(run, cfg):
            X.append(v['features'])
            y.append(STATUSES.index(true[v['station']]['status']))
        sources.append({'run': run, 'seed': spec['seed'], 'scenario_digest': spec['digest'], 'paint_rgb': json.loads(key)})
    paints = {}
    for key, (X, y) in sorted(data.items()):
        try:
            paints[key] = fit(np.asarray(X), np.asarray(y))
        except ValueError as e:
            raise ValueError(f'paint {key}: {e}') from None
    return {'kind': 'linear discriminant per paint scheme (a mean per status, pooled covariance)', 'statuses': list(STATUSES),
            'features': list(FEATURES), 'paints': paints, 'settings_sha256': _digest(settings(cfg)), 'trained_on': sources}


def load_model(cfg: dict, path: str | None = None) -> dict:
    """The station model (default: the packaged one named by stations.model), checked against the settings."""
    text = open(path).read() if path else files('auvinspect').joinpath('models', cfg['stations']['model']).read_text()
    model = json.loads(text)
    if model['settings_sha256'] != _digest(settings(cfg)):
        raise ValueError('the station model was trained with other settings (baseline, station window or coverage limits); '
                         'retrain it with `auvinspect train-stations`')
    return model


def classify(views: list[dict], model: dict, plan: list[dict], max_distance2: float, paint_rgb) -> list[dict]:
    """Each planned station's status, confidence, number of views, distance to that status (median
    squared Mahalanobis over its views) and best view (the most central), with the discriminant of
    the hull's paint; unclear for every viewed station when the paint has none."""
    sub = model['paints'].get(paint_key(paint_rgb))
    mu = np.asarray(sub['means']) if sub else np.zeros((len(STATUSES), len(FEATURES)))
    P = np.linalg.inv(np.asarray(sub['covariance'])) if sub else np.eye(len(FEATURES))
    by: dict[str, list] = {}
    for v in views:
        d = np.asarray(v['features']) - mu
        by.setdefault(v['station'], []).append((v, -0.5 * np.einsum('ki,ij,kj->k', d, P, d)))
    out = []
    for st in plan:
        vs = by.get(st['id'], [])
        if not vs:
            out.append({**st, 'status': 'not_inspected', 'confidence': None, 'views': 0, 'distance2': None, 'evidence': None})
            continue
        ll = np.array([l for _, l in vs])
        k = int(np.argmax(ll.sum(0)))
        mean = ll.mean(0)
        post = np.exp(mean - mean.max())
        d2 = float(np.median(-2 * ll[:, k]))                     # robust to a stray view
        best = min((v for v, _ in vs), key=lambda v: v['offcentre_px'])
        if sub is None:
            out.append({**st, 'status': 'unclear', 'confidence': None, 'views': len(vs), 'distance2': None,
                        'evidence': {'frame': best['frame'], 'box': best['box']}})
            continue
        out.append({**st, 'status': model['statuses'][k] if d2 <= max_distance2 else 'unclear',
                    'confidence': round(float(post[k] / post.sum()), 3), 'views': len(vs), 'distance2': round(d2, 2),
                    'evidence': {'frame': best['frame'], 'box': best['box']}})
    return out


def evaluate(results: list[dict], true: dict[str, dict]) -> dict:
    """Simulation only: the station check against the true statuses (inspected stations; an unclear
    one is neither right nor wrong)."""
    seen = [r for r in results if r['status'] != 'not_inspected']
    confusion: dict[str, dict[str, int]] = {}
    for r in results:
        row = confusion.setdefault(true[r['id']]['status'], {})
        row[r['status']] = row.get(r['status'], 0) + 1
    per = {}
    for s in STATUSES:
        mine = [r for r in seen if true[r['id']]['status'] == s]
        per[s] = {'true': len(mine), 'found': sum(1 for r in mine if r['status'] == s),
                  'false': sum(1 for r in seen if r['status'] == s and true[r['id']]['status'] != s)}
    correct = sum(1 for r in seen if r['status'] == true[r['id']]['status'])
    unclear = sum(1 for r in seen if r['status'] == 'unclear')
    return {'stations': len(results), 'inspected': len(seen), 'correct': correct, 'unclear': unclear,
            'wrong': len(seen) - correct - unclear, 'accuracy': round(correct / len(seen), 4) if seen else None,
            'per_status': per, 'confusion': confusion}


def check(run_dir: str, cfg: dict, model: dict | None = None) -> tuple[list[dict], dict]:
    """Station statuses for a recorded run, and (simulation only) their score."""
    spec = _spec(run_dir)
    plan = anode_stations(spec['config'])
    results = classify(measure(run_dir, cfg), model or load_model(cfg), plan, cfg['stations']['max_distance2'],
                       spec['config']['appearance']['paint_rgb'])
    return results, evaluate(results, truth(spec))
