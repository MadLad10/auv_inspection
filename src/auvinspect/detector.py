"""A learned class for the classical baseline's candidate regions.

The baseline finds candidate regions well (anomalies against the run's own paint), but its
hand-written class rules were written for red-brown paint. Here each region (baseline.components) is
described by measures that do not depend on the paint, and by the paint itself, and a small neural
network (numpy; detector.hidden) gives it a class or `none` (seams, gratings, glare, noise).

Features: the region's median colour relative to the paint (log, and in median deviations on a
signed log scale); the paint's log albedo (the hull's paint scheme, from the hull plan) and the
region's estimated log albedo (their sum); the colour of the hull in a ring detector.context_px wide
around the region, relative to the paint, and the region's contrast with it; the mean anomaly, the
pixel count, fill, elongation, smaller side, larger side as a share of the image, the share lying
near the hull's edge and the share of its ring that is fouling-coloured.

Training (simulation only): regions from the frames of training-split hulls, labelled from the
per-pixel ground truth: the class of the item with the most pixels in the region when it covers at
least detector.min_label_share of the region, or the region holds at least that share of the item's
pixels in the frame (a crack a few millimetres wide never covers half of the region around it), else
none. Classes are weighted towards balance (weight = frequency^-class_weight_power).
Full-batch Adam on the weighted cross-entropy with an L2 penalty, from a seeded initialisation: the
same data give the same model. The model records the settings its features depend on and the hulls
it was trained on, and is refused under other settings.

A region is reported when its most probable class is not none, with that probability as its score.
"""
from __future__ import annotations

import hashlib
import json
import os
from importlib.resources import files

import numpy as np
from PIL import Image

from . import baseline
from .rng import stream


def classes(cfg: dict) -> list[str]:
    return list(cfg['perception']['classes']) + ['none']


def _slog(x):
    x = np.asarray(x, float)
    return np.sign(x) * np.log1p(np.abs(x))


def features(c: dict, log_paint: np.ndarray) -> np.ndarray:
    """The measures of one candidate region (see module notes)."""
    H, W = c['image']
    lg, ctx = np.asarray(c['log'], float), np.asarray(c['context_log'], float)
    h, w = c['h'], c['w']
    shape = [np.log1p(c['z']), np.log(c['area_px']), c['fill'], np.log(max(h, w) / min(h, w)), np.log(min(h, w)),
             max(h / H, w / W), c['edge_share'], c['fouling_ring']]
    return np.concatenate([lg, _slog(c['dev']), log_paint, lg + log_paint, ctx, lg - ctx, shape])


def settings(cfg: dict) -> dict:
    """The settings the features depend on: a model is used only with the same."""
    d = cfg['detector']
    return {'baseline': cfg['baseline'], 'context_px': d['context_px'], 'min_label_share': d['min_label_share'],
            'classes': classes(cfg), 'labels': 'share of the region or of the item'}


def _digest(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def _spec(run_dir: str) -> dict:
    with open(os.path.join(os.path.dirname(os.path.normpath(run_dir)), 'scenario.json')) as f:
        return json.load(f)


def regions(run_dir: str, cfg: dict) -> tuple[np.ndarray, np.ndarray]:
    """Features and ground-truth labels (indices into `classes`) of the candidate regions in a recorded
    run with ground truth, cached in the run directory for the current settings."""
    key = _digest(settings(cfg))[:16]
    cache = os.path.join(run_dir, f'regions_{key}.npz')
    if os.path.exists(cache):
        z = np.load(cache)
        return z['X'], z['y']
    spec = _spec(run_dir)
    names = classes(cfg)
    item_class = {it['id']: it['class'] for it in spec['items']}
    log_paint = np.log(np.asarray(spec['config']['appearance']['paint_rgb'], float))
    recs = [r for r in (json.loads(line) for line in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera'] and r.get('signal', True)]

    def image(r):
        return np.asarray(Image.open(os.path.join(run_dir, 'frames', r['frame'] + '.png')).convert('RGB'))
    ref = baseline.paint_reference([image(r) for r in recs[::cfg['baseline']['reference_stride']]], cfg)
    X, y = [], []
    for r in recs:
        truth = np.load(os.path.join(run_dir, 'gt', r['frame'] + '.npz'))['label']
        item_px = dict(zip(*np.unique(truth[truth >= 0], return_counts=True)))
        share = cfg['detector']['min_label_share']
        for c in baseline.components(image(r), cfg, ref, None, cfg['detector']['context_px']):
            x0, y0, x1, y1 = c['box']
            inside = truth[y0:y1, x0:x1][c['mask']]
            ids, counts = np.unique(inside[inside >= 0], return_counts=True)
            cls = None
            if len(ids):
                top = int(np.argmax(counts))
                if counts[top] >= share * c['area_px'] or counts[top] >= share * item_px[ids[top]]:
                    cls = item_class.get(int(ids[top]))
            X.append(features(c, log_paint))
            y.append(names.index(cls) if cls in names else names.index('none'))
    X, y = np.asarray(X, float).reshape(-1, 26), np.asarray(y, int)
    np.savez(cache, X=X, y=y)
    return X, y


# ------------------------------------------------------------------------------------- the network
def _forward(params: list, X: np.ndarray) -> tuple[np.ndarray, list]:
    acts = [X]
    for W, b in params[:-1]:
        acts.append(np.maximum(0.0, acts[-1] @ W + b))
    W, b = params[-1]
    return acts[-1] @ W + b, acts


def _softmax(z: np.ndarray) -> np.ndarray:
    e = np.exp(z - z.max(1, keepdims=True))
    return e / e.sum(1, keepdims=True)


def fit(X: np.ndarray, y: np.ndarray, n_classes: int, cfg: dict) -> dict:
    """The network's weights for standardised features (see module notes)."""
    d = cfg['detector']
    mean, std = X.mean(0), X.std(0) + 1e-6
    Xs = (X - mean) / std
    freq = np.bincount(y, minlength=n_classes).astype(float)
    cw = np.where(freq > 0, (np.maximum(freq, 1) / freq.sum()) ** -d['class_weight_power'], 0.0)
    w = cw[y] / cw[y].mean()
    rng = stream(0, 'detector_init')
    sizes = [X.shape[1], *d['hidden'], n_classes]
    params = [[rng.normal(0, np.sqrt(2 / a), (a, b)), np.zeros(b)] for a, b in zip(sizes[:-1], sizes[1:])]
    m = [[np.zeros_like(p) for p in layer] for layer in params]
    v = [[np.zeros_like(p) for p in layer] for layer in params]
    onehot = np.eye(n_classes)[y]
    b1, b2, lr = 0.9, 0.999, d['learning_rate']
    for t in range(1, d['iterations'] + 1):
        logits, acts = _forward(params, Xs)
        g = (_softmax(logits) - onehot) * w[:, None] / len(y)        # weighted cross-entropy gradient
        grads = []
        for k in range(len(params) - 1, -1, -1):
            W, _ = params[k]
            grads.append([acts[k].T @ g + d['l2'] * W, g.sum(0)])
            if k:
                g = (g @ W.T) * (acts[k] > 0)
        grads.reverse()
        for k, layer in enumerate(params):
            for j in range(2):
                m[k][j] = b1 * m[k][j] + (1 - b1) * grads[k][j]
                v[k][j] = b2 * v[k][j] + (1 - b2) * grads[k][j] ** 2
                layer[j] -= lr * (m[k][j] / (1 - b1 ** t)) / (np.sqrt(v[k][j] / (1 - b2 ** t)) + 1e-8)
    return {'mean': mean.tolist(), 'std': std.tolist(), 'layers': [[W.tolist(), b.tolist()] for W, b in params]}


def predict(net: dict, X: np.ndarray) -> np.ndarray:
    """Class probabilities of feature rows."""
    params = [[np.asarray(W), np.asarray(b)] for W, b in net['layers']]
    return _softmax(_forward(params, (np.asarray(X, float) - np.asarray(net['mean'])) / np.asarray(net['std']))[0])


def train(runs: list[str], cfg: dict) -> dict:
    """A region classifier from recorded runs of training hulls, with its provenance."""
    names = classes(cfg)
    Xs, ys, sources = [], [], []
    for run in runs:
        X, y = regions(run, cfg)
        spec = _spec(run)
        Xs.append(X)
        ys.append(y)
        sources.append({'run': run, 'seed': spec['seed'], 'scenario_digest': spec['digest'],
                        'paint_rgb': spec['config']['appearance']['paint_rgb'], 'regions': int(len(y))})
    X, y = np.concatenate(Xs), np.concatenate(ys)
    return {'kind': 'region classifier: ReLU network on region measures, weighted cross-entropy', 'classes': names,
            'hidden': cfg['detector']['hidden'], 'context_px': cfg['detector']['context_px'], **fit(X, y, len(names), cfg),
            'regions': {c: int((y == k).sum()) for k, c in enumerate(names)},
            'settings_sha256': _digest(settings(cfg)), 'trained_on': sources}


def load_model(cfg: dict, path: str | None = None) -> dict | None:
    """The region classifier (`path`, else detector.model: a file, or a name in the package's models
    directory; None when that is null), checked against the settings."""
    name = path or cfg['detector']['model']
    if name is None:
        return None
    text = open(name).read() if os.path.isfile(name) else files('auvinspect').joinpath('models', name).read_text()
    model = json.loads(text)
    if model['settings_sha256'] != _digest(settings(cfg)):
        raise ValueError('the region classifier was trained with other settings (baseline or detector); '
                         'retrain it with `auvinspect train-detector`')
    return model


class Classifier:
    """The region classifier for one hull's paint, as baseline.detect uses it."""

    def __init__(self, model: dict, paint_rgb):
        self.model = model
        self.context_px = model['context_px']
        self.log_paint = np.log(np.asarray(paint_rgb, float))

    def classify(self, comps: list[dict]) -> list[tuple[str | None, float]]:
        if not comps:
            return []
        p = predict(self.model, np.stack([features(c, self.log_paint) for c in comps]))
        names = self.model['classes']
        out = []
        for row in p:
            k = int(np.argmax(row))
            out.append((None if names[k] == 'none' else names[k], float(row[k])))
        return out


def for_run(run_dir: str, cfg: dict, model: dict | None = None) -> Classifier | None:
    """The classifier for a recorded run's hull (its scenario's paint), or None for the rules."""
    model = model or load_model(cfg)
    if model is None:
        return None
    return Classifier(model, _spec(run_dir)['config']['appearance']['paint_rgb'])
