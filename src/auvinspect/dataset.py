"""Labelled datasets from recorded frames, for training and scoring perception.

A run directory (a survey or a mission inside a scenario directory) holds each camera frame with its
exact per-pixel ground truth (gt/*.npz: item label, depth, pixel footprint, contrast-to-noise ratio
and the referee's quality mask). The exporter writes, per frame, the image, a class mask (0 =
background, k = perception.classes[k - 1]) and YOLO box labels, plus one annotations.json listing
every visible object: item id, class, box, pixel count, the share of its pixels that meet the
coverage limits, its physical size and size bin, and whether a detector is expected to find it in
that frame. An object is `required` when it shows at least perception.min_visible_px pixels and at
least perception.min_inspectable_share of them are inspectable; otherwise it is `ignore`, and a
detection on it is neither a hit nor a false alarm. Frames whose camera delivered no image are left
out.

Frames are assigned to train, val and test by scenario seed (perception.splits), so the overlapping
frames of one hull never straddle two splits.
"""
from __future__ import annotations

import json
import os
import shutil

import numpy as np
from PIL import Image

from .rng import stream

SPLITS = ('train', 'val', 'test')


def split_of(seed: int, shares) -> str:
    u = float(stream(seed, 'dataset_split').uniform())
    edges = np.cumsum(shares)
    return SPLITS[int(np.searchsorted(edges, u, side='right').clip(0, 2))]


def size_bin(area_m2: float, bins) -> str:
    return ('small', 'medium', 'large')[int(np.searchsorted(bins, area_m2, side='right'))]


def item_area(it: dict) -> float:
    if 'area_m2' in it:
        return float(it['area_m2'])
    L, _, H = it['size_m']                                         # anode: its face on the hull
    return float(L * H)


def frame_objects(label: np.ndarray, quality: np.ndarray, items: dict, cfg: dict) -> list[dict]:
    """Every visible item of a perception class in one frame, with its box and whether it is required."""
    p = cfg['perception']
    objs = []
    ids, counts = np.unique(label[label >= 0], return_counts=True)
    for iid, n in zip(ids.tolist(), counts.tolist()):
        it = items[iid]
        if it['class'] not in p['classes']:
            continue
        m = label == iid
        rows, cols = np.nonzero(m)
        share = float(quality[m].mean())
        area = item_area(it)
        objs.append({'item': iid, 'class': it['class'], 'status': it.get('status'),
                     'box': [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1],
                     'pixels': int(n), 'inspectable_share': round(share, 4), 'area_m2': round(area, 6),
                     'size': size_bin(area, p['size_bins_m2']),
                     'required': bool(n >= p['min_visible_px'] and share >= p['min_inspectable_share'])})
    return objs


def export(run_dirs: list[str], out_dir: str, cfg: dict) -> dict:
    """Write the dataset for the given run directories (each inside its scenario directory)."""
    p = cfg['perception']
    classes = list(p['classes'])
    for split in SPLITS:
        for sub in ('images', 'masks', 'labels'):
            os.makedirs(os.path.join(out_dir, sub, split), exist_ok=True)
    frames, splits = {}, {s: [] for s in SPLITS}
    for run in run_dirs:
        run = os.path.normpath(run)
        with open(os.path.join(os.path.dirname(run), 'scenario.json')) as f:
            spec = json.load(f)
        items = {it['id']: it for it in spec['items']}
        seed, split = int(spec['seed']), split_of(int(spec['seed']), p['splits'])
        tag = f'{seed}_{os.path.basename(run)}'
        for line in open(os.path.join(run, 'frames.jsonl')):
            rec = json.loads(line)
            if not rec.get('signal', True):
                continue
            gt = np.load(os.path.join(run, 'gt', rec['frame'] + '.npz'))
            label = gt['label']
            objs = frame_objects(label, gt['quality'], items, cfg)
            fid = f'{tag}_{rec["frame"]}'
            H, W = label.shape
            shutil.copyfile(os.path.join(run, 'frames', rec['frame'] + '.png'), os.path.join(out_dir, 'images', split, fid + '.png'))
            mask = np.zeros(label.shape, np.uint8)
            for o in objs:
                mask[label == o['item']] = classes.index(o['class']) + 1
            Image.fromarray(mask).save(os.path.join(out_dir, 'masks', split, fid + '.png'))
            with open(os.path.join(out_dir, 'labels', split, fid + '.txt'), 'w') as f:
                for o in objs:
                    if o['pixels'] >= p['min_visible_px']:
                        x0, y0, x1, y1 = o['box']
                        f.write(f'{classes.index(o["class"])} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} '
                                f'{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n')
            frames[fid] = {'split': split, 'run': run, 'scenario_seed': seed, 'frame': rec['frame'], 'camera': rec['camera'],
                           'width': W, 'height': H, 'image': os.path.join('images', split, fid + '.png'), 'objects': objs}
            splits[split].append(fid)
    ann = {'classes': classes, 'splits': splits, 'frames': frames}
    with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
        json.dump(ann, f)
    return summary(ann)


def summary(ann: dict) -> dict:
    out = {'frames': {s: len(v) for s, v in ann['splits'].items()}, 'objects': {}, 'required': {}}
    for fr in ann['frames'].values():
        for o in fr['objects']:
            out['objects'][o['class']] = out['objects'].get(o['class'], 0) + 1
            out['required'][o['class']] = out['required'].get(o['class'], 0) + int(o['required'])
    return out
