"""LIACi: images of real underwater ship inspections, imported as a perception dataset.

LIACi (M. Waszak, A. Cardaillac, B. Elvesaeter, F. Rodolen, M. Ludvigsen, "Semantic segmentation in
underwater ship inspections: benchmark and data set", IEEE Journal of Oceanic Engineering 48(2),
462-473, 2023, doi:10.1109/JOE.2022.3219129) holds 1893 images from ROV inspections of ships with
polygons drawn by inspection experts in ten categories. It is licensed CC BY-NC-SA 4.0 and is not part
of this repository: it is downloaded from https://liaci.sintef.cloud and imported with
`auvinspect import-liaci`.

The import writes the dataset format of dataset.py, so the perception referee reads it as it reads
simulated frames:
- splits: the official split. Its train images, less every real_images.liaci.val_every-th block of
  val_block consecutive images (held out for validation; neighbouring images come from one video),
  and its test images. The images it marks not_used (near-duplicates of others) are left out.
- images resized to real_images.liaci.width (the simulator's camera width);
- the categories in real_images.liaci.classes, mapped to perception classes. The others are
  background: ship hull, propeller, bilge keel, sea chest grating, overboard valve, and `defect`
  (mechanical damage, not a crack).
- masks drawn from the polygons, larger objects first, so a defect inside a larger one stays
  visible. LIACi's own per-class bitmaps hold one label per pixel and lose some defects under the
  hull or propeller drawn over them.
- every object with at least perception.min_visible_px pixels is required: a real image has no
  coverage limits to apply. Physical sizes are unknown (size bin `unknown`).

Scores on these images are scores on LIACi's images (several ships, unknown cameras), not field
performance; its test images come from the same inspections as its training images.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os

import numpy as np
from PIL import Image, ImageDraw

from .dataset import SPLITS, summary

SOURCE = {'name': 'LIACi', 'license': 'CC BY-NC-SA 4.0', 'url': 'https://liaci.sintef.cloud',
          'citation': 'Waszak et al., IEEE Journal of Oceanic Engineering 48(2):462-473, 2023, doi:10.1109/JOE.2022.3219129'}


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def split_of(file_name: str, official: str, cfg: dict) -> str | None:
    """Our split of a LIACi image: its official split, with blocks of the training images held out
    for validation; None for the images LIACi leaves out."""
    if official == 'test':
        return 'test'
    if official != 'train':
        return None
    p = cfg['real_images']['liaci']
    number = int(os.path.splitext(file_name)[0].rsplit('_', 1)[1])
    return 'val' if (number - 1) // p['val_block'] % p['val_every'] == p['val_every'] - 1 else 'train'


def _raster(polygons: list, scale: float, size: tuple[int, int]) -> np.ndarray:
    m = Image.new('L', size, 0)
    d = ImageDraw.Draw(m)
    for p in polygons:
        if len(p) >= 6:
            d.polygon([(x * scale, y * scale) for x, y in zip(p[::2], p[1::2])], fill=1)
    return np.asarray(m, bool)


def import_dataset(root: str, out_dir: str, cfg: dict) -> dict:
    """Write the LIACi images under `root` (the unpacked LIACi_dataset_pretty) as a perception dataset."""
    p, lp = cfg['perception'], cfg['real_images']['liaci']
    classes = list(p['classes'])
    labels_path, split_path = os.path.join(root, 'coco-labels.json'), os.path.join(root, 'train_test_split.csv')
    with open(labels_path) as f:
        coco = json.load(f)
    with open(split_path) as f:
        official = {r['file_name']: r['split'] for r in csv.DictReader(f)}
    names = {c['id']: c['name'] for c in coco['categories']}
    unknown = set(lp['classes']) - set(names.values())
    if unknown:
        raise ValueError(f'real_images.liaci.classes names categories LIACi does not have: {sorted(unknown)}')
    by_image: dict[int, list] = {}
    for a in coco['annotations']:
        if names[a['category_id']] in lp['classes']:
            by_image.setdefault(a['image_id'], []).append(a)
    for split in SPLITS:
        for sub in ('images', 'masks', 'labels'):
            os.makedirs(os.path.join(out_dir, sub, split), exist_ok=True)
    frames, splits = {}, {s: [] for s in SPLITS}
    for im in sorted(coco['images'], key=lambda i: i['file_name']):
        split = split_of(im['file_name'], official[im['file_name']], cfg)
        if split is None:
            continue
        W, H = lp['width'], int(round(im['height'] * lp['width'] / im['width']))
        scale = W / im['width']
        fid = 'liaci_' + os.path.splitext(im['file_name'])[0]
        objs, mask = [], np.zeros((H, W), np.uint8)
        for a in sorted(by_image.get(im['id'], []), key=lambda a: -a['area']):   # larger first: smaller stay on top
            m = _raster(a['segmentation'], scale, (W, H))
            n = int(m.sum())
            if n == 0:
                continue
            cls = lp['classes'][names[a['category_id']]]
            mask[m] = classes.index(cls) + 1
            rows, cols = np.nonzero(m)
            objs.append({'item': int(a['id']), 'class': cls, 'status': None, 'liaci_category': names[a['category_id']],
                         'box': [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1],
                         'pixels': n, 'inspectable_share': 1.0, 'area_m2': None, 'size': 'unknown',
                         'required': bool(n >= p['min_visible_px'])})
        image = Image.open(os.path.join(root, 'images', im['file_name'])).convert('RGB')
        image.resize((W, H), Image.LANCZOS).save(os.path.join(out_dir, 'images', split, fid + '.png'))
        Image.fromarray(mask).save(os.path.join(out_dir, 'masks', split, fid + '.png'))
        with open(os.path.join(out_dir, 'labels', split, fid + '.txt'), 'w') as f:
            for o in objs:
                if o['required']:
                    x0, y0, x1, y1 = o['box']
                    f.write(f'{classes.index(o["class"])} {(x0 + x1) / 2 / W:.6f} {(y0 + y1) / 2 / H:.6f} '
                            f'{(x1 - x0) / W:.6f} {(y1 - y0) / H:.6f}\n')
        frames[fid] = {'split': split, 'source': 'liaci', 'file': im['file_name'], 'scenario_seed': None,
                       'original_size': [im['width'], im['height']], 'width': W, 'height': H,
                       'image': os.path.join('images', split, fid + '.png'), 'objects': objs}
        splits[split].append(fid)
    ann = {'classes': classes, 'splits': splits, 'frames': frames,
           'source': {**SOURCE, 'labels_sha256': _sha256(labels_path), 'split_sha256': _sha256(split_path),
                      'settings': {'liaci': lp, 'min_visible_px': p['min_visible_px']}}}
    with open(os.path.join(out_dir, 'annotations.json'), 'w') as f:
        json.dump(ann, f)
    return summary(ann)
