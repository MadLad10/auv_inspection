"""Real defects for the simulator: corrosion, coating damage and fouling cut from LIACi images.

`auvinspect liaci-patches` cuts the corrosion, paint_peel and marine_growth polygons out of LIACi's
original images, from our training split only (neither our validation blocks nor LIACi's test
images, so those stay a clean test of anything trained on patched hulls). The settings are in
real_images.liaci.patches. It keeps an object when:
- it is whole: at least min_pixels, clear of the image border, at most max_overlap of it under other
  labelled objects (the hull aside);
- it lies on plain hull: a ring around it (ring_width x sqrt(object pixels) wide, at least
  ring_min_px) is at least min_ring_hull ship_hull and no other object, with a median brightness
  within ring_luma (neither murky nor glaring), and at most max_glare of the object's pixels are
  saturated (any channel at glare_dn or above).
A patch is the object's reflectance relative to the paint around it: the image decoded from sRGB to
linear, divided per channel by a plane fitted to the log of the ring, clamped to `reflectance` (an
object with more than max_clamped of its pixels at the clamp is dropped: its ring was not the paint
under it, as on a structure's edge), and stored at most max_side_px on its longer side. The division removes the paint's colour, the lamp's
colour and its gradient across the patch, and the water's colour. It leaves in place the haze in
front of the patch, glare, and the 3-D relief of fouling (its shading is painted flat).

The scenario generator, given a library (`generate --patches`), takes each corrosion, coating damage
and biofouling defect from it instead of drawing one. The area is drawn as before. A patch of the
class is drawn among those enlarged at most max_upscale to reach it (the largest one when none is),
then turned by a drawn angle, maybe mirrored, and scaled to the area. Its relative reflectance
multiplies the hull paint under it. Its mask is the item's label, so ground truth stays exact.
Cracks and missing anodes are drawn as before (LIACi has neither).
The scenario records the library's digest and each item its patch: the same seed, configuration and
library give the same digest. The paint around their defects was LIACi's ships' paint, so a patch on
a paint unlike theirs carries their contrast, not its own.

Patches, and every hull painted with them, are adaptations of LIACi (CC BY-NC-SA 4.0). They stay out
of this repository and its CI.
"""
from __future__ import annotations

import hashlib
import json
import math
import os

import numpy as np
from PIL import Image
from scipy import ndimage

from . import liaci

PAINTED = ('corrosion', 'coating_damage', 'biofouling')     # the defect classes a patch can paint
LIBRARY_FILE = 'library.json'
_LOG_FLOOR = 1e-4                                           # keeps the log finite on black pixels


def _linear(rgb8: np.ndarray) -> np.ndarray:
    v = rgb8.astype(np.float32) / 255.0
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4)


def _resize(arr: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """A float (h, w) or (h, w, c) array resized to size = (w, h), bilinear (antialiased when shrinking)."""
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32)).resize(size, Image.BILINEAR), np.float32)
    return np.stack([_resize(arr[..., k], size) for k in range(arr.shape[2])], -1)


def _turn(arr: np.ndarray, deg: float, fill: float) -> np.ndarray:
    if arr.ndim == 2:
        return np.asarray(Image.fromarray(arr.astype(np.float32)).rotate(deg, Image.BILINEAR, expand=True, fillcolor=fill), np.float32)
    return np.stack([_turn(arr[..., k], deg, fill) for k in range(arr.shape[2])], -1)


def _relative(lin: np.ndarray, plain: np.ndarray) -> np.ndarray:
    """lin divided per channel by exp(a + b x + c y), fitted by least squares to the log of the plain-paint pixels."""
    h, w = plain.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    A = np.stack([np.ones(h * w, np.float32), xx.ravel() / w, yy.ravel() / h], 1)
    logs = np.log(np.maximum(lin.reshape(-1, 3), _LOG_FLOOR))
    coef = np.linalg.lstsq(A[plain.ravel()], logs[plain.ravel()], rcond=None)[0]
    return lin / np.exp(A @ coef).reshape(h, w, 3)


def build(root: str, out_dir: str, cfg: dict) -> dict:
    """Write the patch library of the LIACi images under `root` (the unpacked LIACi_dataset_pretty)."""
    lp = cfg['real_images']['liaci']
    pp = lp['patches']
    coco, official, names, digests = liaci.read(root, cfg)
    wanted = {cat: cls for cat, cls in lp['classes'].items() if cls in PAINTED}
    by_image: dict[int, list] = {}
    for a in coco['annotations']:
        by_image.setdefault(a['image_id'], []).append(a)
    lib = {cls: [] for cls in PAINTED}
    rejected = {cls: {} for cls in PAINTED}
    for cls in PAINTED:
        os.makedirs(os.path.join(out_dir, cls), exist_ok=True)

    def reject(cls, why):
        rejected[cls][why] = rejected[cls].get(why, 0) + 1

    for im in sorted(coco['images'], key=lambda i: i['file_name']):
        anns = by_image.get(im['id'], [])
        if liaci.split_of(im['file_name'], official[im['file_name']], cfg) != 'train' \
                or not any(names[a['category_id']] in wanted for a in anns):
            continue
        W, H = im['width'], im['height']
        masks = {a['id']: liaci._raster(a['segmentation'], 1.0, (W, H)) for a in anns}
        hull = np.zeros((H, W), bool)
        cover = np.zeros((H, W), np.int32)                  # how many labelled objects (the hull aside) cover a pixel
        for a in anns:
            if names[a['category_id']] == 'ship_hull':
                hull |= masks[a['id']]
            else:
                cover += masks[a['id']]
        img = None
        for a in anns:
            cat = names[a['category_id']]
            if cat not in wanted:
                continue
            cls, m = wanted[cat], masks[a['id']]
            n = int(m.sum())
            if n < pp['min_pixels']:
                reject(cls, 'small')
                continue
            rows, cols = np.nonzero(m)
            r0, r1, c0, c1 = rows.min(), rows.max() + 1, cols.min(), cols.max() + 1
            if r0 == 0 or c0 == 0 or r1 == H or c1 == W:
                reject(cls, 'cut by the border')
                continue
            others = (cover - m) > 0
            if (m & others).sum() > pp['max_overlap'] * n:
                reject(cls, 'overlapped')
                continue
            ring_px = max(pp['ring_min_px'], int(round(pp['ring_width'] * math.sqrt(n))))
            win = (slice(max(0, r0 - ring_px), min(H, r1 + ring_px)), slice(max(0, c0 - ring_px), min(W, c1 + ring_px)))
            mw = m[win]
            ring = ndimage.binary_dilation(mw, iterations=ring_px) & ~mw
            plain = ring & hull[win] & ~others[win]
            if plain.sum() < pp['min_ring_hull'] * ring.sum():
                reject(cls, 'not on plain hull')
                continue
            if img is None:
                img = np.asarray(Image.open(os.path.join(root, 'images', im['file_name'])).convert('RGB'))
            iw = img[win]
            luma = float(np.median(iw[plain].mean(1)))
            if not pp['ring_luma'][0] <= luma <= pp['ring_luma'][1]:
                reject(cls, 'murky or glaring')
                continue
            if (iw[mw] >= pp['glare_dn']).any(1).mean() > pp['max_glare']:
                reject(cls, 'glare')
                continue
            rel = _relative(_linear(iw), plain)
            lo, hi = pp['reflectance']
            if ((rel[mw] < lo) | (rel[mw] > hi)).any(1).mean() > pp['max_clamped']:
                reject(cls, 'out of the reflectance range')
                continue
            rel = np.clip(rel, lo, hi)
            box = (slice(r0 - win[0].start, r1 - win[0].start), slice(c0 - win[1].start, c1 - win[1].start))
            rel, mb = np.where(mw[box][..., None], rel[box], 1.0), mw[box]   # paint (1.0) outside the outline
            scale = min(1.0, pp['max_side_px'] / max(mb.shape))
            if scale < 1.0:
                size = (max(1, round(mb.shape[1] * scale)), max(1, round(mb.shape[0] * scale)))
                rel, mb = _resize(rel, size), _resize(mb.astype(np.float32), size) > 0.5
            rel16 = rel.astype(np.float16)
            pid = f'{os.path.splitext(im["file_name"])[0]}_a{a["id"]}'
            np.savez_compressed(os.path.join(out_dir, cls, pid + '.npz'), rel=rel16, mask=mb)
            lib[cls].append({'id': pid, 'file': im['file_name'], 'annotation': int(a['id']), 'liaci_category': cat,
                             'pixels': int(mb.sum()), 'shape': list(mb.shape), 'scale': round(scale, 4),
                             'sha256': hashlib.sha256(rel16.tobytes() + mb.tobytes()).hexdigest()})
    index = {'source': {**liaci.SOURCE, **digests}, 'settings': {'patches': pp, 'classes': wanted,
             'val_block': lp['val_block'], 'val_every': lp['val_every']}, 'classes': lib}
    with open(os.path.join(out_dir, LIBRARY_FILE), 'w') as f:
        json.dump(index, f, sort_keys=True)
    return {'out': out_dir, 'library_sha256': _digest(index), 'patches': {c: len(v) for c, v in lib.items()},
            'rejected': rejected}


def _digest(index: dict) -> str:
    return hashlib.sha256(json.dumps(index, sort_keys=True).encode()).hexdigest()


class Library:
    """A patch library on disk (see the module notes)."""

    def __init__(self, path: str):
        with open(os.path.join(path, LIBRARY_FILE)) as f:
            self.index = json.load(f)
        self.path = path
        self.digest = _digest(self.index)

    def info(self, cfg: dict) -> dict:
        """What a scenario painted from this library records (part of its digest)."""
        src = self.index['source']
        return {'library_sha256': self.digest, 'max_upscale': cfg['real_images']['liaci']['patches']['max_upscale'],
                **{k: src[k] for k in ('name', 'license', 'citation')}}

    def patch(self, cls: str, entry: dict) -> tuple[np.ndarray, np.ndarray]:
        with np.load(os.path.join(self.path, cls, entry['id'] + '.npz')) as z:
            rel, mask = z['rel'], z['mask']
        if hashlib.sha256(rel.tobytes() + mask.tobytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'patch {cls}/{entry["id"]} does not match its library entry')
        return rel.astype(np.float32), mask

    def draw(self, cls: str, area_px: float, rng: np.random.Generator, cfg: dict) -> tuple[np.ndarray, np.ndarray, str]:
        """A defect of about area_px texture pixels: its mask, its relative reflectance and the patch id."""
        entries = self.index['classes'].get(cls) or []
        if not entries:
            raise ValueError(f'the patch library at {self.path} has no {cls} patches')
        up = cfg['real_images']['liaci']['patches']['max_upscale']
        fit = [e for e in entries if e['pixels'] * up ** 2 >= area_px] or [max(entries, key=lambda e: e['pixels'])]
        e = fit[int(rng.integers(len(fit)))]
        deg, mirrored = float(rng.uniform(0, 360)), bool(rng.integers(2))
        rel, mask = self.patch(cls, e)
        f = math.sqrt(area_px / e['pixels'])
        size = (max(1, round(mask.shape[1] * f)), max(1, round(mask.shape[0] * f)))
        rel, m = _turn(_resize(rel, size), deg, 1.0), _turn(_resize(mask.astype(np.float32), size), deg, 0.0)
        if mirrored:
            rel, m = rel[:, ::-1], m[:, ::-1]
        m = m > 0.5
        if not m.any():
            return np.ones((1, 1), bool), np.ones((1, 1, 3), np.float32), e['id']
        rows, cols = np.flatnonzero(m.any(1)), np.flatnonzero(m.any(0))
        box = (slice(rows[0], rows[-1] + 1), slice(cols[0], cols[-1] + 1))
        return m[box], np.ascontiguousarray(rel[box]), e['id']
