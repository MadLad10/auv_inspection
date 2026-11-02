"""Seeded hull-inspection scenarios with exact ground truth.

A scenario is the hull geometry, its appearance texture, a label raster (one item id per texture
pixel, -1 for sound paint) and the item list: painted defects (corrosion, coating damage,
biofouling, cracks, missing-anode footprints) and anodes, which are 3-D geometry (intact or
depleted). Areas, positions and sizes come from the masks and the analytic hull, so they are exact.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
from dataclasses import dataclass, field

import numpy as np
from PIL import Image

from . import config as config_mod
from .hull import HullGeometry
from .rng import stream
from .texture import blob_mask, crack_mask, disc, erode, polygon_mask, value_noise

DEFECT_CLASSES = ('corrosion', 'coating_damage', 'biofouling', 'crack', 'missing_anode')
SCENARIO_FILES = ('scenario.json', 'texture.png', 'labels.npz')


@dataclass
class Scenario:
    seed: int
    cfg: dict
    hull: HullGeometry
    texture: np.ndarray            # (rows, cols, 3) uint8; rows run down the girth, cols along x
    labels: np.ndarray             # (rows, cols) int32 item id, -1 = sound paint
    items: list[dict] = field(default_factory=list)
    features: list[dict] = field(default_factory=list)   # benign distractors (sea chests)

    @property
    def mm_per_px(self) -> float:
        return float(self.cfg['hull']['texture_mm_per_px'])

    def defects(self) -> list[dict]:
        return [i for i in self.items if i['class'] in DEFECT_CLASSES]

    def anodes(self) -> list[dict]:
        return [i for i in self.items if i['class'] == 'anode']

    def spec(self) -> dict:
        return {
            'schema_version': config_mod.SCHEMA_VERSION,
            'seed': self.seed,
            'hull': self.hull.to_dict(),
            'texture_shape': list(self.texture.shape),
            'texture_sha256': hashlib.sha256(self.texture.tobytes()).hexdigest(),
            'labels_sha256': hashlib.sha256(self.labels.tobytes()).hexdigest(),
            'items': self.items,
            'features': self.features,
            'config_sha256': config_mod.digest(config_mod.world(self.cfg)),
            'config': config_mod.world(self.cfg),
        }

    def digest(self) -> str:
        return hashlib.sha256(json.dumps(self.spec(), sort_keys=True).encode()).hexdigest()

    def save(self, out_dir: str) -> str:
        os.makedirs(out_dir, exist_ok=True)
        spec = self.spec()
        spec['digest'] = self.digest()
        with open(os.path.join(out_dir, 'scenario.json'), 'w') as f:
            json.dump(spec, f, indent=1, sort_keys=True)
        Image.fromarray(self.texture).save(os.path.join(out_dir, 'texture.png'), compress_level=1)
        np.savez_compressed(os.path.join(out_dir, 'labels.npz'), labels=self.labels)
        return spec['digest']

    @classmethod
    def load(cls, path: str, run_cfg: dict | None = None) -> 'Scenario':
        """Load a saved scenario; run settings come from run_cfg (default: the package defaults)."""
        with open(os.path.join(path, 'scenario.json')) as f:
            spec = json.load(f)
        texture = np.asarray(Image.open(os.path.join(path, 'texture.png')).convert('RGB'))
        labels = np.load(os.path.join(path, 'labels.npz'))['labels']
        cfg = config_mod.merge_world(run_cfg or config_mod.load(), spec['config'])
        scn = cls(spec['seed'], cfg, HullGeometry(**spec['hull']), texture, labels,
                  spec['items'], spec['features'])
        if 'digest' in spec and scn.digest() != spec['digest']:
            raise ValueError(f'scenario at {path} does not match its recorded digest')
        return scn


def generate(seed: int, cfg: dict) -> Scenario:
    config_mod.validate(cfg)
    return _Builder(seed, cfg).build()


class _Builder:
    def __init__(self, seed: int, cfg: dict):
        self.seed, self.cfg = seed, cfg
        self.hull = HullGeometry.from_config(cfg)
        self.px = cfg['hull']['texture_mm_per_px'] / 1000.0
        self.rows, self.cols = math.ceil(self.hull.girth / self.px), math.ceil(self.hull.length / self.px)
        self.labels = np.full((self.rows, self.cols), -1, np.int32)
        self.occupied = np.zeros((self.rows, self.cols), bool)
        self.items: list[dict] = []
        self.features: list[dict] = []
        g0, g1 = cfg['hull']['inspect_girth_m']
        m = cfg['hull']['inspect_margin_m']
        self.region = (int(g0 / self.px), int(math.floor(g1 / self.px)), int(m / self.px), int((self.hull.length - m) / self.px))
        self.gap = int(math.ceil(cfg['defects']['min_gap_m'] / self.px))

    def m2px(self, metres: float) -> float:
        return metres / self.px

    def build(self) -> Scenario:
        self.tex = self._paint_base()
        self._sea_chests()
        self._anodes()
        for name in ('corrosion', 'coating_damage', 'biofouling', 'crack'):
            self._defects(name)
        self._slime()
        tex = (np.clip(self.tex, 0, 1) * 255 + 0.5).astype(np.uint8)
        return Scenario(self.seed, self.cfg, self.hull, tex, self.labels, self.items, self.features)

    # ---------------------------------------------------------------- appearance
    def _paint_base(self) -> np.ndarray:
        a = self.cfg['appearance']
        rng = stream(self.seed, 'paint')
        shape = (self.rows, self.cols)
        shade = 1.0 + a['paint_variation'] * value_noise(shape, self.m2px(a['paint_variation_scale_m']), rng)
        shade += a['grain'] * value_noise(shape, self.m2px(a['grain_scale_m']), rng)
        tex = np.asarray(a['paint_rgb'], np.float32)[None, None, :] * shade[..., None]
        # weld seams: horizontal at every strake, vertical seams staggered by half a plate per strake
        pw, ph = (self.m2px(v) for v in a['plate_size_m'])
        sw = max(1, int(round(a['seam_width_mm'] / 1000 / self.px)))
        seam = np.zeros(shape, bool)
        for k, r in enumerate(np.arange(ph, self.rows, ph).astype(int)):
            seam[max(0, r - sw // 2):r + sw - sw // 2, :] = True
        for k, r0 in enumerate(np.arange(0, self.rows, ph).astype(int)):
            r1 = min(self.rows, int(r0 + ph))
            for c in np.arange((k % 2) * pw / 2, self.cols, pw).astype(int):
                seam[r0:r1, max(0, c - sw // 2):c + sw - sw // 2] = True
        tex[seam] *= a['seam_shade']
        return tex

    def _slime(self):
        a = self.cfg['appearance']
        rng = stream(self.seed, 'slime')
        strength = rng.uniform(*a['slime_strength'])
        m = 0.5 * (1.0 + value_noise((self.rows, self.cols), self.m2px(a['slime_scale_m']), rng))
        w = (strength * m)[..., None]
        self.tex = self.tex * (1 - w) + np.asarray(a['slime_rgb'], np.float32) * w
        self.features.append({'class': 'slime_film', 'strength': round(float(strength), 4)})

    def _sea_chests(self):
        a = self.cfg['appearance']
        rng = stream(self.seed, 'sea_chest')
        w, h = (int(self.m2px(v)) for v in a['sea_chest_size_m'])
        bar = max(2, int(self.m2px(a['sea_chest_bar_spacing_m'])))
        r_lo, r_hi = int(self.m2px(a['sea_chest_min_girth_m'])), int(self.m2px(self.hull.side_depth)) - h
        for _ in range(a['sea_chests']):
            for _ in range(self.cfg['defects']['max_tries']):
                r0 = int(rng.integers(r_lo, max(r_lo + 1, r_hi)))
                c0 = int(rng.integers(self.region[2], self.region[3] - w))
                if not self.occupied[r0:r0 + h, c0:c0 + w].any():
                    break
            else:
                continue
            block = self.tex[r0:r0 + h, c0:c0 + w]
            grid = np.zeros((h, w), bool)
            grid[:, ::bar] = True
            grid[:, 1::bar] = True
            block[~grid] *= a['sea_chest_opening_shade']
            self._occupy(r0, c0, np.ones((h, w), bool))
            x0, s0 = c0 * self.px, r0 * self.px
            self.features.append({'class': 'sea_chest', 'bbox_xs': _r([x0, s0, x0 + w * self.px, s0 + h * self.px])})

    # ---------------------------------------------------------------- anodes
    def _anodes(self):
        a = self.cfg['anodes']
        rng = stream(self.seed, 'anodes')
        L, T, H = a['size_m']
        xs = np.arange(a['spacing_m'] / 2, self.hull.length - L / 2, a['spacing_m'])
        for x in xs:
            u = rng.uniform()
            status = 'missing' if u < a['p_missing'] else 'depleted' if u < a['p_missing'] + a['p_depleted'] else 'intact'
            if status == 'missing':
                self._missing_anode(float(x))
                continue
            scale = a['depleted_scale'] if status == 'depleted' else 1.0
            r0, c0 = int(self.m2px(a['girth_m'] - H / 2)), int(self.m2px(x - L / 2))
            self._occupy(r0, c0, np.ones((int(self.m2px(H)), int(self.m2px(L))), bool))
            self.items.append({
                'id': len(self.items), 'class': 'anode', 'status': status,
                'centroid_xs': _r([x, a['girth_m']]),
                'size_m': _r([L, T * scale, H * scale]),
                'centroid_world': _r(self.hull.point(x, a['girth_m']).tolist()),
            })

    def _missing_anode(self, x: float):
        a = self.cfg['anodes']
        L, _, H = a['size_m']
        r0, c0 = int(self.m2px(a['girth_m'] - H / 2)), int(self.m2px(x - L / 2))
        mask = np.ones((int(self.m2px(H)), int(self.m2px(L))), bool)
        self.tex[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] = a['footprint_rgb']
        bolt = disc(self.m2px(a['bolt_radius_mm'] / 1000))
        for fx in (0.2, 0.8):
            br, bc = r0 + mask.shape[0] // 2 - bolt.shape[0] // 2, c0 + int(fx * mask.shape[1]) - bolt.shape[1] // 2
            self.tex[br:br + bolt.shape[0], bc:bc + bolt.shape[1]][bolt] = a['bolt_rgb']
        self._record('missing_anode', r0, c0, mask)

    # ---------------------------------------------------------------- defects
    def _defects(self, name: str):
        d = self.cfg['defects']
        c = d[name]
        rng = stream(self.seed, f'defect_{name}')
        n = int(rng.integers(c['count'][0], c['count'][1] + 1))
        for _ in range(n):
            mask = self._shape(name, rng)
            pos = self._place(mask, rng)
            if pos is None:
                continue
            r0, c0 = pos
            self._paint_defect(name, r0, c0, mask, rng)
            item = self._record(name, r0, c0, mask)
            if name == 'crack':
                item['length_m'] = round(float(self._crack_len), 4)

    def _shape(self, name: str, rng) -> np.ndarray:
        d = self.cfg['defects']
        c = d[name]
        aspect = rng.uniform(*d['aspect'])
        if name == 'crack':
            self._crack_len = rng.uniform(*c['length_m'])
            return crack_mask(self.m2px(self._crack_len), self.m2px(rng.uniform(*c['width_mm']) / 1000),
                              self.m2px(c['step_m']), math.radians(c['turn_std_deg']), rng)
        area = math.exp(rng.uniform(*np.log(c['area_m2']))) / self.px ** 2
        if name == 'coating_damage':
            return polygon_mask(area, aspect, int(rng.integers(d['polygon_vertices'][0], d['polygon_vertices'][1] + 1)),
                                d['blob_irregularity'], rng)
        return blob_mask(area, aspect, d['blob_irregularity'], rng)

    def _place(self, mask: np.ndarray, rng) -> tuple[int, int] | None:
        h, w = mask.shape
        r_lo, r_hi, c_lo, c_hi = self.region
        if r_hi - r_lo <= h or c_hi - c_lo <= w:
            return None
        g = self.gap
        for _ in range(self.cfg['defects']['max_tries']):
            r0, c0 = int(rng.integers(r_lo, r_hi - h)), int(rng.integers(c_lo, c_hi - w))
            if not self.occupied[max(0, r0 - g):r0 + h + g, max(0, c0 - g):c0 + w + g].any():
                return r0, c0
        return None

    def _paint_defect(self, name: str, r0: int, c0: int, mask: np.ndarray, rng):
        c = self.cfg['defects'][name]
        block = self.tex[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]]
        rgb = np.asarray(c['rgb'], np.float32)
        if name in ('corrosion', 'biofouling'):
            mot = 1.0 + c['mottle'] * value_noise(mask.shape, max(2.0, self.m2px(c['scale_m'])), rng)
            block[mask] = (rgb[None, None, :] * mot[..., None])[mask]
            if name == 'biofouling':
                self._barnacles(block, mask, c, rng)
        elif name == 'coating_damage':
            block[mask] = rgb
            edge = mask & ~erode(mask, c['edge_px'])
            block[edge] *= c['edge_shade']
        else:
            block[mask] = rgb

    def _barnacles(self, block: np.ndarray, mask: np.ndarray, c: dict, rng):
        area_m2 = mask.sum() * self.px ** 2
        rows, cols = np.nonzero(mask)
        for k in rng.choice(rows.size, size=int(c['barnacle_density_per_m2'] * area_m2), replace=True):
            d = disc(self.m2px(rng.uniform(*c['barnacle_radius_mm']) / 1000))
            r, cc = rows[k] - d.shape[0] // 2, cols[k] - d.shape[1] // 2
            rs, cs = slice(max(0, r), min(mask.shape[0], r + d.shape[0])), slice(max(0, cc), min(mask.shape[1], cc + d.shape[1]))
            sub = d[rs.start - r:rs.stop - r, cs.start - cc:cs.stop - cc] & mask[rs, cs]
            block[rs, cs][sub] = c['barnacle_rgb']

    def _occupy(self, r0: int, c0: int, mask: np.ndarray):
        self.occupied[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] |= mask

    def _record(self, name: str, r0: int, c0: int, mask: np.ndarray) -> dict:
        iid = len(self.items)
        self.labels[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]][mask] = iid
        self._occupy(r0, c0, mask)
        rr, cc = np.nonzero(mask)
        xc, sc = (c0 + cc.mean() + 0.5) * self.px, (r0 + rr.mean() + 0.5) * self.px
        item = {
            'id': iid, 'class': name,
            'area_m2': round(float(mask.sum()) * self.px ** 2, 6),
            'bbox_xs': _r([c0 * self.px, r0 * self.px, (c0 + mask.shape[1]) * self.px, (r0 + mask.shape[0]) * self.px]),
            'centroid_xs': _r([xc, sc]),
            'centroid_world': _r(self.hull.point(xc, sc).tolist()),
        }
        self.items.append(item)
        return item


def _r(v, nd: int = 4):
    return [round(float(x), nd) for x in v]


def png_bytes(arr: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format='PNG', compress_level=1)
    return buf.getvalue()
