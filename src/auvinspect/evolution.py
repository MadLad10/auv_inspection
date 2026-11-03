"""The same hull some months later: defects grow, new ones appear, anodes deplete and fall off.

The evolved scenario keeps every item's id, so each change has exact ground truth. Rates come from the
`evolution` configuration and are recorded in the scenario's lineage (with the base digest), so the
evolved hull has its own digest:
  corrosion, coating damage, biofouling  the area grows by (1 + growth_per_month)^months: the outline
                     advances by a distance that varies smoothly along it (evolution.edge_noise), so the
                     shape stays irregular; it stops short of other items (defects.min_gap_m)
  cracks             both tips extend by growth_mm_per_month in total, continuing the random walk
  new defects        Poisson count of new_per_m2_month x band area x months per class, placed and painted
                     as the generator places and paints them
  anodes             an intact anode is depleted with probability 1 - (1 - deplete_per_month)^months; a
                     depleted one falls off with 1 - (1 - detach_per_month)^months and leaves a
                     missing-anode footprint under the same item id
Painting reuses the generator's appearance code, and the slime film is laid over the new pixels as it
lies over the rest of the hull. Old pixels are untouched. A hull painted with real defects (patches.py)
evolves with the same patch library: new defects come from it, and a real defect grows by continuing
its own texture, each new pixel copying the pixel mirrored across the old outline (slime included).
"""
from __future__ import annotations

import copy
import math

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from .patches import Library
from .rng import stream
from .scenario import Scenario, _Builder, _r
from .texture import value_noise

GROWING = ('corrosion', 'coating_damage', 'biofouling')


class _Evolver(_Builder):
    def __init__(self, scn: Scenario, months: float, ecfg: dict, library: Library | None = None):
        super().__init__(scn.seed, scn.cfg, library)
        self.scn, self.months, self.e = scn, months, ecfg
        self.tex = scn.texture.astype(np.float32) / 255.0
        self.labels = scn.labels.copy()
        self.items = copy.deepcopy(scn.items)
        self.features = copy.deepcopy(scn.features)
        self.occupied = self.labels >= 0
        a = scn.cfg['anodes']
        for it in self.items:                                   # anode fittings and sea chests are occupied too
            if it['class'] == 'anode':
                L, _, H = a['size_m']
                x, s = it['centroid_xs']
                r0, c0 = int(self.m2px(s - H / 2)), int(self.m2px(x - L / 2))
                self.occupied[r0:r0 + int(self.m2px(H)), c0:c0 + int(self.m2px(L))] = True
        for f in self.features:
            if 'bbox_xs' in f:
                x0, s0, x1, s1 = f['bbox_xs']
                self.occupied[int(self.m2px(s0)):int(self.m2px(s1)), int(self.m2px(x0)):int(self.m2px(x1))] = True
        self.changed = np.zeros(self.labels.shape, bool)              # newly painted pixels, still without the slime film

    def rng(self, what: str) -> np.random.Generator:
        return stream(self.seed, f'evolve:{self.months:g}:{what}')

    # ------------------------------------------------------------------ growth
    def _allowed(self, iid: int, window) -> np.ndarray:
        """Pixels an item may grow into: not within defects.min_gap_m of anything else, inside the band."""
        others = self.occupied[window] & (self.labels[window] != iid)
        near = ndimage.binary_dilation(others, iterations=self.gap) if self.gap else others
        r_lo, r_hi, c_lo, c_hi = self.region
        rows = np.arange(window[0].start, window[0].stop)[:, None]
        cols = np.arange(window[1].start, window[1].stop)[None, :]
        return ~near & (rows >= r_lo) & (rows < r_hi) & (cols >= c_lo) & (cols < c_hi)

    def _window(self, iid: int, pad: int):
        rr, cc = np.nonzero(self.labels == iid)
        return (slice(max(0, rr.min() - pad), min(self.rows, rr.max() + pad + 1)),
                slice(max(0, cc.min() - pad), min(self.cols, cc.max() + pad + 1)))

    def grow_area(self, it: dict, rng):
        iid, name = it['id'], it['class']
        factor = (1 + self.e[name]['growth_per_month']) ** self.months
        cur = int((self.labels == iid).sum())
        target = cur * factor
        pad = int(math.ceil(math.sqrt(target / math.pi) - math.sqrt(cur / math.pi))) * 2 + 4
        win = self._window(iid, pad)
        mask = self.labels[win] == iid
        dist = ndimage.distance_transform_edt(~mask)
        wobble = 1 + self.e['edge_noise'] * value_noise(mask.shape, max(4.0, pad / 2), rng)
        allowed = self._allowed(iid, win)
        lo, hi = 0.0, float(pad)
        for _ in range(30):                                     # outline advance that reaches the target area
            mid = (lo + hi) / 2
            grown = mask | ((dist <= mid * wobble) & allowed)
            lo, hi = (mid, hi) if grown.sum() < target else (lo, mid)
        grown = mask | ((dist <= hi * wobble) & allowed)
        new = grown & ~mask
        if not new.any():
            return
        r0, c0 = win[0].start, win[1].start
        if 'patch' in it:
            self._continue(self.tex[win], mask, new)
            self.labels[win][new] = iid
            self.occupied[win] |= new
            return
        self._paint_defect(name, r0, c0, new, rng)
        if name == 'coating_damage':                            # the flaking edge moves out with the outline
            c = self.cfg['defects'][name]
            edge = grown & ~ndimage.binary_erosion(grown, iterations=c['edge_px'])
            self.tex[win][edge & new] *= c['edge_shade']
        self.labels[win][new] = iid
        self.occupied[win] |= new
        self.changed[win] |= new

    @staticmethod
    def _continue(block: np.ndarray, mask: np.ndarray, new: np.ndarray):
        """New pixels of a real defect copy the pixel mirrored across the old outline (its nearest
        outline pixel when the mirror image falls outside the old mask)."""
        near = ndimage.distance_transform_edt(~mask, return_indices=True)[1]
        rr, cc = np.nonzero(new)
        qr, qc = near[0][rr, cc], near[1][rr, cc]
        mr, mc = np.clip(2 * qr - rr, 0, mask.shape[0] - 1), np.clip(2 * qc - cc, 0, mask.shape[1] - 1)
        inside = mask[mr, mc]
        block[rr, cc] = block[np.where(inside, mr, qr), np.where(inside, mc, qc)]

    def grow_crack(self, it: dict, rng):
        iid = it['id']
        c = self.cfg['defects']['crack']
        extra = self.e['crack']['growth_mm_per_month'] * self.months / 1000.0
        rr, cc = np.nonzero(self.labels == iid)
        pts = np.stack([cc, rr], 1).astype(float)
        mean = pts.mean(0)
        u = np.linalg.svd(pts - mean, full_matrices=False)[2][0]
        proj = (pts - mean) @ u
        width = max(1, int(round(len(rr) / max(self.m2px(it.get('length_m', 0.1)), 1.0))))
        steps = max(1, int(round(self.m2px(extra / 2) / self.m2px(c['step_m']))))
        pad = steps * int(math.ceil(self.m2px(c['step_m']))) + width + 4
        win = self._window(iid, pad)
        canvas = Image.new('L', (win[1].stop - win[1].start, win[0].stop - win[0].start), 0)
        d = ImageDraw.Draw(canvas)
        for end, sign in ((pts[np.argmax(proj)], 1.0), (pts[np.argmin(proj)], -1.0)):
            heading = math.atan2(sign * u[1], sign * u[0]) + np.cumsum(rng.normal(0, math.radians(c['turn_std_deg']), steps))
            walk = end + np.cumsum(np.stack([np.cos(heading), np.sin(heading)], -1) * self.m2px(c['step_m']), 0)
            line = np.vstack([end, walk]) - [win[1].start, win[0].start]
            d.line([tuple(p) for p in line], fill=1, width=width, joint='curve')
        new = (np.asarray(canvas, bool) & (self.labels[win] != iid) & self._allowed(iid, win))
        if not new.any():
            return
        self._paint_defect('crack', win[0].start, win[1].start, new, rng)
        self.labels[win][new] = iid
        self.occupied[win] |= new
        self.changed[win] |= new
        it['length_m'] = round(float(it.get('length_m', 0.0) + extra), 4)

    # ------------------------------------------------------------------ anodes and new defects
    def age_anode(self, it: dict, rng):
        a, e = self.cfg['anodes'], self.e['anode']
        if it['status'] == 'intact' and rng.uniform() < 1 - (1 - e['deplete_per_month']) ** self.months:
            L, T, H = it['size_m']
            it['status'] = 'depleted'
            it['size_m'] = _r([L, T * a['depleted_scale'], H * a['depleted_scale']])
        elif it['status'] == 'depleted' and rng.uniform() < 1 - (1 - e['detach_per_month']) ** self.months:
            iid, n0 = it['id'], len(self.items)
            self._missing_anode(float(it['centroid_xs'][0]))  # paints the footprint, records it as item n0 ...
            footprint = self.items.pop()                      # ... and that record takes over the anode's id
            new = self.labels == n0
            self.labels[new] = iid
            self.changed |= new
            it.clear()
            it.update({**footprint, 'id': iid, 'was': 'anode'})

    def add_new(self, name: str, rng):
        band = self.hull.length * (self.cfg['hull']['inspect_girth_m'][1] - self.cfg['hull']['inspect_girth_m'][0])
        for _ in range(int(rng.poisson(self.e[name]['new_per_m2_month'] * band * self.months))):
            mask = self._shape(name, rng)
            pos = self._place(mask, rng)
            if pos is None:
                continue
            r0, c0 = pos
            self._paint_defect(name, r0, c0, mask, rng)
            item = self._record(name, r0, c0, mask)
            item['new_after_months'] = self.months
            if name == 'crack':
                item['length_m'] = round(float(self._crack_len), 4)
            self.changed[r0:r0 + mask.shape[0], c0:c0 + mask.shape[1]] |= mask

    def _restate(self, it: dict):
        """Area, box and centre of a painted item from the label raster."""
        rr, cc = np.nonzero(self.labels == it['id'])
        if rr.size == 0:
            return
        xc, sc = (cc.mean() + 0.5) * self.px, (rr.mean() + 0.5) * self.px
        it.update({'area_m2': round(float(rr.size) * self.px ** 2, 6),
                   'bbox_xs': _r([cc.min() * self.px, rr.min() * self.px, (cc.max() + 1) * self.px, (rr.max() + 1) * self.px]),
                   'centroid_xs': _r([xc, sc]), 'centroid_world': _r(self.hull.point(xc, sc).tolist())})

    def evolve(self) -> Scenario:
        for it in list(self.items):
            rng = self.rng(f'item{it["id"]}')
            if it['class'] in GROWING:
                self.grow_area(it, rng)
            elif it['class'] == 'crack':
                self.grow_crack(it, rng)
            elif it['class'] == 'anode':
                self.age_anode(it, rng)
        for name in ('corrosion', 'coating_damage', 'biofouling', 'crack'):
            self.add_new(name, self.rng(f'new_{name}'))
        for it in self.items:
            if it['class'] != 'anode':
                self._restate(it)
        self._slime_over(self.changed)
        tex = (np.clip(self.tex, 0, 1) * 255 + 0.5).astype(np.uint8)
        lineage = {'base_digest': self.scn.digest(), 'months': self.months, 'evolution': copy.deepcopy(self.e)}
        return Scenario(self.seed, self.cfg, self.hull, tex, self.labels, self.items, self.features, lineage,
                        self.scn.patch_library)

    def _slime_over(self, where: np.ndarray):
        """The scenario's slime film (same draws as the generator) over newly painted pixels only."""
        a = self.cfg['appearance']
        rng = stream(self.seed, 'slime')
        strength = rng.uniform(*a['slime_strength'])
        m = 0.5 * (1.0 + value_noise((self.rows, self.cols), self.m2px(a['slime_scale_m']), rng))
        w = (strength * m)[where][:, None]
        self.tex[where] = self.tex[where] * (1 - w) + np.asarray(a['slime_rgb'], np.float32) * w


def evolve(scn: Scenario, months: float, cfg: dict, library: Library | None = None) -> Scenario:
    """The scenario's hull `months` later (see module notes); run settings are taken from cfg. A hull
    painted with real defects needs the library it was painted from."""
    want = (scn.patch_library or {}).get('library_sha256')
    if want != (library.digest if library else None):
        raise ValueError('this hull was painted procedurally: evolve it without a patch library' if want is None else
                         f'this hull was painted from patch library {want[:12]}: evolve it with that library')
    return _Evolver(scn, months, cfg['evolution'], library).evolve()


def changes(base: Scenario | dict, later: Scenario | dict, growth_ratio: float) -> dict:
    """Ground-truth changes by item id: new, grown (area ratio at least growth_ratio, or a longer crack),
    anode depleted, anode lost, unchanged."""
    a = {it['id']: it for it in (base.items if isinstance(base, Scenario) else base['items'])}
    b = {it['id']: it for it in (later.items if isinstance(later, Scenario) else later['items'])}
    out = {}
    for iid, it in b.items():
        old = a.get(iid)
        if old is None:
            out[iid] = 'new'
        elif old['class'] == 'anode' and it['class'] == 'missing_anode':
            out[iid] = 'anode_lost'
        elif old['class'] == 'anode' and old['status'] != it['status']:
            out[iid] = 'anode_depleted'
        elif 'area_m2' in it and it['area_m2'] >= growth_ratio * old['area_m2']:
            out[iid] = 'grown'
        else:
            out[iid] = 'unchanged'
    return out
