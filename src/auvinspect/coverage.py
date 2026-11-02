"""The referee: which part of the hull, and which items, were seen well enough to inspect.

A hull cell (coverage.cell_m square in surface coordinates, inside the inspection band) is covered
when at least one inspection-camera pixel lands in it within the configured limits, in a frame that
carried an image and was not blurred beyond coverage.max_blur_px. The limits are the range, the
incidence angle, the pixel footprint on the hull and the contrast-to-noise ratio a faint defect
(coverage.reflectance_step on coverage.reference_albedo) reaches over one cell, from the image
physics (optics.py). A defect counts as inspected when the covered share of its area reaches
coverage.defect_seen_fraction.
Everything is computed from the scenario's ground truth, never from a perception result.

`CellGrid` is the bare cell grid. The vehicle keeps its own grid of what it believes it covered
(onboard.py); only `Coverage`, the referee, reads the scenario.
"""
from __future__ import annotations

import math

import numpy as np

from .hull import HullGeometry
from .render import Frame
from .scenario import Scenario


def within_limits(c: dict, range_m, incidence_deg, gsd_mm, cell_cnr) -> np.ndarray:
    """Per-pixel test against the coverage limits (c is the coverage configuration section)."""
    return ((range_m <= c['max_range_m']) & (incidence_deg <= c['max_incidence_deg'])
            & (gsd_mm <= c['max_gsd_mm']) & (cell_cnr >= c['min_cell_cnr']))


class CellGrid:
    """Hull cells inside the inspection band (rows = girth, cols = length) and how often each was hit."""

    def __init__(self, hull: HullGeometry, cfg: dict):
        self.c = cfg['coverage']
        self.cell = self.c['cell_m']
        self.g0, self.g1 = cfg['hull']['inspect_girth_m']
        self.nx = math.ceil(hull.length / self.cell)
        self.ns = math.ceil((self.g1 - self.g0) / self.cell)
        self.hits = np.zeros((self.ns, self.nx), np.int32)

    def cells(self, x: np.ndarray, s: np.ndarray) -> np.ndarray:
        """Flat cell index for surface points inside the band; -1 outside."""
        ix = np.floor(np.asarray(x) / self.cell).astype(np.int64)
        js = np.floor((np.asarray(s) - self.g0) / self.cell).astype(np.int64)
        ok = (ix >= 0) & (ix < self.nx) & (js >= 0) & (js < self.ns)
        return np.where(ok, js * self.nx + ix, -1)

    def add_points(self, x: np.ndarray, s: np.ndarray) -> int:
        """Count each cell hit by these surface points once; return the number of newly covered cells."""
        cells = self.cells(x, s)
        return self.add_cells(np.unique(cells[cells >= 0]))

    def add_cells(self, cells: np.ndarray) -> int:
        """Count each of these (distinct) flat cell indices once; return the number newly covered."""
        new = int((self.hits.flat[cells] == 0).sum())
        self.hits.flat[cells] += 1
        return new

    def eroded_cells(self, x: np.ndarray, s: np.ndarray, margin_x_m: float, margin_s_m: float) -> np.ndarray:
        """Cells hit by these points that stay hit if the points shift by up to the margins along the
        ship and along the girth: the footprint shrunk by the margins (separable erosion). Points may lie
        outside the grid (past the hull ends on the extended surface, beyond the band): they hold the
        footprint's edge out there, so only its real edges shrink."""
        if len(x) == 0:
            return np.zeros(0, np.int64)
        ix_all = np.floor(np.asarray(x) / self.cell).astype(np.int64)
        js_all = np.floor((np.asarray(s) - self.g0) / self.cell).astype(np.int64)
        pairs = np.unique(np.stack([js_all, ix_all], 1), axis=0)
        js, ix = pairs[:, 0], pairs[:, 1]
        rx, rs = int(math.ceil(margin_x_m / self.cell)), int(math.ceil(margin_s_m / self.cell))
        j0, i0 = js.min() - rs - 1, ix.min() - rx - 1
        m = np.zeros((js.max() - j0 + rs + 2, ix.max() - i0 + rx + 2), bool)
        m[js - j0, ix - i0] = True
        for axis, r in ((1, rx), (0, rs)):
            if r > 0:
                c = np.cumsum(np.pad(m, [(r + 1, r) if a == axis else (0, 0) for a in range(2)]).astype(np.int32), axis=axis)
                hi = np.take(c, np.arange(2 * r + 1, c.shape[axis]), axis=axis)
                lo = np.take(c, np.arange(0, c.shape[axis] - 2 * r - 1), axis=axis)
                m = (hi - lo) == 2 * r + 1                      # every cell within r is hit
        jj, ii = np.nonzero(m)
        jj, ii = jj + j0, ii + i0
        ok = (jj >= 0) & (jj < self.ns) & (ii >= 0) & (ii < self.nx)
        return jj[ok] * self.nx + ii[ok]

    def covered(self) -> np.ndarray:
        return self.hits > 0

    def fraction(self) -> float:
        return float(self.covered().mean())

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        """Hull (x, s) of every cell centre, each shaped (ns, nx)."""
        return np.meshgrid((np.arange(self.nx) + 0.5) * self.cell, self.g0 + (np.arange(self.ns) + 0.5) * self.cell)

    def image(self) -> np.ndarray:
        """Coverage map (rows = girth, cols = length): 0 uncovered, 255 covered."""
        return (self.covered() * 255).astype(np.uint8)


class Coverage(CellGrid):
    def __init__(self, scn: Scenario):
        super().__init__(scn.hull, scn.cfg)
        self.scn = scn

    def quality(self, f: Frame) -> np.ndarray:
        if not f.signal or f.blur_px > self.c['max_blur_px']:
            return np.zeros(f.label.shape, bool)
        return (f.on_hull | f.on_anode) & within_limits(self.c, f.range, f.incidence_deg, f.gsd_mm, f.cnr)

    def add(self, f: Frame) -> int:
        """Count each covered cell once per frame; return the number of newly covered cells."""
        m = self.quality(f)
        return self.add_points(f.x[m], f.s[m])

    def item_status(self) -> list[dict]:
        scn, px = self.scn, self.scn.mm_per_px / 1000.0
        covered = self.covered().ravel()
        rr, cc = np.nonzero(scn.labels >= 0)
        ids = scn.labels[rr, cc]
        cells = self.cells((cc + 0.5) * px, (rr + 0.5) * px)
        out = []
        for it in scn.items:
            if it['class'] == 'anode':
                L, _, H = it['size_m']
                xc, sc = it['centroid_xs']
                xg, sg = np.meshgrid(np.arange(xc - L / 2, xc + L / 2, self.cell / 2), np.arange(sc - H / 2, sc + H / 2, self.cell / 2))
                ic = self.cells(xg.ravel(), sg.ravel())
            else:
                ic = cells[ids == it['id']]
            ic = ic[ic >= 0]
            frac = float(covered[ic].mean()) if ic.size else 0.0
            out.append({'id': it['id'], 'class': it['class'], 'status': it.get('status'),
                        'covered_fraction': round(frac, 4), 'inspected': frac >= self.c['defect_seen_fraction']})
        return out

    def summary(self) -> dict:
        items = self.item_status()
        by_class: dict[str, dict] = {}
        for it in items:
            b = by_class.setdefault(it['class'], {'total': 0, 'inspected': 0})
            b['total'] += 1
            b['inspected'] += int(it['inspected'])
        return {'coverage_fraction': round(self.fraction(), 5), 'cells': [self.ns, self.nx], 'cell_m': self.cell,
                'inspect_girth_m': [self.g0, self.g1], 'by_class': by_class, 'items': items}
