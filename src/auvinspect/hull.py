"""Analytic hull section: a flat side, a circular bilge turn and a flat bottom.

Surface coordinates are (x, s): x along the ship and s the girth (arc length from the waterline down
the side, round the bilge and inboard along the bottom). Every ground-truth quantity (defect
positions, sizes, per-pixel labels, coverage) is computed from this parametrisation, not from the
render mesh, so it is exact up to floating point.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class HullGeometry:
    length: float
    side_depth: float
    bilge_radius: float
    bottom_halfwidth: float
    offset_y: float

    @classmethod
    def from_config(cls, cfg: dict) -> 'HullGeometry':
        h = cfg['hull']
        return cls(h['length_m'], h['side_depth_m'], h['bilge_radius_m'], h['bottom_halfwidth_m'], h['offset_y_m'])

    def to_dict(self) -> dict:
        return asdict(self)

    @property
    def bilge_end(self) -> float:
        return self.side_depth + np.pi / 2 * self.bilge_radius

    @property
    def girth(self) -> float:
        return self.bilge_end + self.bottom_halfwidth

    def turn_angle(self, s) -> np.ndarray:
        """Bilge turn angle at girth s: 0 on the side, pi/2 on the bottom."""
        return np.clip((np.asarray(s, float) - self.side_depth) / self.bilge_radius, 0.0, np.pi / 2)

    def point(self, x, s) -> np.ndarray:
        """World point(s) on the surface, shape (..., 3)."""
        x, s = np.broadcast_arrays(np.asarray(x, float), np.asarray(s, float))
        h, R, y0 = self.side_depth, self.bilge_radius, self.offset_y
        phi = self.turn_angle(s)
        y = np.where(s <= h, y0, np.where(s <= self.bilge_end, y0 + R - R * np.cos(phi), y0 + R + (s - self.bilge_end)))
        z = np.where(s <= h, -s, np.where(s <= self.bilge_end, -h - R * np.sin(phi), -h - R))
        return np.stack([x, y, z], -1)

    def normal(self, s) -> np.ndarray:
        """Outward unit normal(s) (pointing into the water, towards the vehicle side), shape (..., 3)."""
        phi = self.turn_angle(s)
        return np.stack([np.zeros_like(phi), -np.cos(phi), -np.sin(phi)], -1)

    def inverse(self, p) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Surface coordinates (x, s) and signed distance (positive outside) of world point(s) p (..., 3)."""
        p = np.asarray(p, float)
        x, y, z = p[..., 0], p[..., 1], p[..., 2]
        h, R, y0 = self.side_depth, self.bilge_radius, self.offset_y
        side = z >= -h
        bottom = ~side & (y >= y0 + R)
        dy, dz = y - (y0 + R), z + h
        phi = np.clip(np.arctan2(-dz, -dy), 0.0, np.pi / 2)
        s = np.where(side, -z, np.where(bottom, self.bilge_end + (y - y0 - R), h + R * phi))
        dist = np.where(side, y0 - y, np.where(bottom, (-h - R) - z, np.hypot(dy, dz) - R))
        return x, s, dist

    def raycast(self, origin, dirs, bounded: bool = True) -> np.ndarray:
        """Ray parameter t of the first hull hit, origin + t * dirs, for rays from the water side;
        inf where the ray misses (t is a distance when dirs are unit vectors). Shapes: origin (..., 3)
        or (3,), dirs (..., 3). bounded=False extends the section beyond the hull's ends."""
        d = np.asarray(dirs, float)
        o = np.broadcast_to(np.asarray(origin, float), d.shape)
        h, R, y0 = self.side_depth, self.bilge_radius, self.offset_y
        yc, zc = y0 + R, -h
        best = np.full(d.shape[:-1], np.inf)
        with np.errstate(divide='ignore', invalid='ignore'):
            pieces = []
            t = (y0 - o[..., 1]) / d[..., 1]                                     # flat side
            p = o + t[..., None] * d
            pieces.append((t, (d[..., 1] > 0) & (p[..., 2] >= -h) & (p[..., 2] <= 0), p))
            a = d[..., 1] ** 2 + d[..., 2] ** 2                                  # bilge turn (entry root)
            b = 2 * ((o[..., 1] - yc) * d[..., 1] + (o[..., 2] - zc) * d[..., 2])
            c = (o[..., 1] - yc) ** 2 + (o[..., 2] - zc) ** 2 - R ** 2
            disc = b * b - 4 * a * c
            t = (-b - np.sqrt(np.maximum(disc, 0))) / (2 * a)
            p = o + t[..., None] * d
            pieces.append((t, (disc >= 0) & (a > 0) & (p[..., 1] <= yc) & (p[..., 2] <= zc), p))
            t = (-h - R - o[..., 2]) / d[..., 2]                                 # flat bottom
            p = o + t[..., None] * d
            pieces.append((t, (d[..., 2] > 0) & (p[..., 1] >= yc) & (p[..., 1] <= yc + self.bottom_halfwidth), p))
            for t, ok, p in pieces:
                ok = ok & (t > 0)
                if bounded:
                    ok = ok & (p[..., 0] >= 0) & (p[..., 0] <= self.length)
                best = np.where(ok & (t < best), t, best)
        return best

    def mesh(self, cells_x: int, cells_s: int, flip_v: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Render mesh over the whole hull: vertices (N, 3), texture coordinates (N, 2), faces (M, 3).

        Texture u runs along x; v along the girth (row 0 of the texture image is the waterline when
        flip_v matches the renderer's convention). Faces wind so that normals point outward.
        """
        xs = np.linspace(0.0, self.length, cells_x + 1)
        ss = self._girth_samples(cells_s)
        X, S = np.meshgrid(xs, ss, indexing='ij')
        verts = self.point(X, S).reshape(-1, 3)
        u = X / self.length
        v = S / self.girth
        uv = np.stack([u, 1.0 - v if flip_v else v], -1).reshape(-1, 2)
        ns = len(ss)
        i, j = np.meshgrid(np.arange(cells_x), np.arange(ns - 1), indexing='ij')
        v00, v10, v01, v11 = i * ns + j, (i + 1) * ns + j, i * ns + j + 1, (i + 1) * ns + j + 1
        faces = np.concatenate([np.stack([v00, v01, v10], -1).reshape(-1, 3),
                                np.stack([v10, v01, v11], -1).reshape(-1, 3)])
        return verts, uv, faces

    def _girth_samples(self, cells_s: int) -> np.ndarray:
        """Uniform girth samples that always include the bilge-turn endpoints (exact flat/curved joins)."""
        s = np.linspace(0.0, self.girth, cells_s + 1)
        return np.unique(np.concatenate([s, [self.side_depth, self.bilge_end]]))
