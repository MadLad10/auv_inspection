"""Procedural texture primitives: smooth noise, shape masks and painting helpers.

All shapes are boolean masks in texture pixels. The scenario module places them on the hull texture
and records their exact pixel areas as ground truth.
"""
from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


def value_noise(shape: tuple[int, int], scale_px: float, rng: np.random.Generator) -> np.ndarray:
    """Smooth noise in roughly [-1, 1] with features about scale_px pixels across."""
    h, w = shape
    gh, gw = max(2, math.ceil(h / max(scale_px, 1.0)) + 2), max(2, math.ceil(w / max(scale_px, 1.0)) + 2)
    lattice = rng.uniform(-1.0, 1.0, (gh, gw)).astype(np.float32)
    big = Image.fromarray(lattice).resize((int((gw - 1) * scale_px), int((gh - 1) * scale_px)), Image.BICUBIC)
    out = np.asarray(big, np.float32)
    if out.shape[0] < h or out.shape[1] < w:
        out = np.asarray(Image.fromarray(out).resize((w, h), Image.BICUBIC), np.float32)
    return np.clip(out[:h, :w], -1.0, 1.0)


def periodic_noise(n: int, scale_px: float, rng: np.random.Generator) -> np.ndarray:
    """Tileable smooth noise in [-1, 1] on an n x n grid (low-pass filtered white noise via the FFT)."""
    k = np.fft.fftfreq(n)
    kk = np.hypot(*np.meshgrid(k, k))
    field = np.real(np.fft.ifft2(np.fft.fft2(rng.normal(size=(n, n))) * np.exp(-(kk * scale_px) ** 2 * 2)))
    return (field / max(np.abs(field).max(), 1e-12)).astype(np.float32)


def blob_mask(area_px: float, aspect: float, irregularity: float, rng: np.random.Generator) -> np.ndarray:
    """Irregular blob: a rotated ellipse whose outline is perturbed by smooth noise."""
    a = math.sqrt(area_px / (math.pi * aspect))
    b = aspect * a
    half = int(math.ceil(a * (1.0 + irregularity) + 2))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(np.float32)
    th = rng.uniform(0, math.pi)
    xr, yr = xx * math.cos(th) + yy * math.sin(th), -xx * math.sin(th) + yy * math.cos(th)
    d = np.sqrt((xr / a) ** 2 + (yr / b) ** 2)
    n = value_noise(d.shape, max(2.0, b / 1.5), rng)
    return _trim(d + irregularity * n < 1.0)


def polygon_mask(area_px: float, aspect: float, vertices: int, irregularity: float, rng: np.random.Generator) -> np.ndarray:
    """Jagged polygon (flaked coating): random radii round a rotated ellipse."""
    a = math.sqrt(area_px / (math.pi * aspect))
    b = aspect * a
    ang = np.sort(rng.uniform(0, 2 * math.pi, vertices))
    rad = 1.0 + irregularity * rng.uniform(-1, 1, vertices)
    th = rng.uniform(0, math.pi)
    px, py = a * rad * np.cos(ang), b * rad * np.sin(ang)
    x, y = px * math.cos(th) - py * math.sin(th), px * math.sin(th) + py * math.cos(th)
    half = int(math.ceil(a * (1 + irregularity))) + 2
    img = Image.new('L', (2 * half + 1, 2 * half + 1), 0)
    ImageDraw.Draw(img).polygon([(float(u + half), float(v + half)) for u, v in zip(x, y)], fill=1)
    return _trim(np.asarray(img, bool))


def crack_mask(length_px: float, width_px: float, step_px: float, turn_std_rad: float,
               rng: np.random.Generator) -> np.ndarray:
    """Thin crack: a random walk with smoothly drifting heading, drawn with a round-jointed line."""
    n = max(2, int(round(length_px / max(step_px, 1e-6))))
    heading = rng.uniform(0, 2 * math.pi) + np.cumsum(rng.normal(0, turn_std_rad, n))
    pts = np.cumsum(np.stack([np.cos(heading), np.sin(heading)], -1) * step_px, 0)
    pts = np.vstack([[0.0, 0.0], pts])
    pad = int(math.ceil(width_px)) + 2
    pts -= pts.min(0) - pad
    w, h = (pts.max(0) + pad + 1).astype(int)
    img = Image.new('L', (int(w), int(h)), 0)
    ImageDraw.Draw(img).line([tuple(p) for p in pts], fill=1, width=max(1, int(round(width_px))), joint='curve')
    return _trim(np.asarray(img, bool))


def erode(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask
    img = Image.fromarray(mask.astype(np.uint8) * 255).filter(ImageFilter.MinFilter(2 * px + 1))
    return np.asarray(img) > 127


def disc(radius_px: float) -> np.ndarray:
    r = max(1, int(math.ceil(radius_px)))
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return xx ** 2 + yy ** 2 <= radius_px ** 2


def _trim(mask: np.ndarray) -> np.ndarray:
    rows, cols = np.flatnonzero(mask.any(1)), np.flatnonzero(mask.any(0))
    if rows.size == 0:
        return np.ones((1, 1), bool)
    return mask[rows[0]:rows[-1] + 1, cols[0]:cols[-1] + 1]
