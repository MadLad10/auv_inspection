"""Human-readable previews: the camera rig at one pose with label overlays, and the annotated texture."""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw

from .render import LABEL_WATER, InspectionRenderer
from .rng import stream
from .scenario import Scenario

CLASS_COLOURS = {
    'corrosion': (255, 140, 0), 'coating_damage': (0, 200, 255), 'biofouling': (60, 220, 60),
    'crack': (255, 0, 200), 'missing_anode': (255, 255, 0), 'anode': (180, 180, 255),
}


def label_overlay(rgb: np.ndarray, label: np.ndarray, scn: Scenario, alpha: float = 0.5) -> np.ndarray:
    out = rgb.astype(np.float32).copy()
    cls = {it['id']: it['class'] for it in scn.items}
    for iid in np.unique(label[label >= 0]):
        m = label == iid
        out[m] = (1 - alpha) * out[m] + alpha * np.asarray(CLASS_COLOURS[cls[int(iid)]], np.float32)
    out[label == LABEL_WATER] *= 0.6
    return out.astype(np.uint8)


def rig_sheet(scn: Scenario, x: float, s: float, standoff: float) -> Image.Image:
    """All rig cameras at one pose (inspection camera facing the hull at girth s): images over overlays."""
    cfg, hull = scn.cfg, scn.hull
    cam = cfg['survey']['camera']
    rend = InspectionRenderer(scn)
    try:
        pos = hull.point(x, s) + standoff * hull.normal(s) - np.asarray(cfg['cameras']['rig'][cam]['offset'])
        rend.set_pose(pos, 0.0, {cam: float(np.degrees(hull.turn_angle(s)))})
        rng = stream(scn.seed, 'preview')
        frames = [rend.capture(c, rng) for c in cfg['cameras']['rig']]
    finally:
        rend.close()
    W, H = cfg['cameras']['width'], cfg['cameras']['height']
    sheet = Image.new('RGB', (W * len(frames), H * 2), (0, 0, 0))
    d = ImageDraw.Draw(sheet)
    for k, f in enumerate(frames):
        sheet.paste(Image.fromarray(f.rgb), (k * W, 0))
        sheet.paste(Image.fromarray(label_overlay(f.albedo, f.label, scn)), (k * W, H))
        d.text((k * W + 8, 8), f'{f.camera}  gain {f.gain:.1f}', fill=(255, 255, 255))
    return sheet


def texture_sheet(scn: Scenario, max_width: int = 2000) -> Image.Image:
    """The hull texture (girth downwards, length rightwards) with every item's bounding box."""
    img = Image.fromarray(scn.texture)
    k = min(1.0, max_width / img.width)
    img = img.resize((int(img.width * k), int(img.height * k)), Image.BILINEAR)
    d = ImageDraw.Draw(img)
    px = scn.mm_per_px / 1000.0 / k
    for it in scn.items:
        if 'bbox_xs' in it:
            x0, s0, x1, s1 = it['bbox_xs']
        else:
            (xc, sc), (L, _, H) = it['centroid_xs'], it['size_m']
            x0, s0, x1, s1 = xc - L / 2, sc - H / 2, xc + L / 2, sc + H / 2
        d.rectangle([x0 / px - 2, s0 / px - 2, x1 / px + 2, s1 / px + 2], outline=CLASS_COLOURS[it['class']], width=2)
    g0, g1 = scn.cfg['hull']['inspect_girth_m']
    for g in (g0, g1):
        d.line([(0, g / px), (img.width, g / px)], fill=(255, 255, 255), width=1)
    return img
