"""The inspection report: a hull photomosaic with the findings, evidence and unverified areas.

Everything in the report proper is the vehicle's own product. The mosaic is built from its frames
placed with its own camera poses: each map cell takes the frame that saw it closest to the image
centre, with that frame's lamp falloff divided out (the baseline's illumination fit, averaged over
the channels so the hull keeps its colour). Findings
(mapping.py) are drawn as boxes, confirmed ones solid and possible ones dashed; each has an evidence
crop from its best view. Areas the vehicle could not verify are taken from its claimed coverage.
A separate appendix, marked as simulation only, scores the map against the scenario's ground truth
and draws findings against the true items.
"""
from __future__ import annotations

import html
import json
import os

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .baseline import illumination
from .camera import focal_px
from .hull import HullGeometry
from .mapping import build, camera_pose, item_box
from .preview import CLASS_COLOURS

FONT = ImageFont.load_default(size=14)


def mosaic(run_dir: str, recs: list[dict], hull: HullGeometry, cfg: dict, cell_m: float) -> np.ndarray:
    """Hull band (rows = girth, cols = length) sampled from the frames at their estimated poses."""
    cams = cfg['cameras']
    f, W, H = focal_px(cams), cams['width'], cams['height']
    g0, g1 = cfg['hull']['inspect_girth_m']
    nx, ns = int(np.ceil(hull.length / cell_m)), int(np.ceil((g1 - g0) / cell_m))
    img = np.zeros((ns, nx, 3), np.float32)
    best = np.full((ns, nx), np.inf)
    xs, ss = (np.arange(nx) + 0.5) * cell_m, g0 + (np.arange(ns) + 0.5) * cell_m
    for rec in recs:
        rgb = np.asarray(Image.open(os.path.join(run_dir, 'frames', rec['frame'] + '.png')).convert('RGB')).astype(np.float32) / 255
        lum = rgb.mean(-1) > cfg['baseline']['dark_level']
        if lum.mean() < 0.05:
            continue
        logc = np.log(np.maximum(rgb, 1e-3))
        fall = illumination(logc, lum, cfg['baseline']).mean(-1, keepdims=True)   # lamp falloff (brightness only)
        flat = rgb / np.exp(fall - np.median(fall[lum]))
        flat = np.clip(flat * (0.45 / max(float(np.median(flat[lum].mean(-1))), 1e-3)), 0, 1)
        C, A = camera_pose(rec)
        X, S = np.meshgrid(xs, ss)
        P = hull.point(X, S)
        pc = (P - C) @ A
        depth = -pc[..., 2]
        with np.errstate(divide='ignore', invalid='ignore'):
            u = f * pc[..., 0] / depth + W / 2
            v = -f * pc[..., 1] / depth + H / 2
        ok = (depth > 0) & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        if not ok.any():
            continue
        r = np.linalg.norm(P - C, axis=-1)
        d = (P - C) / r[..., None]
        ok &= np.abs(hull.raycast(C, d, bounded=False) - r) < 0.02          # not hidden behind the bilge
        centre = np.hypot(u - W / 2, v - H / 2)
        take = ok & (centre < best)
        ui, vi = u[take].astype(int), v[take].astype(int)
        img[take] = flat[vi, ui]
        best[take] = centre[take]
    return (img * 255).astype(np.uint8)


def _dashed_rect(d: ImageDraw.ImageDraw, box, colour, width=2, dash=6):
    x0, y0, x1, y1 = box
    for a, b, horiz, c in ((x0, x1, True, y0), (x0, x1, True, y1), (y0, y1, False, x0), (y0, y1, False, x1)):
        for t in np.arange(a, b, 2 * dash):
            seg = [(t, c), (min(t + dash, b), c)] if horiz else [(c, t), (c, min(t + dash, b))]
            d.line(seg, fill=colour, width=width)


def _to_px(box, g0, k):
    return [box[0] * k, (box[1] - g0) * k, box[2] * k, (box[3] - g0) * k]


def draw_map(base: np.ndarray, findings: list[dict], unverified: np.ndarray | None, g0: float, cell_m: float,
             scale: int = 1) -> Image.Image:
    """The mosaic with unverified areas hatched and findings boxed (solid confirmed, dashed possible)."""
    img = Image.fromarray(base).resize((base.shape[1] * scale, base.shape[0] * scale), Image.NEAREST)
    d = ImageDraw.Draw(img, 'RGBA')
    k = scale / cell_m
    if unverified is not None:
        m = np.asarray(Image.fromarray(unverified.astype(np.uint8) * 255).resize(img.size, Image.NEAREST)) > 0
        hatch = np.zeros(m.shape, bool)
        yy, xx = np.mgrid[0:m.shape[0], 0:m.shape[1]]
        hatch[((xx + yy) // 4) % 3 == 0] = True
        arr = np.asarray(img).copy()
        arr[m & hatch] = (arr[m & hatch] * 0.3 + np.array([255, 60, 60]) * 0.7).astype(np.uint8)
        img = Image.fromarray(arr)
        d = ImageDraw.Draw(img, 'RGBA')
    for fd in findings:
        box = _to_px(fd['hull_box'], g0, k)
        col = CLASS_COLOURS[fd['class']]
        if fd['tier'] == 'confirmed':
            d.rectangle(box, outline=col, width=3)
        else:
            _dashed_rect(d, box, col)
        if fd['tier'] == 'confirmed':
            d.text((box[0], max(0, box[1] - 15)), fd['id'], font=FONT, fill=(255, 255, 255), stroke_width=2, stroke_fill=(0, 0, 0))
    return img


def truth_map(base: np.ndarray, findings: list[dict], spec: dict, inspected: set, g0: float, cell_m: float) -> Image.Image:
    """Simulation only: true items (thin) against findings (green matched, red false, orange duplicate)."""
    img = Image.fromarray(base)
    d = ImageDraw.Draw(img)
    k = 1 / cell_m
    matched = {f['match'].get('item') for f in findings}
    for it in spec['items']:
        box = _to_px(item_box(it), g0, k)
        col = CLASS_COLOURS[it['class']]
        if it['id'] in inspected and it['id'] not in matched:
            d.rectangle([box[0] - 3, box[1] - 3, box[2] + 3, box[3] + 3], outline=(255, 0, 255), width=3)   # missed
        d.rectangle(box, outline=col, width=1)
    for fd in findings:
        m = fd['match']
        col = (40, 220, 60) if 'item' in m else (255, 140, 0) if 'duplicate_of_item' in m else (255, 40, 40)
        d.rectangle(_to_px(fd['hull_box'], g0, k), outline=col, width=2)
    return img


def evidence_crop(run_dir: str, ev: dict, out_path: str, margin: int = 24, size: int = 220):
    rgb = Image.open(os.path.join(run_dir, 'frames', ev['frame'] + '.png')).convert('RGB')
    x0, y0, x1, y1 = ev['box']
    crop = rgb.crop((max(0, x0 - margin), max(0, y0 - margin), min(rgb.width, x1 + margin), min(rgb.height, y1 + margin)))
    crop.thumbnail((size, size))
    d = ImageDraw.Draw(crop)
    s = crop.width / (min(rgb.width, x1 + margin) - max(0, x0 - margin))
    bx0, by0 = (x0 - max(0, x0 - margin)) * s, (y0 - max(0, y0 - margin)) * s
    d.rectangle([bx0, by0, bx0 + (x1 - x0) * s, by0 + (y1 - y0) * s], outline=(255, 255, 0), width=1)
    crop.save(out_path)


def write(run_dir: str, detections: dict, out_dir: str, cfg: dict) -> dict:
    """Findings, mosaic, maps, evidence and report.html for a recorded run (inside its scenario directory)."""
    os.makedirs(os.path.join(out_dir, 'evidence'), exist_ok=True)
    scn_dir = os.path.dirname(os.path.normpath(run_dir))
    spec = json.load(open(os.path.join(scn_dir, 'scenario.json')))
    hull = HullGeometry(**spec['hull'])
    cov = json.load(open(os.path.join(run_dir, 'coverage.json')))
    recs = [r for r in (json.loads(l) for l in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera'] and r.get('signal', True)]
    findings, metrics = build(run_dir, detections, cfg)
    cell = cfg['mapping']['mosaic_cell_m']
    g0 = cfg['hull']['inspect_girth_m'][0]
    base = mosaic(run_dir, recs, hull, cfg, cell)
    claimed_png = os.path.join(run_dir, 'claimed.png')
    claimed = np.asarray(Image.open(claimed_png)) > 0 if os.path.exists(claimed_png) else np.asarray(Image.open(os.path.join(run_dir, 'coverage.png'))) > 0
    claimed_share = float(claimed.mean())
    claimed = np.asarray(Image.fromarray(claimed.astype(np.uint8) * 255).resize((base.shape[1], base.shape[0]), Image.NEAREST)) > 0
    Image.fromarray(base).save(os.path.join(out_dir, 'mosaic.png'))
    draw_map(base, findings, ~claimed, g0, cell).save(os.path.join(out_dir, 'map.png'))
    inspected = {it['id'] for it in cov['items'] if it['inspected']}
    truth_map(base, findings, spec, inspected, g0, cell).save(os.path.join(out_dir, 'map_vs_truth.png'))
    for fd in findings:
        evidence_crop(run_dir, fd['evidence'][0], os.path.join(out_dir, 'evidence', fd['id'] + '.png'))
    review = {}
    review_path = os.path.join(out_dir, 'review.json')
    if os.path.exists(review_path):
        review = json.load(open(review_path))
    for fd in findings:
        fd['review'] = review.get(fd['id'], {'status': 'pending'})
    with open(os.path.join(out_dir, 'findings.json'), 'w') as f:
        json.dump(findings, f, indent=1)
    with open(os.path.join(out_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump({'run': os.path.abspath(run_dir), 'scenario': os.path.abspath(scn_dir), 'scenario_digest': spec.get('digest')}, f, indent=1)
    _html(out_dir, spec, cov, findings, metrics, claimed_share)
    return {'findings': len(findings), 'confirmed': sum(f['tier'] == 'confirmed' for f in findings),
            'claimed_coverage': round(claimed_share, 4), 'map_referee': {k: {x: v[x] for x in ('items_required', 'items_found', 'recall', 'findings', 'precision')}
                                                                          for k, v in metrics.items()}}


def _html(out_dir, spec, cov, findings, metrics, claimed_share):
    e = html.escape
    rows = []
    for fd in sorted(findings, key=lambda f: (f['tier'] != 'confirmed', -f['score'])):
        x0, s0, x1, s1 = fd['hull_box']
        rows.append(f"<tr class='{fd['tier']}'><td>{fd['id']}</td><td>{e(fd['class'].replace('_', ' '))}</td><td>{fd['tier']}</td>"
                    f"<td>{fd['centre'][0]:.2f}</td><td>{fd['centre'][1]:.2f}</td><td>{(x1 - x0) * 100:.0f} x {(s1 - s0) * 100:.0f}</td>"
                    f"<td>{fd['area_m2'] * 1e4:.0f}</td><td>{fd['support']:.2f} ({len(fd['evidence'])}/{fd['seen_in']})</td>"
                    f"<td>{e(fd['review']['status'])}</td><td><img src='evidence/{fd['id']}.png' height='90'></td></tr>")
    health = cov.get('navigation', {}).get('health', [])
    holes = cov.get('claimed', {})
    m = metrics
    appendix = ''.join(
        f"<tr><td>{tier}</td><td>{v['items_found']} / {v['items_required']}</td><td>{v['findings']}</td>"
        f"<td>{v['false_findings']}</td><td>{v['precision']}</td></tr>" for tier, v in m.items())
    per_class = ''.join(
        f"<tr><td>{e(c.replace('_', ' '))}</td><td>{v['found']} / {v['required']}</td><td>{v['findings']}</td><td>{v['false']}</td>"
        f"<td>{v['centre_error_m_median'] if v['centre_error_m_median'] is not None else ''}</td>"
        f"<td>{v['area_ratio_median'] if v['area_ratio_median'] is not None else ''}</td></tr>"
        for c, v in m['all']['per_class'].items())
    doc = f"""<!DOCTYPE html><html><head><meta charset="utf-8"><title>Hull inspection report, scenario {spec['seed']}</title>
<style>body{{font-family:sans-serif;margin:24px;max-width:1500px}} table{{border-collapse:collapse}} td,th{{border:1px solid #ccc;padding:4px 8px;text-align:left}}
tr.possible{{color:#666}} img.map{{width:100%;image-rendering:pixelated}} .sim{{background:#f6f0e6;padding:12px;border:1px solid #d9c9a8}}</style></head><body>
<h1>Hull inspection report</h1>
<p>Scenario seed {spec['seed']}, hull section {spec['hull']['length']:.0f} m long. Simulation: the vehicle, the water and the hull are simulated.</p>
<h2>Coverage</h2>
<p>The vehicle verified {claimed_share * 100:.1f}% of the inspection band; hatched red on the map is not verified
({holes.get('holes_remaining', 0)} hole(s)). Mission {('aborted: ' + e(str(cov['aborted']))) if cov.get('aborted') else 'completed'}.</p>
<h2>Hull map</h2>
<p>Photomosaic from the vehicle's frames (ship length across, girth down from the waterline); solid boxes are confirmed findings,
dashed ones possible.</p>
<img class="map" src="map.png">
<h2>Findings ({sum(f['tier'] == 'confirmed' for f in findings)} confirmed, {sum(f['tier'] == 'possible' for f in findings)} possible)</h2>
<p>x: along the ship from the aft end of the section (m); s: girth down from the waterline (m). Support: share of the frames
that saw the spot in which it was detected.</p>
<table><tr><th>id</th><th>class</th><th>tier</th><th>x (m)</th><th>s (m)</th><th>extent (cm)</th><th>area (cm²)</th>
<th>support</th><th>review</th><th>evidence</th></tr>{''.join(rows)}</table>
<h2>Vehicle health</h2><p>{'<br>'.join(e(f"t={h['t']} s: {h['sensor']} {h['event']}") for h in health) or 'No sensor was excluded.'}</p>
<div class="sim"><h2>Appendix: scored against the simulator's ground truth (simulation only)</h2>
<table><tr><th>findings</th><th>items found</th><th>findings</th><th>false</th><th>precision</th></tr>{appendix}</table>
<h3>All findings by class</h3>
<table><tr><th>class</th><th>items found</th><th>findings</th><th>false</th><th>median centre error (m)</th><th>median area ratio</th></tr>{per_class}</table>
<p>Green: finding matches a true item; red: false finding; orange: duplicate; magenta: an inspected item no finding matches.</p>
<img class="map" src="map_vs_truth.png"></div></body></html>"""
    with open(os.path.join(out_dir, 'report.html'), 'w') as f:
        f.write(doc)
