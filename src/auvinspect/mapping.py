"""The defect map: detections placed on the hull from the vehicle's own camera poses, fused across
frames into findings, and scored against the scenario (the map referee).

Placing: a detection's box is sampled with rays from the camera pose the vehicle estimated for that
frame (in a mission: navigation estimate, AHRS attitude and tilt encoder; in a kinematic survey: the
planned pose) and cast onto the analytic hull, extended past its ends. The hits give the view's box
on the hull in (x along the ship, s down the girth). Its area is the detector's pixel count times
the surface area of a pixel at the box centre, or the box's area on the hull when the detector
gives no pixel count.

Fusing: a view joins a finding of its class when its hull box and one of the finding's views overlap
by at least mapping.merge_overlap of the smaller box (both grown by mapping.merge_gap_m); findings are
built greedily from the highest score. Overlap relative to the smaller box joins partial views of a
large patch, and comparing views rather than the finding's growing union keeps neighbours apart. A finding keeps its views as evidence
(frame, box, score); its hull box is their union, its centre and area are their medians, its score
the highest. Its support is the share of the frames that saw its centre (by the vehicle's own poses:
in the image, unobstructed, within range) that detected it; a real defect is seen again and again,
noise is not. Findings at or above mapping.confirm_support are `confirmed`, the rest `possible`;
none is dropped, since a missed defect costs more than a finding an inspector rejects.

Scoring (simulation only, from the scenario's ground truth), for the confirmed findings and for all
of them: a finding matches an item of its class
when their hull boxes, each grown by mapping.match_tolerance_m, overlap; each item takes at most one
finding, further ones are duplicates. Items are required when the coverage referee counted them
inspected and some frame showed at least perception.min_visible_px of their pixels; inspected items
that never did (thin cracks about a pixel wide) are counted apart as below the camera's resolution.
Reported per class: items required and found, recall by size, false findings (and how many lie on a
decoy such as a sea chest), duplicates, and the centre error and area ratio of matched findings.
"""
from __future__ import annotations

import json
import os

import numpy as np

from .camera import focal_px, pixel_area_mm2
from .dataset import item_area, size_bin
from .hull import HullGeometry


def camera_pose(rec: dict) -> tuple[np.ndarray, np.ndarray]:
    """The camera pose the vehicle believed it had for a frame record (a kinematic survey's planned pose)."""
    ob = rec.get('onboard')
    if ob is None:
        return np.asarray(rec['cam_pos'], float), np.asarray(rec['cam_axes'], float)
    if 'cam_axes_est' not in ob:
        raise ValueError(f'frame {rec["frame"]} has no estimated camera axes; re-run the mission')
    return np.asarray(ob['cam_pos_est'], float), np.asarray(ob['cam_axes_est'], float)


def place(det: dict, rec: dict, hull: HullGeometry, cfg: dict) -> dict | None:
    """One detection on the hull: its box (x0, s0, x1, s1), centre and area; None if it misses the hull."""
    cams = cfg['cameras']
    f, W, H = focal_px(cams), cams['width'], cams['height']
    C, A = camera_pose(rec)
    x0, y0, x1, y1 = det['box']
    n = cfg['mapping']['samples_per_box']
    us, vs = np.meshgrid(np.linspace(x0, x1 - 1, n) + 0.5, np.linspace(y0, y1 - 1, n) + 0.5)
    q = np.stack([(us - W / 2) / f, -(vs - H / 2) / f, -np.ones_like(us)], -1).reshape(-1, 3)
    dirs = q / np.linalg.norm(q, axis=1, keepdims=True) @ A.T
    t = hull.raycast(C, dirs, bounded=False)
    hit = np.isfinite(t)
    if not hit.any():
        return None
    x, s, _ = hull.inverse(C + t[hit, None] * dirs[hit])
    box = [float(x.min()), float(s.min()), float(x.max()), float(s.max())]
    if 'area_px' in det:
        k = len(q) // 2                                         # the centre sample
        if hit[k]:
            sc = float(hull.inverse(C + t[k] * dirs[k])[1])
            depth = t[k] / np.linalg.norm(q[k])
            area = det['area_px'] * float(pixel_area_mm2(depth, q[k], hull.normal(sc) @ A, f)) * 1e-6
        else:
            area = (box[2] - box[0]) * (box[3] - box[1])
    else:
        area = (box[2] - box[0]) * (box[3] - box[1])
    return {'class': det['class'], 'score': float(det['score']), 'frame': rec['frame'], 'box_px': list(det['box']),
            'hull_box': box, 'centre': [float(np.median(x)), float(np.median(s))], 'area_m2': float(area)}


def _boxes_touch(a, b, gap: float) -> bool:
    return a[0] - gap <= b[2] and b[0] - gap <= a[2] and a[1] - gap <= b[3] and b[1] - gap <= a[3]


def _overlap(a, b, gap: float) -> float:
    """Intersection over the smaller box, both grown by gap."""
    a = [a[0] - gap, a[1] - gap, a[2] + gap, a[3] + gap]
    b = [b[0] - gap, b[1] - gap, b[2] + gap, b[3] + gap]
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    return ix * iy / min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1]))


def fuse(views: list[dict], cfg: dict) -> list[dict]:
    """Greedy fusion of views into findings (see module notes)."""
    m = cfg['mapping']
    findings: list[dict] = []
    for v in sorted(views, key=lambda v: -v['score']):
        def link(f):
            return max(_overlap(w['hull_box'], v['hull_box'], m['merge_gap_m']) for w in f['views'])
        f = max((f for f in findings if f['class'] == v['class']), default=None, key=link)
        if f is None or link(f) < m['merge_overlap']:
            findings.append({'class': v['class'], 'hull_box': list(v['hull_box']), 'views': [v]})
            continue
        f['views'].append(v)
        b = f['hull_box']
        f['hull_box'] = [min(b[0], v['hull_box'][0]), min(b[1], v['hull_box'][1]), max(b[2], v['hull_box'][2]), max(b[3], v['hull_box'][3])]
    out = []
    for k, f in enumerate(sorted(findings, key=lambda f: (f['hull_box'][0], f['hull_box'][1]))):
        vs = f['views']
        if len(vs) < cfg['mapping']['min_views']:
            continue
        out.append({'id': f'F{k + 1:03d}', 'class': f['class'], 'score': round(max(v['score'] for v in vs), 4),
                    'views': len(vs), 'hull_box': [round(c, 4) for c in f['hull_box']],
                    'centre': [round(float(np.median([v['centre'][i] for v in vs])), 4) for i in range(2)],
                    'area_m2': round(float(np.median([v['area_m2'] for v in vs])), 5),
                    'evidence': [{'frame': v['frame'], 'box': v['box_px'], 'score': round(v['score'], 4)}
                                 for v in sorted(vs, key=lambda v: -v['score'])]})
    return out


def item_box(it: dict) -> list[float]:
    if 'bbox_xs' in it:
        return list(it['bbox_xs'])
    (xc, sc), (L, _, H) = it['centroid_xs'], it['size_m']
    return [xc - L / 2, sc - H / 2, xc + L / 2, sc + H / 2]


def evaluate_map(findings: list[dict], spec: dict, inspected: set, cfg: dict, unresolved: set | None = None) -> dict:
    """Score findings against the scenario's items (see module notes). Adds 'match' to each finding."""
    tol = cfg['mapping']['match_tolerance_m']
    bins = cfg['perception']['size_bins_m2']
    items = [it for it in spec['items'] if it['class'] in cfg['perception']['classes']]
    decoys = [f['bbox_xs'] for f in spec.get('features', []) if 'bbox_xs' in f]
    taken: dict[int, str] = {}
    for fd in sorted(findings, key=lambda f: -f['score']):
        cands = [it for it in items if it['class'] == fd['class'] and _boxes_touch(item_box(it), fd['hull_box'], 2 * tol)]
        cands.sort(key=lambda it: np.hypot(it['centroid_xs'][0] - fd['centre'][0], it['centroid_xs'][1] - fd['centre'][1]))
        free = [it for it in cands if it['id'] not in taken]
        if free:
            it = free[0]
            taken[it['id']] = fd['id']
            fd['match'] = {'item': it['id'], 'centre_error_m': round(float(np.hypot(it['centroid_xs'][0] - fd['centre'][0],
                                                                                     it['centroid_xs'][1] - fd['centre'][1])), 4),
                           'area_ratio': round(fd['area_m2'] / item_area(it), 3)}
        elif cands:
            fd['match'] = {'duplicate_of_item': cands[0]['id']}
        else:
            fd['match'] = {'false': True, 'on_decoy': any(_boxes_touch(d, fd['hull_box'], tol) for d in decoys)}
    per_class = {}
    for cls in cfg['perception']['classes']:
        req = [it for it in items if it['class'] == cls and it['id'] in inspected]
        mine = [f for f in findings if f['class'] == cls]
        hits = [f for f in mine if 'item' in f['match']]
        sizes = sorted({size_bin(item_area(it), bins) for it in req})
        per_class[cls] = {
            'required': len(req), 'found': sum(1 for it in req if it['id'] in taken),
            'recall': round(sum(1 for it in req if it['id'] in taken) / len(req), 4) if req else None,
            'recall_by_size': {z: round(sum(1 for it in req if it['id'] in taken and size_bin(item_area(it), bins) == z)
                                        / sum(1 for it in req if size_bin(item_area(it), bins) == z), 4) for z in sizes},
            'findings': len(mine), 'false': sum(1 for f in mine if f['match'].get('false')),
            'false_on_decoys': sum(1 for f in mine if f['match'].get('on_decoy')),
            'duplicates': sum(1 for f in mine if 'duplicate_of_item' in f['match']),
            'centre_error_m_median': round(float(np.median([f['match']['centre_error_m'] for f in hits])), 4) if hits else None,
            'area_ratio_median': round(float(np.median([f['match']['area_ratio'] for f in hits])), 3) if hits else None,
        }
    n_req = sum(v['required'] for v in per_class.values())
    n_found = sum(v['found'] for v in per_class.values())
    n_false = sum(v['false'] for v in per_class.values())
    return {'items_required': n_req, 'items_found': n_found, 'recall': round(n_found / n_req, 4) if n_req else None,
            'items_below_resolution': sorted(unresolved or []),
            'findings': len(findings), 'false_findings': n_false,
            'precision': round((len(findings) - n_false - sum(v['duplicates'] for v in per_class.values())) / len(findings), 4)
            if findings else None, 'per_class': per_class}


def frames_seeing(point: np.ndarray, recs: list[dict], hull: HullGeometry, cfg: dict) -> set[str]:
    """Frames whose (estimated) view contains a hull point: in the image, first hit along the ray, in range."""
    cams = cfg['cameras']
    f, W, H = focal_px(cams), cams['width'], cams['height']
    out = set()
    for rec in recs:
        C, A = camera_pose(rec)
        v = point - C
        pc = v @ A
        if pc[2] >= 0:
            continue
        u, w = f * pc[0] / -pc[2] + W / 2, -f * pc[1] / -pc[2] + H / 2
        r = float(np.linalg.norm(v))
        if 0 <= u < W and 0 <= w < H and r <= cfg['coverage']['max_range_m'] \
                and float(hull.raycast(C, v / r, bounded=False)) >= r - 0.02:
            out.add(rec['frame'])
    return out


def confirm(findings: list[dict], recs: list[dict], hull: HullGeometry, cfg: dict) -> list[dict]:
    """Support of each finding across the frames that saw it, and its tier."""
    for fd in findings:
        seen = frames_seeing(hull.point(*fd['centre']), recs, hull, cfg)
        hits = {e['frame'] for e in fd['evidence']}
        fd['seen_in'] = len(seen | hits)
        fd['support'] = round(len(hits) / max(fd['seen_in'], 1), 3)
        fd['tier'] = 'confirmed' if fd['support'] >= cfg['mapping']['confirm_support'] else 'possible'
    return findings


def build(run_dir: str, detections: dict, cfg: dict) -> tuple[list[dict], dict]:
    """Findings for a recorded run (inside its scenario directory) and their score."""
    with open(os.path.join(os.path.dirname(os.path.normpath(run_dir)), 'scenario.json')) as f:
        spec = json.load(f)
    hull = HullGeometry(**spec['hull'])
    recs = {r['frame']: r for r in (json.loads(l) for l in open(os.path.join(run_dir, 'frames.jsonl')))
            if r['camera'] == cfg['survey']['camera']}
    views = []
    for frame, dets in detections.items():
        rec = recs.get(frame)
        if rec is None:
            continue
        for d in dets:
            if d['score'] >= cfg['perception']['score_threshold']:
                v = place(d, rec, hull, cfg)
                if v is not None:
                    views.append(v)
    findings = confirm(fuse(views, cfg), [r for r in recs.values() if r.get('signal', True)], hull, cfg)
    with open(os.path.join(run_dir, 'coverage.json')) as f:
        inspected = {it['id'] for it in json.load(f)['items'] if it['inspected']}
    most = {}
    for r in recs.values():
        for iid, n in r.get('visible_items', {}).items():
            most[int(iid)] = max(most.get(int(iid), 0), n)
    findable = {i for i in inspected if most.get(i, 0) >= cfg['perception']['min_visible_px']}
    confirmed = evaluate_map([dict(f) for f in findings if f['tier'] == 'confirmed'], spec, findable, cfg, inspected - findable)
    full = evaluate_map(findings, spec, findable, cfg, inspected - findable)            # adds each finding's match
    return findings, {'confirmed': confirmed, 'all': full}
