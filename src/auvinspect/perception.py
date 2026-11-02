"""The perception referee: detections scored against the exact ground truth.

Detections are boxes per frame, {frame id: [{class, score, box: [x0, y0, x1, y1]}, ...]}, for frames of
a dataset (dataset.py). In each frame and class, detections are taken in order of score and matched
to the unmatched object of that class with the largest box overlap (IoU) of at least
perception.iou_match. A match with a required object is a hit; a match with an ignored object (too
small or too poorly seen in that frame) counts neither way; an unmatched detection is a false alarm.

Reported per class:
  ap               average precision over all scores (all-point interpolation)
  precision/recall at perception.score_threshold (the findings a report would list)
  recall_by_size   per physical size bin of the objects
  items            hull-level recall: of the items required in at least one frame, the share found
                   (a hit at or above the threshold) in at least one frame, also by size
plus the mean AP over classes with required objects and false alarms per frame. Scores are the
simulator's ground truth, never the detector's own confidence.
"""
from __future__ import annotations

import numpy as np


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def average_precision(scores: np.ndarray, hits: np.ndarray, n_required: int) -> float:
    if n_required == 0:
        return float('nan')
    if len(scores) == 0:
        return 0.0
    order = np.argsort(-scores, kind='stable')
    tp = np.cumsum(hits[order])
    fp = np.cumsum(~hits[order])
    recall = np.concatenate([[0.0], tp / n_required, [tp[-1] / n_required]])
    precision = np.concatenate([[1.0], tp / np.maximum(tp + fp, 1), [0.0]])
    precision = np.maximum.accumulate(precision[::-1])[::-1]      # interpolated: best precision at or beyond each recall
    return float(np.sum(np.diff(recall) * precision[1:]))


def match_frame(objects: list[dict], dets: list[dict], cls: str, thr_iou: float):
    """Hits and false alarms of one class in one frame: [(score, hit, matched object or None)]."""
    gts = [o for o in objects if o['class'] == cls]
    used = set()
    out = []
    for d in sorted((d for d in dets if d['class'] == cls), key=lambda d: -d['score']):
        best, best_iou = None, thr_iou
        for k, o in enumerate(gts):
            if k in used:
                continue
            v = iou(d['box'], o['box'])
            if v >= best_iou:
                best, best_iou = k, v
        if best is None:
            out.append((d['score'], False, None))
            continue
        used.add(best)
        if gts[best]['required']:
            out.append((d['score'], True, gts[best]))
        # a match with an ignored object counts neither way
    return out


def evaluate(annotations: dict, detections: dict, cfg: dict, split: str | None = None) -> dict:
    p = cfg['perception']
    fids = annotations['splits'][split] if split else list(annotations['frames'])
    thr = p['score_threshold']
    per_class, aps, false_alarms = {}, [], 0
    for cls in annotations['classes']:
        scores, hits = [], []
        n_req, req_by_size, hit_by_size = 0, {}, {}
        items_req, items_found = {}, set()
        tp_thr = fp_thr = 0
        for fid in fids:
            fr = annotations['frames'][fid]
            objs = fr['objects']
            seed = fr['scenario_seed']                            # item ids are unique within a scenario only
            for o in objs:
                if o['class'] == cls and o['required']:
                    n_req += 1
                    req_by_size[o['size']] = req_by_size.get(o['size'], 0) + 1
                    items_req[(seed, o['item'])] = o['size']
            for score, hit, obj in match_frame(objs, detections.get(fid, []), cls, p['iou_match']):
                scores.append(score)
                hits.append(hit)
                if score >= thr:
                    if hit:
                        tp_thr += 1
                        hit_by_size[obj['size']] = hit_by_size.get(obj['size'], 0) + 1
                        items_found.add((seed, obj['item']))
                    else:
                        fp_thr += 1
        false_alarms += fp_thr
        ap = average_precision(np.asarray(scores, float), np.asarray(hits, bool), n_req)
        if n_req:
            aps.append(ap)
        sizes = sorted(set(items_req.values()))
        per_class[cls] = {
            'required': n_req, 'detections': len(scores), 'ap': None if np.isnan(ap) else round(ap, 4),
            'precision': round(tp_thr / (tp_thr + fp_thr), 4) if tp_thr + fp_thr else None,
            'recall': round(tp_thr / n_req, 4) if n_req else None,
            'recall_by_size': {s: round(hit_by_size.get(s, 0) / n, 4) for s, n in sorted(req_by_size.items())},
            'items': {'required': len(items_req), 'found': len(items_found),
                      'recall': round(len(items_found) / len(items_req), 4) if items_req else None,
                      'recall_by_size': {s: round(sum(1 for i, z in items_req.items() if z == s and i in items_found)
                                                  / sum(1 for z in items_req.values() if z == s), 4) for s in sizes}},
        }
    return {'split': split, 'frames': len(fids), 'score_threshold': thr, 'iou_match': p['iou_match'],
            'map': round(float(np.mean(aps)), 4) if aps else None,
            'false_alarms_per_frame': round(false_alarms / max(len(fids), 1), 4), 'per_class': per_class}
