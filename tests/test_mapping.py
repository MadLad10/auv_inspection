"""The defect map and the inspection report."""
import json

import numpy as np
import pytest

from auvinspect import mapping, report


def _perfect_detections(root, cfg):
    """Ground-truth boxes as detections, with their pixel counts."""
    run = root / 'scn' / 'survey'
    spec = json.load(open(root / 'scn' / 'scenario.json'))
    classes = {it['id']: it['class'] for it in spec['items']}
    det = {}
    for line in open(run / 'frames.jsonl'):
        rec = json.loads(line)
        lab = np.load(run / 'gt' / (rec['frame'] + '.npz'))['label']
        out = []
        for iid in np.unique(lab[lab >= 0]):
            m = lab == iid
            if m.sum() < cfg['perception']['min_visible_px']:
                continue
            rows, cols = np.nonzero(m)
            out.append({'class': classes[int(iid)], 'score': 0.9, 'box': [int(cols.min()), int(rows.min()), int(cols.max()) + 1, int(rows.max()) + 1],
                        'area_px': int(m.sum())})
        det[rec['frame']] = out
    return det


@pytest.fixture(scope='module')
def perfect_map(small_survey, small_scn):
    det = _perfect_detections(small_survey, small_scn.cfg)
    return det, *mapping.build(str(small_survey / 'scn' / 'survey'), det, small_scn.cfg)


def test_perfect_detections_map_every_inspected_item_in_place(perfect_map):
    _, findings, metrics = perfect_map
    m = metrics['all']
    assert m['items_found'] == m['items_required'] > 0 and m['false_findings'] == 0
    for cls, v in m['per_class'].items():
        if v['found'] and cls != 'anode':
            assert v['centre_error_m_median'] < 0.03                  # the planned poses place them on the hull
            assert 0.8 < v['area_ratio_median'] < 1.25               # pixel counts times exact pixel areas
    assert all(f['tier'] == 'confirmed' for f in findings if 'item' in f['match'])


def test_views_of_one_patch_fuse_and_neighbours_stay_apart(small_cfg):
    def view(box, score=0.9):
        return {'class': 'corrosion', 'score': score, 'frame': 'f', 'box_px': [0, 0, 1, 1], 'hull_box': box,
                'centre': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], 'area_m2': (box[2] - box[0]) * (box[3] - box[1])}
    big = [view([1.0, 1.0, 1.6, 1.4]), view([1.0, 1.0, 1.3, 1.4], 0.8), view([1.3, 1.1, 1.6, 1.4], 0.7)]   # partial views
    neighbour = [view([1.75, 1.0, 1.95, 1.2], 0.85)]                                                        # 15 cm away
    out = mapping.fuse(big + neighbour, small_cfg)
    assert sorted(f['views'] for f in out) == [1, 3]


def test_finding_area_from_whole_views_or_the_union_of_cut_ones(small_cfg):
    def view(box, area, clipped):
        return {'class': 'biofouling', 'score': 0.9, 'frame': 'f', 'box_px': [0, 0, 1, 1], 'hull_box': box,
                'centre': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2], 'area_m2': area, 'clipped': clipped}
    halves = [view([0.0, 0.0, 0.6, 0.6], 0.18, True), view([0.3, 0.0, 1.0, 0.6], 0.21, True)]   # a 1.0 x 0.6 m patch, half filled
    [f] = mapping.fuse(halves, small_cfg)
    assert f['area_m2'] == pytest.approx(0.30, rel=0.03)               # union 0.6 m2 times fill 0.5, not one half
    [f] = mapping.fuse(halves + [view([0.0, 0.0, 1.0, 0.6], 0.31, False)], small_cfg)
    assert f['area_m2'] == pytest.approx(0.31)                         # a view that saw it whole decides


def test_report_has_findings_maps_evidence_and_reviews(tmp_path, small_survey, small_scn, perfect_map):
    det, findings, _ = perfect_map
    out = tmp_path / 'report'
    out.mkdir()
    fid = findings[0]['id']
    (out / 'review.json').write_text(json.dumps({fid: {'status': 'confirmed', 'note': 'checked'}}))
    s = report.write(str(small_survey / 'scn' / 'survey'), det, str(out), small_scn.cfg)
    assert s['findings'] == len(findings)
    for name in ('report.html', 'map.png', 'map_vs_truth.png', 'mosaic.png', 'findings.json', 'metrics.json'):
        assert (out / name).exists()
    saved = json.load(open(out / 'findings.json'))
    assert all((out / 'evidence' / (f['id'] + '.png')).exists() for f in saved)
    assert next(f for f in saved if f['id'] == fid)['review']['status'] == 'confirmed'
    html = (out / 'report.html').read_text()
    assert fid in html and 'simulation only' in html
