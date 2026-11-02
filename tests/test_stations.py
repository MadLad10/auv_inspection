"""Anode stations: the hull plan, the station check and its classifier."""
import copy
import json

import numpy as np
import pytest

from auvinspect import config, dataset, mapping, stations
from auvinspect.hull import anode_stations


def test_the_plan_holds_every_generated_anode(small_scn):
    plan = anode_stations(small_scn.cfg)
    fitted = [it for it in small_scn.items if it['class'] in ('anode', 'missing_anode')]
    assert len(plan) == len(fitted) > 0
    spec = {'config': small_scn.cfg, 'items': small_scn.items}
    true = stations.truth(spec)
    assert sorted(true) == [st['id'] for st in plan]
    assert sorted(t['item'] for t in true.values()) == sorted(it['id'] for it in fitted)


def test_the_station_check_on_the_small_survey(small_survey, small_scn):
    run = str(small_survey / 'scn' / 'survey')
    results, metrics = stations.check(run, small_scn.cfg)
    assert metrics['inspected'] == metrics['stations'] > 0
    assert metrics['correct'] == metrics['inspected']                   # unseen hull, packaged model
    assert all(0 < r['confidence'] <= 1 and r['views'] > 0 and r['evidence'] for r in results)
    masks = stations.window_masks(run, small_scn.cfg)
    assert masks and all(m.any() for m in masks.values())


def test_detections_at_a_station_make_no_findings(small_survey, small_scn):
    cfg = small_scn.cfg
    run = str(small_survey / 'scn' / 'survey')
    v = stations.measure(run, cfg)[0]                                   # a frame seeing a station whole
    x0, y0, x1, y1 = v['box']
    det = {v['frame']: [{'class': 'coating_damage', 'score': 0.9, 'box': v['box'], 'area_px': 500},       # the anode's glare
                        {'class': 'anode', 'score': 0.9, 'box': [x0, y0 + 1, x1, y1], 'area_px': 500},    # left to the stations
                        {'class': 'corrosion', 'score': 0.9, 'box': [x0, y1 + 60, x0 + 40, y1 + 100], 'area_px': 900}]}
    findings, metrics = mapping.build(run, det, cfg)
    assert [f['class'] for f in findings] == ['corrosion']
    assert 'anode' not in metrics['all']['per_class'] and 'missing_anode' not in metrics['all']['per_class']


def test_the_classifier_votes_over_views_and_never_guesses_unseen_stations():
    rng = np.random.default_rng(0)
    centres = np.array([[0.10, 0.3, 0.9, 1.2, 0.5, 0.9, 1.2], [0.06, 0.1, 0.4, 0.6, 0.1, 0.5, 0.7], [0.04, 0.1, 0.3, 0.3, 0.1, 0.3, 0.3]])
    y = np.repeat([0, 1, 2], 40)
    X = centres[y] + rng.normal(0, 0.03, (len(y), 7))
    paint = [0.4, 0.15, 0.11]
    model = {'statuses': list(stations.STATUSES), 'paints': {stations.paint_key(paint): stations.fit(X, y)}}
    plan = [{'id': f'S0{k}', 'x': float(k), 's': 2.6, 'box': [k - 0.15, 2.5, k + 0.15, 2.7]} for k in range(1, 5)]
    views = [{'station': f'S0{k + 1}', 'frame': f'f{j}', 'box': [0, 0, 10, 10], 'offcentre_px': float(j),
              'features': (centres[k] + rng.normal(0, 0.03, 7)).tolist()} for k in range(3) for j in range(3)]
    views.append({**views[0], 'frame': 'odd', 'features': centres[2].tolist()})   # one odd view does not outvote three
    out = stations.classify(views, model, plan, 50.0, paint)
    assert [r['status'] for r in out] == ['intact', 'depleted', 'missing', 'not_inspected']
    assert out[0]['evidence']['frame'] == 'f0' and out[3]['confidence'] is None
    odd = [{**views[0], 'features': (centres[0] + 1.0).tolist()}]                # like nothing it was trained on
    assert stations.classify(odd, model, plan, 50.0, paint)[0]['status'] == 'unclear'
    other_paint = stations.classify(views, model, plan, 50.0, [0.09, 0.09, 0.10])   # no model for this paint
    assert [r['status'] for r in other_paint] == ['unclear', 'unclear', 'unclear', 'not_inspected']
    with pytest.raises(ValueError):
        stations.fit(X[y < 2], y[y < 2])                                  # no missing anode to learn from


def test_the_packaged_model_matches_the_settings_and_was_trained_on_training_hulls():
    cfg = config.load()
    model = stations.load_model(cfg)
    assert model['statuses'] == list(stations.STATUSES) and model['features'] == list(stations.FEATURES)
    assert all(dataset.split_of(t['seed'], cfg['perception']['splits']) == 'train' for t in model['trained_on'])
    paints = {stations.paint_key(p) for p in cfg['benchmark']['world']['paint_rgb']}
    assert paints <= set(model['paints'])                                # a discriminant for every benchmark paint
    other = copy.deepcopy(cfg)
    other['baseline']['threshold_mad'] += 1.0
    with pytest.raises(ValueError):
        stations.load_model(other)
    assert json.dumps(model)                                           # plain JSON
