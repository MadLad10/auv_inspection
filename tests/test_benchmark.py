"""The benchmark: per-hull worlds, pooled scores with hull-resampling intervals, the gate."""
import copy
import json

import numpy as np
import pytest

from auvinspect import benchmark, config, optics
from auvinspect.mapping import map_classes


def test_each_hull_draws_its_water_and_paint_from_its_seed():
    cfg = config.load()
    suites = cfg['benchmark']['suites']
    w = cfg['benchmark']['world']
    for name in ('ci', 'test'):
        for seed in suites[name]['seeds']:
            wcfg, drawn = benchmark.world(seed, cfg, suites[name])
            assert benchmark.world(seed, cfg, suites[name])[1] == drawn                      # deterministic
            assert w['visibility_m'][0] <= drawn['visibility_m'] <= w['visibility_m'][1]
            assert optics.visibility_m(wcfg['water']) == pytest.approx(drawn['visibility_m'], abs=0.01)
            assert drawn['paint_rgb'] in w['paint_rgb'] and wcfg['appearance']['paint_rgb'] == drawn['paint_rgb']
            assert wcfg['hull']['length_m'] == (8.0 if name == 'ci' else cfg['hull']['length_m'])
    assert cfg['water'] == config.load()['water']                                              # the defaults untouched


def test_the_coverage_criterion_judges_against_the_hulls_paint():
    cfg = config.load()
    s1, back, ranges = np.full((4, 3), 2.0e5), np.full((4, 3), 1.0e4), np.full(4, 1.0)
    explicit = copy.deepcopy(cfg)
    explicit['coverage']['reference_albedo'] = list(cfg['appearance']['paint_rgb'])
    a = optics.background_rate(s1, back, ranges, cfg['water'], cfg)
    assert np.allclose(a, optics.background_rate(s1, back, ranges, cfg['water'], explicit))
    black = copy.deepcopy(cfg)
    black['appearance']['paint_rgb'] = [0.09, 0.09, 0.10]
    assert np.all(optics.background_rate(s1, back, ranges, cfg['water'], black)[:, 0] < a[:, 0])


def _result(seed, found, required, false, correct, inspected=7, coverage=1.0):
    classes = map_classes(config.load())
    tier = {'required': required, 'found': found, 'findings': found + false, 'false': false, 'duplicates': 0,
            'per_class': {c: {'required': required if c == 'corrosion' else 0, 'found': found if c == 'corrosion' else 0, 'false': 0}
                          for c in classes}}
    per_status = {s: {'true': 0, 'found': 0, 'false': 0} for s in ('intact', 'depleted', 'missing')}
    per_status['intact'] = {'true': inspected, 'found': correct, 'false': 0}
    return {'seed': seed, 'world': {'visibility_m': 10.0, 'paint_rgb': [0.4, 0.15, 0.11]}, 'coverage': coverage, 'false_claim': 0.0,
            'aborted': None, 'map': {'confirmed': tier, 'all': tier},
            'stations': {'stations': 7, 'inspected': inspected, 'correct': correct, 'unclear': 0, 'wrong': inspected - correct,
                         'per_status': per_status},
            'versions': {'auvinspect': 'test'}}


def test_scores_pool_over_hulls_and_the_gate_reads_them(tmp_path):
    cfg = config.load()
    rs = [_result(1, 8, 10, 1, 7), _result(2, 5, 10, 3, 6), _result(3, 10, 10, 0, 7, coverage=0.96)]
    assert benchmark.score(rs, cfg, 'unit')['map_recall_confirmed']['low'] is None            # three hulls: no interval
    cfg['benchmark']['min_hulls_for_interval'] = 3
    v = benchmark.score(rs, cfg, 'unit')
    assert v['map_recall_confirmed']['value'] == pytest.approx(23 / 30, abs=1e-4)
    assert v['map_precision_confirmed']['value'] == pytest.approx(23 / 27, abs=1e-4)
    assert v['station_accuracy']['value'] == pytest.approx(20 / 21, abs=1e-4)
    assert v['coverage']['value'] == pytest.approx(2.96 / 3, abs=1e-4)
    for x in v.values():
        if x['value'] is not None:
            assert x['low'] <= x['value'] <= x['high']
    assert v['map_recall_confirmed']['low'] < v['map_recall_confirmed']['high']                # three hulls: a real spread
    assert v['recall_all_crack']['value'] is None                                             # no cracks: no value, not zero
    summary = {'suite': 'ci', 'missing': [4], 'per_hull': [{'seed': 1, 'aborted': None}], 'values': v}
    cfg['benchmark']['suites']['ci']['gate'] = {'min': {'map_recall_confirmed': 0.7}, 'max': {'false_claim': 0.01}}
    assert benchmark.gate(summary, cfg) == ['hull s4 has no result']
    cfg['benchmark']['suites']['ci']['gate']['min']['station_accuracy'] = 0.99
    assert benchmark.gate({**summary, 'missing': []}, cfg) == [f"station_accuracy = {v['station_accuracy']['value']} below 0.99"]


def test_reused_missions_must_be_this_suites_hulls(tmp_path):
    cfg = config.load()
    other = tmp_path / 'other'
    with pytest.raises(ValueError, match='no flown mission'):
        benchmark.run_hull(401, 'ci', str(tmp_path / 'out'), cfg, missions=str(other))
    (other / 's401' / 'mission').mkdir(parents=True)
    (other / 's401' / 'mission' / 'coverage.json').write_text('{}')
    (other / 's401' / 'scenario.json').write_text(json.dumps({'digest': 'not this hull'}))
    with pytest.raises(ValueError, match='not this hull of the suite'):
        benchmark.run_hull(401, 'ci', str(tmp_path / 'out'), cfg, missions=str(other))
