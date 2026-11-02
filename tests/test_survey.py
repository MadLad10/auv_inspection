import json

import numpy as np
import pytest

from auvinspect import config, planner, scenario, survey


def test_planned_survey_covers_the_band_and_inspects_every_item(tmp_path, small_scn):
    s = survey.run(small_scn, str(tmp_path), save_frames=False)
    assert s['coverage_fraction'] == 1.0
    assert all(it['inspected'] for it in s['items'])
    assert json.load(open(tmp_path / 'coverage.json'))['scenario_digest'] == small_scn.digest()


def test_referee_catches_a_plan_with_a_missing_lane(tmp_path, small_scn, monkeypatch):
    full = planner.lane_girths(small_scn.hull, small_scn.cfg)
    monkeypatch.setattr(planner, 'lane_girths', lambda hull, cfg: np.delete(full, len(full) // 2))
    monkeypatch.setattr(survey, 'lane_plan', planner.lane_plan)
    s = survey.run(small_scn, str(tmp_path), save_frames=False)
    assert s['coverage_fraction'] < 0.99


def test_lanes_overlap_and_reach_both_ends_of_the_band(small_cfg, small_scn):
    sv = small_cfg['survey']
    half = planner.footprint(small_cfg)[0] * (1 - sv['overlap_along']) / 2
    g0, g1 = small_cfg['hull']['inspect_girth_m']
    iv = [planner.lane_interval(small_scn.hull, small_cfg, float(s), half) for s in planner.lane_girths(small_scn.hull, small_cfg)]
    assert iv[0][0] <= g0 and iv[-1][1] >= g1
    for a, b in zip(iv, iv[1:]):
        assert b[0] < a[1]


def test_impossible_standoff_is_refused(small_scn):
    cfg = config.load(overrides={'hull': {'length_m': 6.0}, 'survey': {'standoff_m': 3.5}})
    with pytest.raises(config.ConfigError):
        planner.lane_plan(small_scn.hull, cfg)


def test_saved_frames_carry_ground_truth(tmp_path, small_cfg):
    cfg = config.load(overrides={'hull': {'length_m': 6.0}, 'survey': {'record_cameras': ['port', 'forward']}})
    scn = scenario.generate(11, cfg)
    s = survey.run(scn, str(tmp_path), save_frames=True)
    lines = [json.loads(l) for l in open(tmp_path / 'frames.jsonl')]
    assert len(lines) == s['frames'] == 2 * s['poses']
    gt = np.load(tmp_path / 'gt' / (lines[0]['frame'] + '.npz'))
    assert gt['label'].shape == (cfg['cameras']['height'], cfg['cameras']['width'])
    seen = set().union(*(map(int, l['visible_items']) for l in lines))
    assert {d['id'] for d in scn.defects()} <= seen
