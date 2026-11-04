"""A fleet on the small hull: lanes shared out, one clock, the fleet referee."""
import json

import pytest

from auvinspect import fleet, observe


def _lanes(run_dir):
    return {json.loads(line)['lane'] for line in open(run_dir / 'frames.jsonl')} - {None}


def test_lanes_are_shared_in_contiguous_groups_with_bands_between_them():
    assert fleet.lane_groups(4, 2) == [[0, 1], [2, 3]] and fleet.lane_groups(4, 3) == [[0, 1], [2], [3]]
    assert fleet.lane_groups(4, 4) == [[0], [1], [2], [3]]
    for n in (0, 5):
        with pytest.raises(ValueError):
            fleet.lane_groups(4, n)
    edges = fleet.band_edges([1.0, 2.0, 3.0, 4.0], [[0, 1], [2, 3]])
    assert edges[1] == 2.5 and edges[0] == float('-inf') and edges[2] == float('inf')


def test_a_fleet_of_one_is_the_single_vehicle_mission(tmp_path, small_scn, small_mission):
    single, out = small_mission
    s = fleet.run(small_scn, str(tmp_path), vehicles=1, save_frames=False)
    v0 = tmp_path / 'v0'
    assert (v0 / 'frames.jsonl').read_text() == (out / 'frames.jsonl').read_text()       # shared renderer and referee change nothing
    assert (v0 / observe.EVENTS_FILE).read_text() == (out / observe.EVENTS_FILE).read_text()
    assert s['makespan_s'] == single['sim_time_s'] and s['coverage_fraction'] == single['coverage_fraction']
    assert s['separation']['min_m'] is None


def test_two_vehicles_cover_the_hull_in_about_half_the_time_and_keep_apart(tmp_path, small_scn, small_mission, small_cfg):
    single, _ = small_mission
    s = fleet.run(small_scn, str(tmp_path), vehicles=2, save_frames=False)
    assert s['coverage_fraction'] == 1.0 and all(it['inspected'] for it in s['items'])
    assert s['claimed']['fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0 and s['aborted'] == []
    assert s['makespan_s'] < 0.6 * single['sim_time_s']
    sep = s['separation']
    assert sep['min_m'] > small_cfg['fleet']['min_separation_m'] and sep['ticks_below'] == 0
    for v in s['per_vehicle']:
        d = tmp_path / f'v{v["vehicle"]}'
        assert _lanes(d) == set(v['lanes'])                                                 # each flies its own share
        assert not observe.health(str(d), small_cfg)[1]
    assert s['per_vehicle'][0]['band_m'] == [None, pytest.approx(s['per_vehicle'][1]['band_m'][0])]
    m = {n: v for n, _, v in observe.parse_prom((tmp_path / observe.METRICS_FILE).read_text())}
    assert m['auvinspect_fleet_makespan_seconds'] == s['makespan_s'] and m['auvinspect_referee_fleet_coverage_fraction'] == 1.0
