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
    merged = [json.loads(line) for line in open(tmp_path / 'merged' / 'frames.jsonl')]          # the fleet as one run
    assert {r['vehicle'] for r in merged} == {0, 1} and all(r['frame'].startswith(f'v{r["vehicle"]}_') for r in merged)
    assert len(merged) == sum(len(open(tmp_path / f'v{k}' / 'frames.jsonl').readlines()) for k in (0, 1))
    cov = json.load(open(tmp_path / 'merged' / 'coverage.json'))
    assert cov['fleet'] and cov['vehicles'] == 2 and cov['aborted'] is None and cov['coverage_fraction'] == 1.0
    assert json.load(open(tmp_path / 'scenario.json'))['digest'] == small_scn.digest()


def _with_fleet(scn, acoustic=None, **fleet_cfg):
    import dataclasses
    from auvinspect import config
    from conftest import SMALL
    over = {**SMALL, 'fleet': fleet_cfg, **({'acoustic': acoustic} if acoustic else {})}
    return dataclasses.replace(scn, cfg=config.merge_world(config.load(overrides=over), scn.cfg))


BATTERY_DROP = {1: [{'type': 'battery_drop', 't_start': 30.0, 'params': {'drop_pct': 95.0}}]}


def test_an_aborted_vehicles_lanes_are_handed_over_and_flown(tmp_path, small_scn, small_cfg):
    s = fleet.run(_with_fleet(small_scn, vehicle_faults=BATTERY_DROP), str(tmp_path), vehicles=2, save_frames=False)
    assert s['aborted'] == [1] and s['coordination']
    (h,) = s['handovers']
    assert h['from'] == 1 and h['to'] == 0 and h['lanes'] == [2, 3] and h['orphaned_t'] < h['assigned_t'] < h['flown_t']
    assert s['lanes']['not_flown'] == [] and s['lanes']['flown_twice'] == []
    assert s['coverage_fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    assert s['separation']['ticks_below'] == 0 and s['separation']['min_m'] > small_cfg['fleet']['min_separation_m']
    names = [e['event'] for e in observe.read_events(str(tmp_path / 'topside'))]
    assert names.index('vehicle_aborted') < names.index('lanes_reassigned') < names.index('vehicle_released')
    assert 'lanes_assigned' in [e['event'] for e in observe.read_events(str(tmp_path / 'v0'))]
    assert s['link']['nodes']['v1']['sent']['abort'] >= 1
    merged = json.load(open(tmp_path / 'merged' / 'coverage.json'))
    assert merged['aborted'] is None and merged['contacts']['ticks'] == 0       # the fleet absorbed the abort; no one touched anything


def test_without_coordination_an_aborted_vehicles_share_is_left_undone(tmp_path, small_scn):
    s = fleet.run(_with_fleet(small_scn, vehicle_faults=BATTERY_DROP, coordination=False), str(tmp_path), vehicles=2, save_frames=False)
    assert s['aborted'] == [1] and not s['coordination'] and 'handovers' not in s
    assert s['lanes']['not_flown'] == [2, 3] and s['coverage_fraction'] < 0.95   # part of lane 2 was flown before the abort
    assert 'not flown' in json.load(open(tmp_path / 'merged' / 'coverage.json'))['aborted']


def test_a_silent_vehicle_keeps_its_share_and_stops_when_it_loses_the_link(tmp_path, small_scn):
    silent = {1: [{'type': 'modem_dead', 't_start': 10.0}]}
    s = fleet.run(_with_fleet(small_scn, vehicle_faults=silent), str(tmp_path), vehicles=2, save_frames=False)
    assert s['silent_vehicles'] == [1] and s['handovers'] == [] and s['lanes']['flown_twice'] == []
    assert s['coverage_fraction'] == 1.0                                       # its frames are read at recovery
    assert 'link_lost' in [e['event'] for e in observe.read_events(str(tmp_path / 'v1'))]
    assert 'vehicle_silent' in [e['event'] for e in observe.read_events(str(tmp_path / 'topside'))]


def test_vehicles_share_the_hull_in_stretches_with_fixes_from_the_ship(tmp_path, small_scn, small_cfg):
    scn = _with_fleet(small_scn, acoustic={'usbl': {'enabled': True}}, allocation='sections')
    s = fleet.run(scn, str(tmp_path), vehicles=3, save_frames=False)
    assert s['allocation'] == 'sections' and s['coverage_fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    assert sorted(s['lanes']['flown_by']) == sorted(f'{k}:{j}' for k in range(3) for j in range(4))     # every lane in every stretch
    assert s['lanes']['not_flown'] == [] and s['lanes']['flown_twice'] == []
    assert s['separation']['ticks_below'] == 0 and s['ship_fixes_sent'] > 0
    for v in s['per_vehicle']:
        x0, x1 = v['x_range_m']
        xs = [json.loads(line)['cam_pos'][0] for line in open(tmp_path / f'v{v["vehicle"]}' / 'frames.jsonl')]
        assert min(xs) >= min(x0, 0.0) - 0.6 and max(xs) <= max(x1, small_scn.hull.length) + 0.6     # frames in its stretch
        assert v['ship_fixes']['accepted'] > 0
