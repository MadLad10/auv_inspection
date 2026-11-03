"""Vehicle in the loop on the small hull: dynamics, navigation, faults and replanning, scored by the referee."""
import dataclasses
import json

import numpy as np
import pytest

from auvinspect import config, mission, observe
from auvinspect.dynamics import VehicleModel
from auvinspect.guidance import Controller

from conftest import SMALL

TRUTH_KEYS = {'x', 'y', 'z', 'pos', 'truth', 'coverage_fraction', 'false_claim_fraction', 'error', 'clearance'}


def _events(out, name=None):
    ev = observe.read_events(str(out))
    return [e for e in ev if name is None or e['event'] == name]


def _with_run(scn, **run):
    """The same frozen world flown with different run settings."""
    cfg = config.merge_world(config.load(overrides={**SMALL, **run}), scn.cfg)
    return dataclasses.replace(scn, cfg=cfg)


@pytest.fixture(scope='module')
def nominal(tmp_path_factory, small_scn):
    out = tmp_path_factory.mktemp('nominal')
    return mission.run(small_scn, str(out), save_frames=False), out


def test_thruster_allocation_produces_the_requested_wrench(small_cfg):
    c = Controller(small_cfg, VehicleModel(small_cfg))
    B = []
    for n in c.names:
        x, y, z, dx, dy, dz = small_cfg['vehicle']['model']['thrusters'][n]
        d = np.array([dx, dy, dz], float)
        B.append(np.concatenate([d, np.cross(np.array([x, y, z]), d)]))   # about the body origin, like the dynamics
    B = np.array(B).T[[0, 1, 2, 4, 5]]                   # Fx Fy Fz My Mz (roll is not actuated)
    assert np.allclose(B @ c.alloc, np.eye(5), atol=1e-9)


def test_nominal_mission_covers_the_band_and_claims_only_what_it_saw(nominal, small_scn):
    s, out = nominal
    assert s['aborted'] is None
    assert s['coverage_fraction'] == 1.0 and all(it['inspected'] for it in s['items'])
    assert s['claimed']['fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    assert s['clearance_min_m'] > 1.0 and s['contacts']['ticks'] == 0
    nav = s['navigation']
    # depth and the standoff sounder pin girth and normal; hull-end fixes bound the along-ship drift
    assert nav['camera_error_max_m']['girth'] < 0.1 and nav['camera_error_max_m']['normal'] < 0.1
    assert nav['camera_error_max_m']['along'] < 0.25
    assert nav['hull_end_updates']['accepted'] >= 2 * sum(l['kind'] == 'lane' for l in s['legs'])
    assert json.load(open(out / 'coverage.json'))['scenario_digest'] == small_scn.digest()
    traj = np.load(out / 'trajectory.npz')
    assert traj['data'].shape[1] == len(traj['columns'])


def test_the_event_stream_and_metrics_of_a_nominal_mission(nominal, small_cfg):
    s, out = nominal
    ev = _events(out)
    assert ev[0]['event'] == 'mission_started' and ev[-1]['event'] == 'mission_finished' and ev[-1]['outcome'] == 'completed'
    assert all(not (TRUTH_KEYS & set(e)) for e in ev)                               # only what the vehicle knows
    status = [e['t'] for e in ev if e['event'] == 'status']
    every = small_cfg['observability']['status_every_s']
    assert len(status) >= int(s['sim_time_s'] // every) and all(abs(b - a - every) < 0.2 for a, b in zip(status, status[1:]))
    assert sum(e['event'] == 'leg_finished' for e in ev) == len(s['legs'])
    assert not [e for e in ev if e['level'] in ('warning', 'critical')]
    m = {(n, tuple(sorted(lb.items()))): v for n, lb, v in observe.parse_prom((out / observe.METRICS_FILE).read_text())}
    lb = (('scenario', s['scenario_digest'][:12]), ('seed', str(s['seed'])))
    assert m[('auvinspect_referee_coverage_fraction', lb)] == s['coverage_fraction']
    assert m[('auvinspect_frames_total', tuple(sorted(dict(lb, claim='claimed').items())))] == s['claimed']['frames']['claimed']
    fired, critical = observe.health(str(out), small_cfg)
    assert not critical and not fired


def test_camera_blackout_leaves_a_hole_that_the_vehicle_finds_and_revisits(tmp_path, small_scn):
    scn = _with_run(small_scn, faults=[{'type': 'camera_blackout', 't_start': 100.0, 'duration': 20.0}])
    s = mission.run(scn, str(tmp_path), save_frames=False)
    survey = s['rounds'][0]
    assert s['claimed']['frames']['black'] >= 3
    assert survey['coverage_fraction_after'] < 0.99                                        # a real hole
    assert 0 <= survey['coverage_fraction_after'] - survey['claimed_fraction_after'] < 0.01   # the vehicle saw it, and claims no more than is true
    assert s['rounds'][1]['phase'] == 'revisit_1' and s['rounds'][1]['frames'] > 0
    assert s['coverage_fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    black, back = _events(tmp_path, 'camera_black'), _events(tmp_path, 'camera_restored')
    assert len(black) == 1 and 100.0 <= black[0]['t'] < 105.0 and 120.0 <= back[0]['t'] < 125.0
    assert _events(tmp_path, 'revisit_planned')[0]['holes'] >= 1
    fired, critical = observe.health(str(tmp_path), scn.cfg)
    assert 'dead frames' in {a['alert'] for a in fired} and not critical


def test_turbid_water_is_measured_and_not_claimed(tmp_path, small_scn):
    scn = _with_run(small_scn, faults=[{'type': 'turbidity', 't_start': 100.0, 'duration': 20.0, 'params': {'visibility_m': 1.0}}])
    s = mission.run(scn, str(tmp_path), save_frames=False)
    survey = s['rounds'][0]
    assert s['claimed']['frames']['below_limits'] >= 3 and survey['coverage_fraction_after'] < 0.99
    assert 0 <= survey['coverage_fraction_after'] - survey['claimed_fraction_after'] < 0.01
    assert s['coverage_fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    turbid, clear = _events(tmp_path, 'turbid_water'), _events(tmp_path, 'water_clearer')
    assert len(turbid) == 1 and 100.0 <= turbid[0]['t'] < 102.0 and turbid[0]['visibility_m'] < 2.0
    assert len(clear) == 1 and 120.0 <= clear[0]['t'] < 122.0


@pytest.mark.parametrize('hull_end_fixes', [True, False])
def test_dvl_dropout_is_never_overclaimed(tmp_path, small_scn, hull_end_fixes):
    scn = _with_run(small_scn, faults=[{'type': 'dvl_dropout', 't_start': 60.0, 'duration': 60.0}],
                    navigation={'hull_end_update': hull_end_fixes})
    s = mission.run(scn, str(tmp_path), save_frames=False)
    along = s['navigation']['camera_error_max_m']['along']
    assert s['claimed']['frames']['uncertain'] > 0                 # dead reckoning: frames are seen, not claimed
    assert s['claimed']['false_claim_fraction'] == 0.0
    lost, back = _events(tmp_path, 'dvl_lost'), _events(tmp_path, 'dvl_restored')
    assert len(lost) == 1 and 60.0 <= lost[0]['t'] < 61.0 and len(back) == 1 and 120.0 <= back[0]['t'] < 121.0
    if hull_end_fixes:
        assert along < 1.0
        assert s['coverage_fraction'] == 1.0 and s['claimed']['fraction'] == 1.0   # revisits after the fixes
    else:
        assert along > 1.0                                          # the drift the fixes remove
        assert s['claimed']['fraction'] < 0.9                       # and nothing lets the vehicle verify it


def test_mission_is_deterministic(tmp_path, small_scn):
    scn = _with_run(small_scn, mission={'max_seconds': 15.0})
    runs = []
    for k in range(2):
        out = tmp_path / str(k)
        s = mission.run(scn, str(out), save_frames=False)
        assert s['aborted'] == 'time limit'
        runs.append((np.load(out / 'trajectory.npz')['data'], (out / 'frames.jsonl').read_text(), (out / observe.EVENTS_FILE).read_text()))
    assert np.array_equal(runs[0][0], runs[1][0]) and runs[0][1] == runs[1][1] and runs[0][2] == runs[1][2]
    assert len(runs[0][1].splitlines()) > 0


def test_faults_the_vehicle_absorbs_cost_no_coverage_and_no_honesty(tmp_path, small_scn):
    """A stuck depth sensor (caught by the frozen-value check), a compass offset and a gyro bias (handled
    by the navigation filter), a sounder dropout (frames marked uncertain) and a dead thruster."""
    scn = _with_run(small_scn, faults=[
        {'type': 'pressure_stuck', 't_start': 20.0, 'duration': 20.0},
        {'type': 'compass_disturbance', 't_start': 40.0, 'duration': 20.0, 'params': {'offset_deg': 20.0}},
        {'type': 'imu_bias', 't_start': 60.0, 'params': {'gyro_bias_z': 0.02}},
        {'type': 'standoff_dropout', 't_start': 80.0, 'duration': 15.0},
        {'type': 'thruster_dead', 't_start': 100.0, 'params': {'name': 'sway_front'}}])
    s = mission.run(scn, str(tmp_path), save_frames=False)
    assert s['aborted'] is None and s['contacts']['ticks'] == 0
    assert s['coverage_fraction'] == 1.0 and s['claimed']['fraction'] == 1.0 and s['claimed']['false_claim_fraction'] == 0.0
    nav = s['navigation']
    events = [(h['sensor'], h['event'].split(':')[0], h['t']) for h in nav['health']]
    assert events[0][:2] == ('pressure', 'frozen') and 20.0 < events[0][2] < 21.0
    assert events[1][:2] == ('pressure', 'live again') and 40.0 <= events[1][2] < 41.0
    out, back = _events(tmp_path, 'sensor_excluded'), _events(tmp_path, 'sensor_restored')
    assert [(e['sensor'], e['level']) for e in out] == [('pressure', 'warning')] and 20.0 < out[0]['t'] < 21.0
    assert [e['sensor'] for e in back] == ['pressure'] and 40.0 <= back[0]['t'] < 41.0
    assert nav['ekf']['updates']['compass']['rejected'] > 100
    assert nav['camera_error_max_m']['girth'] < 0.05


def test_low_battery_aborts_the_mission_and_surfaces_along_the_hull(tmp_path, small_scn):
    scn = _with_run(small_scn, faults=[{'type': 'battery_drop', 't_start': 80.0, 'params': {'drop_pct': 95.0}}])
    s = mission.run(scn, str(tmp_path), save_frames=False)
    assert s['aborted'].startswith('battery reserve')
    abort = s['legs'][-1]
    assert abort['phase'] == 'abort' and abort['outcome'] == 'done' and abort['to_xs'][1] == 0.0
    traj = np.load(tmp_path / 'trajectory.npz')
    assert -traj['data'][-1, 3] < 0.5                            # at the surface
    assert s['contacts']['ticks'] == 0
    assert s['coverage_fraction'] < 0.9 and s['claimed']['false_claim_fraction'] == 0.0   # honest about the rest
    names = [e['event'] for e in _events(tmp_path)]
    assert names.index('battery_low') < names.index('abort') and names[-1] == 'mission_finished'
    assert _events(tmp_path, 'abort')[0]['level'] == 'critical'
    fired, critical = observe.health(str(tmp_path), scn.cfg)
    assert critical and 'mission abandoned' in {a['alert'] for a in fired}
