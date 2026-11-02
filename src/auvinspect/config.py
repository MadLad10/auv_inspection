"""Load the YAML configuration (package defaults, then a user file, then overrides) and validate it."""
from __future__ import annotations

import copy
import hashlib
import json
import math
from importlib.resources import files

import yaml

SCHEMA_VERSION = 2
# The world (saved with a scenario, part of its digest) versus the run (mission and system settings
# that may change between surveys of the same world).
WORLD_KEYS = ('schema_version', 'hull', 'appearance', 'defects', 'anodes', 'water')


class ConfigError(ValueError):
    pass


def default_config() -> dict:
    return yaml.safe_load(files('auvinspect').joinpath('config/default.yaml').read_text())


def deep_update(base: dict, upd: dict) -> dict:
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def load(path: str | None = None, overrides: dict | None = None) -> dict:
    cfg = default_config()
    if path:
        with open(path) as f:
            deep_update(cfg, yaml.safe_load(f) or {})
    if overrides:
        deep_update(cfg, overrides)
    validate(cfg)
    return cfg


def world(cfg: dict) -> dict:
    return {k: copy.deepcopy(cfg[k]) for k in WORLD_KEYS}


def merge_world(run_cfg: dict, world_cfg: dict) -> dict:
    """A run configuration whose world sections are replaced by a scenario's."""
    out = copy.deepcopy(run_cfg)
    out.update(world(world_cfg))
    validate(out)
    return out


def digest(cfg: dict) -> str:
    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode()).hexdigest()


def _check(cond: bool, msg: str):
    if not cond:
        raise ConfigError(msg)


def validate(cfg: dict) -> None:
    _check(cfg.get('schema_version') == SCHEMA_VERSION, f'schema_version must be {SCHEMA_VERSION}')
    h = cfg['hull']
    for k in ('length_m', 'side_depth_m', 'bilge_radius_m', 'bottom_halfwidth_m', 'texture_mm_per_px'):
        _check(h[k] > 0, f'hull.{k} must be positive')
    girth = h['side_depth_m'] + math.pi / 2 * h['bilge_radius_m'] + h['bottom_halfwidth_m']
    g0, g1 = h['inspect_girth_m']
    _check(0 <= g0 < g1 <= girth, f'hull.inspect_girth_m must lie within [0, {girth:.3f}]')
    _check(2 * h['inspect_margin_m'] < h['length_m'], 'hull.inspect_margin_m too large for the hull length')
    a = cfg['anodes']
    _check(a['size_m'][2] / 2 <= a['girth_m'] <= h['side_depth_m'] - a['size_m'][2] / 2,
           'anodes.girth_m must keep the anode on the flat side')
    _check(0 <= a['p_missing'] + a['p_depleted'] <= 1, 'anode status probabilities must sum to at most 1')
    for name in ('corrosion', 'coating_damage', 'biofouling', 'crack'):
        lo, hi = cfg['defects'][name]['count']
        _check(0 <= lo <= hi, f'defects.{name}.count must be [lo, hi] with 0 <= lo <= hi')
    cams = cfg['cameras']
    _check(cfg['survey']['camera'] in cams['rig'], 'survey.camera must name a rig camera')
    for c in cfg['survey']['record_cameras']:
        _check(c in cams['rig'], f'survey.record_cameras: unknown camera {c}')
    for name, c in cams['rig'].items():
        _check(c['look'] in ('forward', 'port', 'starboard', 'down'), f'cameras.rig.{name}.look is invalid')
    _check(0 < cams['hfov_deg'] < 170, 'cameras.hfov_deg out of range')
    s = cfg['survey']
    _check(0 <= s['overlap_along'] < 1 and 0 <= s['overlap_across'] < 1, 'survey overlaps must be in [0, 1)')
    _check(s['standoff_m'] > 0, 'survey.standoff_m must be positive')
    _check(s['band_margin_m'] >= 0, 'survey.band_margin_m must be non-negative')
    _check(cfg['coverage']['cell_m'] > 0, 'coverage.cell_m must be positive')
    _validate_mission(cfg)


def _positive(cfg: dict, section: str, keys) -> None:
    sec = cfg
    for k in section.split('.'):
        sec = sec[k]
    for k in keys:
        vals = sec[k] if isinstance(sec[k], list) else [sec[k]]
        _check(all(v > 0 for v in vals), f'{section}.{k} must be positive')


def _validate_physics(cfg: dict) -> None:
    w = cfg['water']
    _positive(cfg, 'water', ('density_kg_m3', 'gravity_m_s2', 'visibility_constant'))
    for k in ('absorption_per_m', 'scattering_per_m'):
        _check(len(w[k]) == 3 and min(w[k]) >= 0, f'water.{k} must be three non-negative values')
    _check(-1 < w['phase_g'] < 1, 'water.phase_g must be in (-1, 1)')
    _check(w['absorption_per_m'][1] + w['scattering_per_m'][1] > 0, 'water: the middle channel needs some attenuation')
    _positive(cfg, 'cameras.sensor', ('pixel_pitch_um', 'f_number', 'quantum_efficiency', 'wavelength_nm', 'full_well_e',
                                      'exposure_s', 'max_gain', 'vignetting_exponent'))
    sen = cfg['cameras']['sensor']
    _check(sen['exposure_s'][0] <= sen['exposure_s'][1] and sen['max_gain'] >= 1, 'cameras.sensor exposure or gain range is invalid')
    _check(isinstance(sen['bits'], int) and 1 <= sen['bits'] <= 16, 'cameras.sensor.bits must be an integer in [1, 16]')
    _positive(cfg, 'cameras.lamps', ('power_w', 'half_power_angle_deg'))
    _check(cfg['cameras']['lamps']['half_power_angle_deg'] < 90, 'cameras.lamps.half_power_angle_deg must be below 90')
    _positive(cfg, 'optics', ('backscatter_samples', 'path_start_m', 'path_max_m', 'backscatter_grid_px',
                              'forward_kernel_radius_px', 'forward_cone_deg'))
    cv = cfg['coverage']
    _positive(cfg, 'coverage', ('max_range_m', 'max_incidence_deg', 'max_gsd_mm', 'reflectance_step', 'min_cell_cnr', 'max_blur_px'))
    _check(cv['cnr_channel'] in (0, 1, 2), 'coverage.cnr_channel must be 0, 1 or 2')
    m = cfg['vehicle']['model']
    _positive(cfg, 'vehicle.model', ('size_m', 'mass_kg', 'max_thrust_n', 'thruster_tau_s', 'thruster_diameter_m',
                                     'thruster_efficiency', 'battery_wh'))
    _check(min(m['damping_linear']) >= 0 and len(m['damping_linear']) == 6, 'vehicle.model.damping_linear: six non-negative values')
    _check(m['drag_coefficient'] >= 0, 'vehicle.model.drag_coefficient must be non-negative')
    for name, t in m['thrusters'].items():
        _check(len(t) == 6 and any(t[3:]), f'vehicle.model.thrusters.{name}: [x, y, z, dx, dy, dz] with a direction')
    v = cfg['vehicle']
    _positive(cfg, 'vehicle', ('integration_dt_s', 'control_dt_s', 'tilt_rate_deg_s'))
    _check(v['integration_dt_s'] <= v['control_dt_s'], 'vehicle.integration_dt_s must not exceed control_dt_s')
    _positive(cfg, 'vehicle.contact', ('stiffness_n_per_m', 'slip_speed_mps'))


def _validate_mission(cfg: dict) -> None:
    from .faults import FAULT_TYPES
    _validate_physics(cfg)
    v = cfg['vehicle']
    for k in ('speed_mps', 'transit_speed_mps', 'lead_m', 'path_step_m', 'arrive_tol_m', 'leg_timeout_factor'):
        _check(v[k] > 0, f'vehicle.{k} must be positive')
    _check(v['start_lead_m'] >= 0 and v['edge_search_m'] >= 0, 'vehicle.start_lead_m / edge_search_m must be non-negative')
    c = v['control']
    for k in ('kp', 'kd', 'ki', 'i_limit_n'):
        _check(len(c[k]) == 3 and min(c[k]) >= 0, f'vehicle.control.{k} must be three non-negative values')
    _check(c['drag_feedforward'] >= 0, 'vehicle.control.drag_feedforward must be non-negative')
    n = cfg['navigation']
    _check(n['initial_fix_std']['pos_m'] > 0 and n['initial_fix_std']['yaw_deg'] > 0,
           'navigation.initial_fix_std must be positive')
    _check(min(n['jacobian_step'], n['standoff_gate'], n['hull_end_gate'], n['hull_end_std_m']) > 0,
           'navigation.jacobian_step, standoff_gate, hull_end_gate and hull_end_std_m must be positive')
    for name, sen in cfg['sensors'].items():
        _check(sen['rate_hz'] > 0, f'sensors.{name}.rate_hz must be positive')
    o = cfg['onboard']
    _check(isinstance(o['claim_pixel_stride'], int) and o['claim_pixel_stride'] >= 1, 'onboard.claim_pixel_stride must be an integer >= 1')
    _check(isinstance(o['replan_rounds'], int) and o['replan_rounds'] >= 0, 'onboard.replan_rounds must be an integer >= 0')
    _check(len(o['claim_max_std_m']) == 2 and min(o['claim_max_std_m']) > 0, 'onboard.claim_max_std_m must be two positive values')
    _check(o['dead_frame_std_dn'] >= 0 and o['claim_sigma_k'] >= 0, 'onboard.dead_frame_std_dn / claim_sigma_k must be non-negative')
    _check(0 <= o['battery_reserve_pct'] <= 100, 'onboard.battery_reserve_pct must be in [0, 100]')
    _check(isinstance(o['frozen_samples'], int) and o['frozen_samples'] >= 2, 'onboard.frozen_samples must be an integer >= 2')
    _check(cfg['mission']['max_seconds'] > 0, 'mission.max_seconds must be positive')
    p = cfg['perception']
    _check(set(p['classes']) <= {'corrosion', 'coating_damage', 'biofouling', 'crack', 'missing_anode', 'anode'} and p['classes'],
           'perception.classes must be scenario item classes')
    _check(0 < p['iou_match'] <= 1 and 0 <= p['min_inspectable_share'] <= 1 and 0 <= p['score_threshold'] <= 1,
           'perception.iou_match, min_inspectable_share and score_threshold must be shares')
    _check(len(p['splits']) == 3 and min(p['splits']) >= 0 and abs(sum(p['splits']) - 1) < 1e-9, 'perception.splits must be three shares summing to 1')
    _check(list(p['size_bins_m2']) == sorted(p['size_bins_m2']), 'perception.size_bins_m2 must be increasing')
    _check(cfg['baseline']['threshold_mad'] > 0 and cfg['baseline']['fit_step_px'] >= 1, 'baseline.threshold_mad / fit_step_px out of range')
    _check(isinstance(cfg['faults'], list), 'faults must be a list')
    for f in cfg['faults']:
        _check(isinstance(f, dict) and f.get('type') in FAULT_TYPES, f'faults: unknown fault {f!r}; known: {sorted(FAULT_TYPES)}')
        _check(set(f) <= {'type', 't_start', 'duration', 'params'}, f'faults: unexpected keys in {f!r}')
        _check(f.get('t_start', 0.0) >= 0 and (f.get('duration') is None or f['duration'] > 0),
               f'faults: t_start must be >= 0 and duration positive or null in {f!r}')
