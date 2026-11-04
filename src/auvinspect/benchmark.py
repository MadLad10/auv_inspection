"""Benchmark: suites of hulls, each flown as a full mission and scored by the referees.

A suite (benchmark.suites) is a list of scenario seeds with optional world overrides. Each hull's
water and paint are drawn from its seed (benchmark.world, stream 'benchmark_world'): the visibility,
reached by scaling the water's particle scattering, and one of the antifouling paints. The coverage
criterion judges a defect against the hull's own paint. The dev suite is for tuning; the test suite
is frozen: scored to report a milestone, never tuned on.

Per hull: generate the scenario, fly the mission, run the detector (the classical baseline, leaving
the stations' glare zones out), build the map and the report (with the packaged station model), and
keep the referees' verdicts: coverage and claimed coverage, the share of claimed cells the referee
does not count covered, items found by class for confirmed and for all findings, false findings and
duplicates, station statuses.

Score: each value is pooled over the suite's hulls (items found over items required, ...) or a mean
over hulls (coverage), with an interval from resampling the hulls with replacement
(benchmark.bootstrap draws, fixed stream): the hull, not the frame or the item, is the unit of
independence. With a handful of hulls the intervals are wide, and they say so; a suite of fewer than
benchmark.min_hulls_for_interval hulls gets none, since resampling two or three hulls understates
the spread.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os

import numpy as np

from . import __version__, baseline, config as config_mod, detector, observe, optics, report, stations
from .mapping import map_classes
from .rng import stream
from .scenario import generate


def world(seed: int, cfg: dict, suite: dict) -> tuple[dict, dict]:
    """The configuration for one benchmark hull (suite overrides, then its drawn water and paint)
    and what was drawn."""
    out = copy.deepcopy(cfg)
    config_mod.deep_update(out, suite.get('overrides', {}))
    w = cfg['benchmark']['world']
    rng = stream(seed, 'benchmark_world')
    vis = float(rng.uniform(*w['visibility_m']))
    paint = list(w['paint_rgb'][int(rng.integers(len(w['paint_rgb'])))])
    out['water'] = optics.with_visibility(out['water'], vis)
    out['appearance']['paint_rgb'] = paint
    config_mod.validate(out)
    return out, {'visibility_m': round(vis, 2), 'paint_rgb': paint}


def _settings_digest(cfg: dict) -> str:
    run = {k: v for k, v in cfg.items() if k not in config_mod.WORLD_KEYS}
    return hashlib.sha256(json.dumps(run, sort_keys=True).encode()).hexdigest()


def run_hull(seed: int, suite_name: str, out_root: str, cfg: dict, model: dict | None = None, rescore: bool = False,
             library=None, seg: dict | None = None, missions: str | None = None) -> dict:
    """One hull of a suite, end to end. Its result is kept in <out_root>/s<seed>/result.json and reused
    when present; with rescore, the flown mission is kept and detection, map, stations and report are
    redone (after a change to them). With a patch library (patches.Library), the hull carries real defects.
    With seg (segmenter.load_for_benchmark), the deep segmenter replaces the classical detector. With
    missions (another run of the same suite), its flown missions are reused, so detectors are compared
    on the same flights."""
    from .mission import run as fly
    d = os.path.join(out_root, f's{seed}')
    path = os.path.join(d, 'result.json')
    if os.path.exists(path) and not rescore:
        with open(path) as f:
            return json.load(f)
    suite = cfg['benchmark']['suites'][suite_name]
    wcfg, drawn = world(seed, cfg, suite)
    src = os.path.join(missions, f's{seed}') if missions else d
    as_fleet = bool(suite.get('fleet'))
    mdir = os.path.join(src, 'fleet', 'merged') if as_fleet else os.path.join(src, 'mission')
    if os.path.exists(os.path.join(mdir, 'coverage.json')):
        with open(os.path.join(src, 'scenario.json')) as f:
            digest = json.load(f)['digest']
        if digest != generate(seed, wcfg, library).digest():
            raise ValueError(f'{src}: the saved scenario is not this hull of the suite; remove it to fly again')
    elif missions:
        raise ValueError(f'{src}: no flown mission to reuse')
    os.makedirs(d, exist_ok=True)
    trace = observe.Trace(d, suite=suite_name, seed=seed)
    if not os.path.exists(os.path.join(mdir, 'coverage.json')):
        with trace.span('generate'):
            scn = generate(seed, wcfg, library)
            digest = scn.save(d)
        with trace.span('fly', vehicles=wcfg['fleet']['vehicles'] if as_fleet else 1):
            if as_fleet:
                from . import fleet
                fleet.run(scn, os.path.join(d, 'fleet'))
            else:
                fly(scn, mdir)
    with trace.span('detect', detector='segmenter' if seg is not None else 'classical'):
        if seg is not None:
            from . import segmenter
            det_model = None
            det = segmenter.run_frames(seg['net'], mdir, {**wcfg, 'segmenter': seg['cfg']['segmenter']})
        else:
            det_model = detector.load_model(wcfg)
            det = baseline.run_frames(mdir, wcfg, stations.window_masks(mdir, wcfg), det_model)
    with trace.span('report'):
        model = model or stations.load_model(wcfg)
        report.write(mdir, det, os.path.join(d, 'report'), wcfg, model)
    observe.write_metrics(os.path.join(d, observe.METRICS_FILE), trace.samples())
    with open(os.path.join(mdir, 'coverage.json')) as f:
        cov = json.load(f)
    with open(os.path.join(d, 'report', 'metrics.json')) as f:
        m = json.load(f)

    def tier(t):
        v = m['map'][t]
        return {'required': v['items_required'], 'found': v['items_found'], 'findings': v['findings'],
                'false': v['false_findings'], 'duplicates': sum(c['duplicates'] for c in v['per_class'].values()),
                'per_class': {c: {'required': x['required'], 'found': x['found'], 'false': x['false']} for c, x in v['per_class'].items()}}
    st = m['stations']
    result = {'seed': seed, 'suite': suite_name, 'world': drawn, 'scenario_digest': digest,
              **({'patch_library_sha256': library.digest} if library else {}),
              'coverage': cov['coverage_fraction'], 'claimed': cov['claimed']['fraction'],
              'false_claim': cov['claimed']['false_claim_fraction'], 'aborted': cov['aborted'],
              'contact_ticks': cov['contacts']['ticks'], 'sim_time_s': cov['sim_time_s'],
              'map': {'confirmed': tier('confirmed'), 'all': tier('all')},
              'stations': {k: st[k] for k in ('stations', 'inspected', 'correct', 'unclear', 'wrong', 'per_status')},
              **({'fleet': _fleet_result(os.path.dirname(mdir))} if as_fleet else {}),
              'versions': {'auvinspect': __version__, 'run_settings_sha256': _settings_digest(cfg),
                           'station_model_sha256': hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest(),
                           'detector_model_sha256': (f'segmenter:{seg["sha256"]}' if seg is not None else
                                                     hashlib.sha256(json.dumps(det_model, sort_keys=True).encode()).hexdigest()
                                                     if det_model else 'rules')}}
    with open(path + '.tmp', 'w') as f:
        json.dump(result, f, indent=1)
    os.replace(path + '.tmp', path)
    return result


# ------------------------------------------------------------------------------------- scoring
def _ratio(num, den):
    def f(rs):
        d = sum(den(r) for r in rs)
        return sum(num(r) for r in rs) / d if d else float('nan')
    return f


def _mean(val):
    return lambda rs: float(np.mean([val(r) for r in rs])) if rs else float('nan')


def metric_functions(cfg: dict) -> dict:
    """Each benchmark value as a function of a list of hull results."""
    fns = {
        'coverage': _mean(lambda r: r['coverage']),
        'false_claim': _mean(lambda r: r['false_claim']),
        'map_recall_confirmed': _ratio(lambda r: r['map']['confirmed']['found'], lambda r: r['map']['confirmed']['required']),
        'map_recall_all': _ratio(lambda r: r['map']['all']['found'], lambda r: r['map']['all']['required']),
        'map_precision_confirmed': _ratio(lambda r: r['map']['confirmed']['findings'] - r['map']['confirmed']['false'] - r['map']['confirmed']['duplicates'],
                                          lambda r: r['map']['confirmed']['findings']),
        'false_confirmed_per_hull': _mean(lambda r: r['map']['confirmed']['false']),
        'station_accuracy': _ratio(lambda r: r['stations']['correct'], lambda r: r['stations']['inspected']),
        'station_wrong': _ratio(lambda r: r['stations']['wrong'], lambda r: r['stations']['inspected']),
        'station_unclear': _ratio(lambda r: r['stations']['unclear'], lambda r: r['stations']['inspected']),
        'stations_inspected': _ratio(lambda r: r['stations']['inspected'], lambda r: r['stations']['stations']),
    }
    for s in stations.STATUSES:
        fns[f'station_{s}_found'] = _ratio(lambda r, s=s: r['stations']['per_status'][s]['found'],
                                           lambda r, s=s: r['stations']['per_status'][s]['true'])
    for c in map_classes(cfg):
        fns[f'recall_all_{c}'] = _ratio(lambda r, c=c: r['map']['all']['per_class'][c]['found'],
                                        lambda r, c=c: r['map']['all']['per_class'][c]['required'])
    return fns


def score(results: list[dict], cfg: dict, name: str = '') -> dict:
    """Pooled values over hulls with hull-resampling intervals (see module notes)."""
    b = cfg['benchmark']
    rng = stream(0, f'benchmark_bootstrap:{name}')
    n = len(results)
    draws = rng.integers(0, n, (b['bootstrap'], n)) if n >= b['min_hulls_for_interval'] else np.zeros((0, n), int)
    q = (1 - b['interval']) / 2
    out = {}
    fns = metric_functions(cfg)
    if results and all('fleet' in r for r in results):
        fns.update(FLEET_METRICS)
    for key, fn in fns.items():
        v = fn(results)
        boot = np.array([fn([results[i] for i in idx]) for idx in draws])
        boot = boot[np.isfinite(boot)]
        out[key] = {'value': None if not np.isfinite(v) else round(float(v), 4),
                    'low': round(float(np.quantile(boot, q)), 4) if boot.size else None,
                    'high': round(float(np.quantile(boot, 1 - q)), 4) if boot.size else None}
    return out


def _fleet_result(fleet_dir: str) -> dict:
    with open(os.path.join(fleet_dir, 'fleet.json')) as f:
        s = json.load(f)
    return {'vehicles': s['vehicles'], 'makespan_s': s['makespan_s'], 'end_s': s['end_s'], 'aborted_vehicles': s['aborted'],
            'handovers': len(s.get('handovers', [])), 'lanes_not_flown': len(s['lanes']['not_flown']),
            'lanes_flown_twice': len(s['lanes']['flown_twice']), 'separation_min_m': s['separation']['min_m'],
            'separation_ticks_below': s['separation']['ticks_below'],
            'link_bytes': sum(n['bytes'] for n in s['link']['nodes'].values()) if s.get('link') else 0}


FLEET_METRICS = {
    'fleet_min_separation_m': lambda rs: min(r['fleet']['separation_min_m'] for r in rs),
    'fleet_lanes_not_flown': lambda rs: float(sum(r['fleet']['lanes_not_flown'] for r in rs)),
    'fleet_lanes_flown_twice': lambda rs: float(sum(r['fleet']['lanes_flown_twice'] for r in rs)),
    'fleet_handovers': lambda rs: float(sum(r['fleet']['handovers'] for r in rs)),
    'fleet_makespan_s': lambda rs: float(np.mean([r['fleet']['makespan_s'] for r in rs])),
}


def summarize(out_root: str, suite_name: str, cfg: dict) -> dict:
    """The suite's score from the hull results present, listing the hulls not yet run."""
    seeds = cfg['benchmark']['suites'][suite_name]['seeds']
    results, missing = [], []
    for seed in seeds:
        path = os.path.join(out_root, f's{seed}', 'result.json')
        if os.path.exists(path):
            with open(path) as f:
                results.append(json.load(f))
        else:
            missing.append(seed)
    summary = {'suite': suite_name, 'seeds': seeds, 'hulls': len(results), 'missing': missing,
               'interval': cfg['benchmark']['interval'], 'values': score(results, cfg, suite_name),
               'per_hull': [{'seed': r['seed'], **r['world'], 'coverage': r['coverage'], 'false_claim': r['false_claim'],
                             'aborted': r['aborted'], 'recall_confirmed': f"{r['map']['confirmed']['found']}/{r['map']['confirmed']['required']}",
                             'false_confirmed': r['map']['confirmed']['false'],
                             'stations': f"{r['stations']['correct']}/{r['stations']['inspected']}/{r['stations']['stations']}"}
                            for r in results],
               'versions': sorted({json.dumps(r['versions'], sort_keys=True) for r in results}),
               'patch_libraries': sorted({r.get('patch_library_sha256') or 'procedural' for r in results})}
    summary['versions'] = [json.loads(v) for v in summary['versions']]
    return summary


def gate(summary: dict, cfg: dict) -> list[str]:
    """Reasons the suite fails its gate (empty when it passes): a hull missing or aborted, or a value
    under its min or over its max (the suite's `gate` in benchmark.suites)."""
    g = cfg['benchmark']['suites'][summary['suite']].get('gate', {})
    fails = [f'hull s{s} has no result' for s in summary['missing']]
    fails += [f"hull s{h['seed']} aborted: {h['aborted']}" for h in summary['per_hull'] if h['aborted']]
    for key, lo in (g.get('min') or {}).items():
        v = summary['values'][key]['value']
        if v is None or v < lo:
            fails.append(f'{key} = {v} below {lo}')
    for key, hi in (g.get('max') or {}).items():
        v = summary['values'][key]['value']
        if v is None or v > hi:
            fails.append(f'{key} = {v} above {hi}')
    return fails
