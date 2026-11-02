"""Command line: generate scenarios, run surveys and vehicle-in-the-loop missions, build perception
datasets, run and score detectors, render previews."""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys

from . import config as config_mod
from .scenario import Scenario, generate


def _generate(a):
    cfg = config_mod.load(a.config)
    scn = generate(a.seed, cfg)
    digest = scn.save(a.out)
    counts: dict[str, int] = {}
    for it in scn.items:
        k = it['class'] if it['class'] != 'anode' else f'anode_{it["status"]}'
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({'out': a.out, 'digest': digest, 'items': counts}, indent=1))


def _survey(a):
    from .survey import run
    scn = Scenario.load(a.scenario, config_mod.load(a.config))
    s = run(scn, a.out, save_frames=not a.no_frames)
    print(json.dumps({k: s[k] for k in ('coverage_fraction', 'poses', 'frames', 'by_class', 'elapsed_s')}, indent=1))


def _mission(a):
    from .mission import run
    scn = Scenario.load(a.scenario, config_mod.load(a.config))
    s = run(scn, a.out, save_frames=not a.no_frames, seed=a.seed)
    keys = ('coverage_fraction', 'claimed', 'by_class', 'rounds', 'navigation', 'clearance_min_m', 'aborted', 'sim_time_s', 'elapsed_s')
    out = {k: s[k] for k in keys}
    out['navigation'] = {k: v for k, v in s['navigation'].items() if k != 'ekf'}
    print(json.dumps(out, indent=1))


def _dataset(a):
    from .dataset import export
    print(json.dumps(export(a.run, a.out, config_mod.load(a.config)), indent=1))


def _detect(a):
    from .baseline import run
    det = run(a.dataset, config_mod.load(a.config), a.split)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(det, f)
    print(json.dumps({'frames': len(det), 'detections': sum(len(v) for v in det.values()), 'out': a.out}))


def _evaluate(a):
    from .perception import evaluate
    with open(os.path.join(a.dataset, 'annotations.json')) as f:
        ann = json.load(f)
    with open(a.detections) as f:
        det = json.load(f)
    m = evaluate(ann, det, config_mod.load(a.config), a.split)
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(m, f, indent=1)
    print(json.dumps(m, indent=1))


def _report(a):
    from .report import write
    cfg = config_mod.load(a.config)
    if a.detections:
        with open(a.detections) as f:
            det = json.load(f)
    else:
        from .baseline import run_frames
        det = run_frames(a.run, cfg)
    print(json.dumps(write(a.run, det, a.out, cfg), indent=1))


def _evolve(a):
    from .evolution import changes, evolve
    cfg = config_mod.load(a.config)
    base = Scenario.load(a.scenario, cfg)
    later = evolve(base, a.months, cfg)
    digest = later.save(a.out)
    counts: dict[str, int] = {}
    for v in changes(base, later, cfg['change']['growth_ratio']).values():
        counts[v] = counts.get(v, 0) + 1
    print(json.dumps({'out': a.out, 'digest': digest, 'months': a.months, 'true_changes': counts}, indent=1))


def _change(a):
    from .change import run
    changes, metrics = run(a.before, a.after, config_mod.load(a.config))
    os.makedirs(a.out, exist_ok=True)
    with open(os.path.join(a.out, 'changes.json'), 'w') as f:
        json.dump(changes, f, indent=1)
    with open(os.path.join(a.out, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=1)
    counts: dict[str, int] = {}
    for c in changes:
        counts[c['change']] = counts.get(c['change'], 0) + 1
    short = {t: {k: v for k, v in m.items() if k != 'confusion'} for t, m in metrics.items()}
    print(json.dumps({'changes': counts, 'referee': short, 'out': a.out}, indent=1))


def _preview(a):
    from .preview import rig_sheet, texture_sheet
    scn = Scenario.load(a.scenario, config_mod.load(a.config))
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    if a.texture:
        texture_sheet(scn).save(a.out)
    else:
        rig_sheet(scn, a.x, a.girth, a.standoff if a.standoff else scn.cfg['survey']['standoff_m']).save(a.out)
    print(a.out)


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
    p = argparse.ArgumentParser(prog='auvinspect', description=__doc__)
    sub = p.add_subparsers(dest='cmd', required=True)
    g = sub.add_parser('generate', help='generate a seeded scenario')
    g.add_argument('--seed', type=int, required=True)
    g.add_argument('--out', required=True)
    g.add_argument('--config', help='YAML file merged over the defaults')
    g.set_defaults(fn=_generate)
    s = sub.add_parser('survey', help='run the planned survey over a scenario')
    s.add_argument('--scenario', required=True)
    s.add_argument('--out', required=True)
    s.add_argument('--no-frames', action='store_true', help='score coverage without saving images')
    s.add_argument('--config', help='YAML run settings (cameras, survey, coverage) merged over the defaults')
    s.set_defaults(fn=_survey)
    m = sub.add_parser('mission', help='fly the survey with the vehicle in the loop (dynamics, navigation, faults, replanning)')
    m.add_argument('--scenario', required=True)
    m.add_argument('--out', required=True)
    m.add_argument('--no-frames', action='store_true', help='score coverage without saving images')
    m.add_argument('--seed', type=int, help='mission seed (sensor noise, initial fix); default: the scenario seed')
    m.add_argument('--config', help='YAML run settings (vehicle, navigation, faults, ...) merged over the defaults')
    m.set_defaults(fn=_mission)
    d = sub.add_parser('dataset', help='export a labelled perception dataset from recorded runs')
    d.add_argument('--run', action='append', required=True, help='a survey or mission directory inside its scenario directory (repeatable)')
    d.add_argument('--out', required=True)
    d.add_argument('--config', help='YAML run settings merged over the defaults')
    d.set_defaults(fn=_dataset)
    t = sub.add_parser('detect', help='run the classical baseline detector over a dataset')
    t.add_argument('--dataset', required=True)
    t.add_argument('--out', required=True, help='detections JSON')
    t.add_argument('--split', choices=['train', 'val', 'test'])
    t.add_argument('--config', help='YAML run settings merged over the defaults')
    t.set_defaults(fn=_detect)
    e = sub.add_parser('evaluate', help='score detections against the ground truth (the perception referee)')
    e.add_argument('--dataset', required=True)
    e.add_argument('--detections', required=True)
    e.add_argument('--split', choices=['train', 'val', 'test'])
    e.add_argument('--out', help='metrics JSON')
    e.add_argument('--config', help='YAML run settings merged over the defaults')
    e.set_defaults(fn=_evaluate)
    r = sub.add_parser('report', help='hull defect map and inspection report for a recorded run')
    r.add_argument('--run', required=True, help='a survey or mission directory inside its scenario directory')
    r.add_argument('--out', required=True)
    r.add_argument('--detections', help='detections JSON keyed by frame name (default: the classical baseline)')
    r.add_argument('--config', help='YAML run settings merged over the defaults')
    r.set_defaults(fn=_report)
    ev = sub.add_parser('evolve', help='the same hull some months later: grown, new and aged defects')
    ev.add_argument('--scenario', required=True)
    ev.add_argument('--months', type=float, required=True)
    ev.add_argument('--out', required=True)
    ev.add_argument('--config', help='YAML settings merged over the defaults (evolution rates)')
    ev.set_defaults(fn=_evolve)
    ch = sub.add_parser('change', help='compare the reports of two surveys of one hull')
    ch.add_argument('--before', required=True, help='report directory of the earlier survey')
    ch.add_argument('--after', required=True, help='report directory of the later survey')
    ch.add_argument('--out', required=True)
    ch.add_argument('--config', help='YAML run settings merged over the defaults')
    ch.set_defaults(fn=_change)
    v = sub.add_parser('preview', help='render the rig at one pose, or the annotated texture')
    v.add_argument('--scenario', required=True)
    v.add_argument('--out', required=True)
    v.add_argument('--x', type=float, default=10.0)
    v.add_argument('--girth', type=float, default=2.0)
    v.add_argument('--standoff', type=float)
    v.add_argument('--texture', action='store_true')
    v.add_argument('--config', help='YAML run settings merged over the defaults')
    v.set_defaults(fn=_preview)
    a = p.parse_args(argv)
    try:
        a.fn(a)
    except (config_mod.ConfigError, FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
