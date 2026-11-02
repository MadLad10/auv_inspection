"""Command line: generate scenarios, run surveys and vehicle-in-the-loop missions, render previews."""
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
