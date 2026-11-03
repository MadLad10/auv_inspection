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


def _library(path):
    if not path:
        return None
    from .patches import Library
    return Library(path)


def _generate(a):
    cfg = config_mod.load(a.config)
    drawn = None
    if a.vary_world:
        from .benchmark import world
        cfg, drawn = world(a.seed, cfg, {})
    scn = generate(a.seed, cfg, _library(a.patches))
    digest = scn.save(a.out)
    counts: dict[str, int] = {}
    for it in scn.items:
        k = it['class'] if it['class'] != 'anode' else f'anode_{it["status"]}'
        counts[k] = counts.get(k, 0) + 1
    print(json.dumps({'out': a.out, 'digest': digest, 'items': counts, **({'world': drawn} if drawn else {})}, indent=1))


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


def _import_liaci(a):
    from .liaci import import_dataset
    print(json.dumps(import_dataset(a.root, a.out, config_mod.load(a.config)), indent=1))


def _liaci_patches(a):
    from .patches import build
    print(json.dumps(build(a.root, a.out, config_mod.load(a.config)), indent=1))


def _train_segmenter(a):
    from . import segmenter
    model = segmenter.train(a.dataset, a.backbone, config_mod.load(a.config), a.seed, log=lambda line: print(line, flush=True))
    segmenter.save(model, a.out)
    print(json.dumps({'out': a.out, 'epoch': model['meta']['epoch'], 'val_map': model['meta']['val_map']}, indent=1))


def _segment(a):
    from . import segmenter
    if bool(a.dataset) == bool(a.run) or (a.dataset and not a.split):
        raise ValueError('segment: give --dataset with --split, or --run')
    net, meta, cfg = segmenter.load(a.model, config_mod.load(a.config))
    if a.run:
        det, extra = segmenter.run_frames(net, a.run, cfg), {}
    else:
        det, conf = segmenter.detect(net, a.dataset, a.split, cfg)
        extra = {'pixel_iou': segmenter.pixel_iou(conf, meta['classes'])}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(det, f)
    print(json.dumps({'frames': len(det), 'detections': sum(len(v) for v in det.values()), 'out': a.out, **extra}, indent=1))


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


def _training_runs(runs: list[str], cfg: dict):
    """Refuse runs whose hull is not in the training split."""
    from .dataset import split_of
    for run in runs:
        with open(os.path.join(os.path.dirname(os.path.normpath(run)), 'scenario.json')) as f:
            seed = json.load(f)['seed']
        if split_of(seed, cfg['perception']['splits']) != 'train':
            raise ValueError(f'{run}: scenario seed {seed} is not in the training split')


def _train_detector(a):
    from .detector import train
    cfg = config_mod.load(a.config)
    _training_runs(a.run, cfg)
    model = train(a.run, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(model, f)
    print(json.dumps({'out': a.out, 'regions': model['regions'], 'hulls': len(model['trained_on'])}, indent=1))


def _train_stations(a):
    from .stations import train
    cfg = config_mod.load(a.config)
    _training_runs(a.run, cfg)
    model = train(a.run, cfg)
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, 'w') as f:
        json.dump(model, f, indent=1)
    print(json.dumps({'out': a.out, 'views_by_paint': {k: v['views'] for k, v in model['paints'].items()},
                      'hulls': len(model['trained_on'])}, indent=1))


def _stations(a):
    from .stations import check, load_model
    cfg = config_mod.load(a.config)
    out = {}
    for run in a.run:
        results, metrics = check(run, cfg, load_model(cfg, a.model))
        out[run] = {'stations': {r['id']: r['status'] for r in results}, 'referee': {k: v for k, v in metrics.items() if k != 'confusion'}}
    if a.out:
        with open(a.out, 'w') as f:
            json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


def _benchmark(a):
    from . import benchmark
    cfg = config_mod.load(a.config)
    if a.suite not in cfg['benchmark']['suites']:
        raise ValueError(f'no benchmark suite {a.suite!r}; suites: {sorted(cfg["benchmark"]["suites"])}')
    if a.action == 'run':
        library = _library(a.patches)
        seg = None
        if a.segmenter:
            from .segmenter import load_for_benchmark
            seg = load_for_benchmark(a.segmenter, cfg)
        seeds = cfg['benchmark']['suites'][a.suite]['seeds']
        stray = [s for s in a.seed or [] if s not in seeds]
        if stray:
            raise ValueError(f'seeds {stray} are not in suite {a.suite!r} ({seeds})')
        for seed in a.seed or seeds:
            r = benchmark.run_hull(seed, a.suite, a.out, cfg, rescore=a.rescore, library=library, seg=seg, missions=a.missions)
            print(json.dumps({'seed': seed, 'world': r['world'], 'coverage': r['coverage'],
                              'recall_confirmed': f"{r['map']['confirmed']['found']}/{r['map']['confirmed']['required']}",
                              'stations': f"{r['stations']['correct']}/{r['stations']['inspected']}"}))
        return 0
    summary = benchmark.summarize(a.out, a.suite, cfg)
    with open(os.path.join(a.out, 'benchmark.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    if a.action == 'score':
        print(json.dumps({k: summary[k] for k in ('suite', 'hulls', 'missing', 'values')}, indent=1))
        return 0
    fails = benchmark.gate(summary, cfg)
    print(json.dumps({'suite': a.suite, 'passed': not fails, 'failures': fails}, indent=1))
    return 1 if fails else 0


def _report(a):
    from .report import write
    from .stations import load_model, window_masks
    cfg = config_mod.load(a.config)
    model = load_model(cfg, a.stations_model)
    if a.detections:
        with open(a.detections) as f:
            det = json.load(f)
    else:
        from .baseline import run_frames
        det = run_frames(a.run, cfg, window_masks(a.run, cfg))
    print(json.dumps(write(a.run, det, a.out, cfg, model), indent=1))


def _evolve(a):
    from .evolution import changes, evolve
    cfg = config_mod.load(a.config)
    base = Scenario.load(a.scenario, cfg)
    later = evolve(base, a.months, cfg, _library(a.patches))
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
    for c in changes['findings'] + changes['stations']:
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
    g.add_argument('--vary-world', action='store_true', help="draw the water and paint from the seed, as the benchmark's hulls do")
    g.add_argument('--patches', help='a patch library (liaci-patches): corrosion, coating damage and biofouling from real images')
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
    q = sub.add_parser('import-liaci', help='import the LIACi images of real ship inspections as a perception dataset')
    q.add_argument('--root', required=True, help='the unpacked LIACi_dataset_pretty directory')
    q.add_argument('--out', required=True)
    q.add_argument('--config', help='YAML settings merged over the defaults')
    q.set_defaults(fn=_import_liaci)
    lp = sub.add_parser('liaci-patches', help='cut real defects out of the LIACi training images, for the simulator')
    lp.add_argument('--root', required=True, help='the unpacked LIACi_dataset_pretty directory')
    lp.add_argument('--out', required=True)
    lp.add_argument('--config', help='YAML settings merged over the defaults')
    lp.set_defaults(fn=_liaci_patches)
    tg = sub.add_parser('train-segmenter', help='fit the deep segmenter on the train splits of perception datasets')
    tg.add_argument('--dataset', action='append', required=True, help='a dataset directory (dataset or import-liaci; repeatable)')
    tg.add_argument('--backbone', required=True, help="torchvision's MobileNetV2 ImageNet weights (mobilenet_v2-7ebf99e0.pth)")
    tg.add_argument('--out', required=True, help='model file (.pt)')
    tg.add_argument('--seed', type=int, default=0)
    tg.add_argument('--config', help='YAML settings merged over the defaults')
    tg.set_defaults(fn=_train_segmenter)
    sg = sub.add_parser('segment', help='run the deep segmenter over a dataset split (boxes for evaluate, pixel IoU) or a recorded run (for report)')
    sg.add_argument('--model', required=True)
    sg.add_argument('--dataset')
    sg.add_argument('--split', choices=['train', 'val', 'test'])
    sg.add_argument('--run', help='a survey or mission directory: detections keyed by frame name')
    sg.add_argument('--out', required=True, help='detections JSON')
    sg.add_argument('--config', help='YAML settings merged over the defaults')
    sg.set_defaults(fn=_segment)
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
    r.add_argument('--stations-model', help='anode station model JSON (default: the packaged model)')
    r.add_argument('--config', help='YAML run settings merged over the defaults')
    r.set_defaults(fn=_report)
    td = sub.add_parser('train-detector', help='fit the region classifier of the detector on runs of training hulls')
    td.add_argument('--run', action='append', required=True, help='a survey or mission directory of a training-split hull (repeatable)')
    td.add_argument('--out', required=True, help='model JSON')
    td.add_argument('--config', help='YAML run settings merged over the defaults')
    td.set_defaults(fn=_train_detector)
    ts = sub.add_parser('train-stations', help='fit the anode station classifier on runs of training hulls')
    ts.add_argument('--run', action='append', required=True, help='a survey or mission directory of a training-split hull (repeatable)')
    ts.add_argument('--out', required=True, help='model JSON')
    ts.add_argument('--config', help='YAML run settings merged over the defaults')
    ts.set_defaults(fn=_train_stations)
    sc = sub.add_parser('stations', help='check the anode stations of recorded runs and score them')
    sc.add_argument('--run', action='append', required=True, help='a survey or mission directory (repeatable)')
    sc.add_argument('--model', help='station model JSON (default: the packaged model)')
    sc.add_argument('--out', help='results JSON')
    sc.add_argument('--config', help='YAML run settings merged over the defaults')
    sc.set_defaults(fn=_stations)
    ev = sub.add_parser('evolve', help='the same hull some months later: grown, new and aged defects')
    ev.add_argument('--scenario', required=True)
    ev.add_argument('--months', type=float, required=True)
    ev.add_argument('--out', required=True)
    ev.add_argument('--config', help='YAML settings merged over the defaults (evolution rates)')
    ev.add_argument('--patches', help='the patch library a hull with real defects was painted from')
    ev.set_defaults(fn=_evolve)
    ch = sub.add_parser('change', help='compare the reports of two surveys of one hull')
    ch.add_argument('--before', required=True, help='report directory of the earlier survey')
    ch.add_argument('--after', required=True, help='report directory of the later survey')
    ch.add_argument('--out', required=True)
    ch.add_argument('--config', help='YAML run settings merged over the defaults')
    ch.set_defaults(fn=_change)
    bm = sub.add_parser('benchmark', help='fly and score a suite of hulls (run), summarise it (score), or check it against its gate (gate)')
    bm.add_argument('action', choices=['run', 'score', 'gate'])
    bm.add_argument('--suite', required=True, help='a suite in benchmark.suites (ci, dev, test)')
    bm.add_argument('--out', required=True, help='directory of the suite\'s hulls and results')
    bm.add_argument('--seed', type=int, action='append', help='run only these hulls of the suite (repeatable)')
    bm.add_argument('--rescore', action='store_true', help='keep the flown missions, redo detection, map, stations and report')
    bm.add_argument('--patches', help='paint the hulls with real defects from this patch library (liaci-patches)')
    bm.add_argument('--segmenter', help='detect with this deep segmenter model instead of the classical detector')
    bm.add_argument('--missions', help="reuse the flown missions of another run of the same suite (same hulls)")
    bm.add_argument('--config', help='YAML run settings merged over the defaults')
    bm.set_defaults(fn=_benchmark)
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
        code = a.fn(a)
    except (config_mod.ConfigError, FileNotFoundError, ValueError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return code or 0


if __name__ == '__main__':
    sys.exit(main())
