"""Model registry: versioned models with their provenance, license, scores and stage.

A registry is a directory (registry.root, or --registry). Each model name holds numbered versions:
`<root>/<name>/v<k>/` keeps the model file, read-only once added, and its card (card.json):
- kind: region_classifier or anode_stations (JSON) or segmenter (.pt), from the file;
- the file's SHA-256 and size, and the identity the benchmark records for the model in a hull's
  versions (detector_model_sha256 or station_model_sha256);
- provenance from the model's own record: what it was trained on, its settings digest; for a
  segmenter, its encoder weights' digest and the epoch kept;
- license. A model trained on LIACi's images, or on hulls painted with LIACi's defects, is an
  adaptation of LIACi (CC BY-NC-SA 4.0) and may not be redistributed. A model whose training
  sources are no longer on disk is `unknown` and may not be either. A segmenter's encoder holds
  torchvision's ImageNet weights, which this project does not redistribute: segmenters never are.
- scores: benchmark summaries (`benchmark score`) attached to the version, each accepted only
  when every hull in it was scored with this model;
- stage: `candidate` when added. `promote` makes it `production` only with an attached score of
  the kind's promotion suite (registry.promotion): every hull present, none aborted, on real or
  procedural defects as configured, its values within the limits. The version it replaces is
  archived.
- events: when it was added, scored, promoted and archived.

`registry:<name>` (its production version) or `registry:<name>@<version>` names a registered model
wherever a model file is taken (detector.model, stations.model, --segmenter, segment --model,
report --stations-model); the file is checked against its digest. `package` copies a production
JSON model into the package's models directory, for the CI gate, only when it may be redistributed.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import stat
from datetime import datetime, timezone
from importlib.resources import files

KINDS = ('region_classifier', 'anode_stations', 'segmenter')
STAGES = ('candidate', 'production', 'archived')
LIACI_LICENSE = 'CC BY-NC-SA 4.0 (adaptation of LIACi)'
PREFIX = 'registry:'


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec='seconds')


def root_of(cfg: dict, root: str | None = None) -> str:
    return root or cfg['registry']['root']


# ---------------------------------------------------------------------------------------------- reading models
def _record(path: str) -> tuple[str, dict]:
    """The kind of a model file and its own record (JSON content, or a segmenter's meta)."""
    if path.endswith('.pt'):
        from .segmenter import _torch
        blob = _torch().load(path, map_location='cpu', weights_only=True)
        return 'segmenter', json.loads(blob['meta'])
    with open(path) as f:
        model = json.load(f)
    if 'statuses' in model and 'paints' in model:
        return 'anode_stations', model
    if str(model.get('kind', '')).startswith('region classifier'):
        return 'region_classifier', model
    raise ValueError(f'{path}: not a model this registry knows ({", ".join(KINDS)})')


def benchmark_id(kind: str, path: str, record: dict) -> str:
    """The model's identity in a benchmark hull's versions."""
    if kind == 'segmenter':
        with open(path, 'rb') as f:
            return 'segmenter:' + _sha256_bytes(f.read())
    return _sha256_bytes(json.dumps(record, sort_keys=True).encode())


def _scenario_dirs(kind: str, record: dict) -> tuple[list[str], list[str]]:
    """Scenario directories behind a model's training data, and the LIACi datasets among its sources."""
    scenarios, liaci = [], []
    if kind == 'segmenter':
        for src in record.get('trained_on', []):
            path = os.path.join(src['dataset'], 'annotations.json')
            if not os.path.exists(path):
                scenarios.append(os.path.join(src['dataset'], '<missing>'))
                continue
            with open(path) as f:
                ann = json.load(f)
            if (ann.get('source') or {}).get('name') == 'LIACi':
                liaci.append(src['dataset'])
            scenarios += sorted({os.path.dirname(os.path.normpath(fr['run'])) for fr in ann['frames'].values() if fr.get('run')})
    else:
        scenarios = [os.path.dirname(os.path.normpath(src['run'])) for src in record.get('trained_on', [])]
    return scenarios, liaci


def license_of(kind: str, record: dict) -> dict:
    """What the model was derived from, its license and whether this project may redistribute it."""
    scenarios, liaci = _scenario_dirs(kind, record)
    patched, missing = [], []
    for d in scenarios:
        path = os.path.join(d, 'scenario.json')
        if not os.path.exists(path):
            missing.append(d)
            continue
        with open(path) as f:
            if json.load(f).get('patch_library'):
                patched.append(d)
    derived = ([f'LIACi images: {", ".join(liaci)}'] if liaci else []) + \
              ([f'hulls painted with LIACi defects: {len(patched)}'] if patched else []) + \
              (['torchvision MobileNetV2 ImageNet weights (encoder)'] if kind == 'segmenter' else [])
    if liaci or patched:
        lic = LIACI_LICENSE
    elif missing:
        lic = 'unknown: training sources not found (' + ', '.join(missing[:3]) + (', ...' if len(missing) > 3 else '') + ')'
    else:
        lic = 'this project (procedural simulation only)'
    return {'license': lic, 'derived_from': derived, 'redistributable': kind != 'segmenter' and not (liaci or patched or missing)}


# ---------------------------------------------------------------------------------------------- the registry
def _versions(root: str, name: str) -> list[int]:
    d = os.path.join(root, name)
    return sorted(int(v[1:]) for v in os.listdir(d) if v.startswith('v') and v[1:].isdigit()) if os.path.isdir(d) else []


def _card_path(root: str, name: str, version: int) -> str:
    return os.path.join(root, name, f'v{version}', 'card.json')


def _write_card(root: str, card: dict):
    path = _card_path(root, card['name'], card['version'])
    with open(path + '.tmp', 'w') as f:
        json.dump(card, f, indent=1)
    os.replace(path + '.tmp', path)


def card(ref: str, cfg: dict, root: str | None = None) -> dict:
    """The card of `name` (its production version) or `name@version`."""
    root = root_of(cfg, root)
    ref = ref[len(PREFIX):] if ref.startswith(PREFIX) else ref
    name, _, sel = ref.partition('@')
    versions = _versions(root, name)
    if not versions:
        raise ValueError(f'no model {name!r} in the registry at {root}')
    if sel in ('', 'production'):
        cards = [c for c in (_load(root, name, v) for v in versions) if c['stage'] == 'production']
        if not cards:
            raise ValueError(f'{name} has no production version (versions {versions}); name one: {name}@<version>')
        return cards[0]
    if not sel.isdigit() or int(sel) not in versions:
        raise ValueError(f'{name} has no version {sel!r} (versions {versions})')
    return _load(root, name, int(sel))


def _load(root: str, name: str, version: int) -> dict:
    with open(_card_path(root, name, version)) as f:
        return json.load(f)


def add(model_path: str, name: str, cfg: dict, root: str | None = None, note: str = '') -> dict:
    """Register a model file as the next version of `name`."""
    root = root_of(cfg, root)
    if not name or '@' in name or '/' in name or name.startswith('.'):
        raise ValueError(f'invalid model name {name!r}')
    kind, record = _record(model_path)
    for v in _versions(root, name):
        if _load(root, name, v)['kind'] != kind:
            raise ValueError(f'{name} holds {_load(root, name, v)["kind"]} models, not {kind}')
    with open(model_path, 'rb') as f:
        data = f.read()
    version = (_versions(root, name) or [0])[-1] + 1
    d = os.path.join(root, name, f'v{version}')
    os.makedirs(d)
    dest = os.path.join(d, 'model' + ('.pt' if kind == 'segmenter' else '.json'))
    with open(dest, 'wb') as f:
        f.write(data)
    os.chmod(dest, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    provenance = {'trained_on': record.get('trained_on', []), 'settings_sha256': record.get('settings_sha256')}
    if kind == 'segmenter':
        provenance.update({k: record.get(k) for k in ('backbone_sha256', 'epoch', 'val_map', 'seed')})
    c = {'name': name, 'version': version, 'kind': kind, 'file': os.path.basename(dest), 'sha256': _sha256_bytes(data),
         'bytes': len(data), 'benchmark_id': benchmark_id(kind, dest, record), 'registered_from': os.path.abspath(model_path),
         'note': note, 'provenance': provenance, **license_of(kind, record), 'stage': 'candidate', 'scores': [],
         'events': [{'at': _now(), 'event': 'added'}]}
    _write_card(root, c)
    return c


def listing(cfg: dict, root: str | None = None) -> list[dict]:
    root = root_of(cfg, root)
    out = []
    for name in sorted(os.listdir(root)) if os.path.isdir(root) else []:
        for v in _versions(root, name):
            c = _load(root, name, v)
            out.append({k: c[k] for k in ('name', 'version', 'kind', 'stage', 'license', 'redistributable')}
                       | {'sha256': c['sha256'][:12], 'scores': [s['suite'] for s in c['scores']]})
    return out


def attach(ref: str, score_path: str, cfg: dict, root: str | None = None) -> dict:
    """Attach a benchmark summary to a version: every hull in it must have been scored with this model."""
    root = root_of(cfg, root)
    c = card(ref, cfg, root)
    with open(score_path, 'rb') as f:
        data = f.read()
    s = json.loads(data)
    key = 'station_model_sha256' if c['kind'] == 'anode_stations' else 'detector_model_sha256'
    used = {v.get(key) for v in s.get('versions', [])}
    if used != {c['benchmark_id']}:
        raise ValueError(f'{score_path}: its hulls were not all scored with {c["name"]}@{c["version"]} ({key}: {sorted(map(str, used))})')
    c['scores'].append({'suite': s['suite'], 'path': os.path.abspath(score_path), 'sha256': _sha256_bytes(data),
                        'hulls': s['hulls'], 'missing': s['missing'], 'aborted': sum(1 for h in s.get('per_hull', []) if h.get('aborted')),
                        'patch_libraries': s.get('patch_libraries', ['procedural']),
                        'values': {k: v['value'] for k, v in s['values'].items()}, 'attached_at': _now()})
    c['events'].append({'at': _now(), 'event': 'scored', 'suite': s['suite']})
    _write_card(root, c)
    return c


def promotion_failures(c: dict, cfg: dict) -> list[str]:
    """Why a version may not be promoted (empty: it may), from its newest score of the promotion suite."""
    rule = cfg['registry']['promotion'][c['kind']]
    scores = [s for s in c['scores'] if s['suite'] == rule['suite']]
    if not scores:
        return [f'no score of the {rule["suite"]} suite attached (benchmark score, then registry attach)']
    s = scores[-1]
    fails = []
    if s['missing']:
        fails.append(f'hulls missing from the score: {s["missing"]}')
    if s['aborted']:
        fails.append(f'{s["aborted"]} hulls aborted')
    real = s['patch_libraries'] != ['procedural']
    if real != rule['real_defects'] or len(s['patch_libraries']) != 1:
        fails.append(f'the score must be on {"real" if rule["real_defects"] else "procedural"} defects only '
                     f'(its hulls: {s["patch_libraries"]})')
    for k, lo in rule['min'].items():
        if s['values'].get(k) is None or s['values'][k] < lo:
            fails.append(f'{k} = {s["values"].get(k)} below {lo}')
    for k, hi in rule['max'].items():
        if s['values'].get(k) is None or s['values'][k] > hi:
            fails.append(f'{k} = {s["values"].get(k)} above {hi}')
    return fails


def promote(ref: str, cfg: dict, root: str | None = None) -> dict:
    root = root_of(cfg, root)
    if not ref.partition('@')[2].isdigit():
        raise ValueError('promote: name the version, <name>@<version>')
    c = card(ref, cfg, root)
    fails = promotion_failures(c, cfg)
    if fails:
        raise ValueError(f'{c["name"]}@{c["version"]} may not be promoted: ' + '; '.join(fails))
    for v in _versions(root, c['name']):
        other = _load(root, c['name'], v)
        if other['stage'] == 'production' and v != c['version']:
            other['stage'] = 'archived'
            other['events'].append({'at': _now(), 'event': 'archived', 'replaced_by': c['version']})
            _write_card(root, other)
    c['stage'] = 'production'
    c['events'].append({'at': _now(), 'event': 'promoted'})
    _write_card(root, c)
    return c


def resolve(ref: str, cfg: dict, root: str | None = None) -> str:
    """The file of a model reference: a path as given, or `registry:<name>[@<version>]` checked against its digest."""
    if not ref.startswith(PREFIX):
        return ref
    root = root_of(cfg, root)
    c = card(ref, cfg, root)
    path = os.path.join(root, c['name'], f'v{c["version"]}', c['file'])
    with open(path, 'rb') as f:
        if _sha256_bytes(f.read()) != c['sha256']:
            raise ValueError(f'{path} does not match its registered digest')
    return path


def package(ref: str, cfg: dict, root: str | None = None, dest_dir: str | None = None) -> str:
    """Copy a production JSON model into the package's models directory, as the file the package's
    default configuration loads."""
    root = root_of(cfg, root)
    c = card(ref, cfg, root)
    if c['stage'] != 'production':
        raise ValueError(f'{c["name"]}@{c["version"]} is {c["stage"]}, not production')
    if not c['redistributable']:
        raise ValueError(f'{c["name"]}@{c["version"]} may not be redistributed ({c["license"]})')
    from .config import default_config
    defaults = default_config()                      # the file the package loads by default, not this run's reference
    target = {'region_classifier': defaults['detector']['model'], 'anode_stations': defaults['stations']['model']}[c['kind']]
    if not target or target.startswith(PREFIX):
        raise ValueError(f'the package has no {c["kind"]} file to replace')
    dest = os.path.join(dest_dir or str(files('auvinspect').joinpath('models')), os.path.basename(target))
    shutil.copyfile(resolve(f'{PREFIX}{c["name"]}@{c["version"]}', cfg, root), dest)
    os.chmod(dest, stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH)
    return dest
