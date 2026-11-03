"""The model registry: versions, license from the training sources, scores made with the model,
promotion rules, digests checked on use, packaging only what may be redistributed."""
import copy
import hashlib
import json
import os

import pytest

from auvinspect import config, detector, registry


def _hull(tmp_path, name, patched):
    d = tmp_path / name
    (d / 'survey').mkdir(parents=True)
    spec = {'seed': 1, 'digest': name, **({'patch_library': {'library_sha256': 'x' * 64}} if patched else {})}
    (d / 'scenario.json').write_text(json.dumps(spec))
    return str(d / 'survey')


def _classifier(tmp_path, runs, tag='a'):
    model = {'kind': 'region classifier: test', 'tag': tag, 'settings_sha256': detector._digest(detector.settings(config.load())),
             'trained_on': [{'run': r, 'scenario_digest': os.path.basename(os.path.dirname(r))} for r in runs]}
    path = tmp_path / f'rc_{tag}.json'
    path.write_text(json.dumps(model))
    return str(path), model


def _score(tmp_path, ident, suite='dev', libraries=('procedural',), **values):
    vals = {'map_recall_confirmed': 0.9, 'map_precision_confirmed': 0.95, 'false_confirmed_per_hull': 0.5, **values}
    s = {'suite': suite, 'hulls': 4, 'missing': [], 'per_hull': [{'aborted': None}] * 4, 'patch_libraries': list(libraries),
         'values': {k: {'value': v} for k, v in vals.items()},
         'versions': [{'detector_model_sha256': ident, 'station_model_sha256': 'other'}]}
    path = tmp_path / f'score_{suite}_{len(os.listdir(tmp_path))}.json'
    path.write_text(json.dumps(s))
    return str(path)


@pytest.fixture
def cfg(tmp_path):
    c = copy.deepcopy(config.load())
    c['registry']['root'] = str(tmp_path / 'registry')
    return c


def test_versions_are_read_only_copies_with_their_benchmark_identity(tmp_path, cfg):
    path, model = _classifier(tmp_path, [_hull(tmp_path, 'h1', False)])
    c1 = registry.add(path, 'rc', cfg)
    c2 = registry.add(path, 'rc', cfg, note='again')
    assert (c1['version'], c2['version'], c1['kind'], c1['stage']) == (1, 2, 'region_classifier', 'candidate')
    assert c1['benchmark_id'] == hashlib.sha256(json.dumps(model, sort_keys=True).encode()).hexdigest()   # as benchmark.py records it
    stored = registry.resolve('registry:rc@1', cfg)
    assert json.load(open(stored)) == model and not os.access(stored, os.W_OK)
    assert [r['version'] for r in registry.listing(cfg)] == [1, 2]
    with pytest.raises(ValueError):
        registry.add(path, 'bad@name', cfg)


def test_license_follows_the_training_sources(tmp_path, cfg):
    plain = registry.add(_classifier(tmp_path, [_hull(tmp_path, 'h1', False)], 'p')[0], 'plain', cfg)
    real = registry.add(_classifier(tmp_path, [_hull(tmp_path, 'h2', False), _hull(tmp_path, 'h3', True)], 'r')[0], 'real', cfg)
    gone = registry.add(_classifier(tmp_path, [str(tmp_path / 'nowhere' / 'survey')], 'g')[0], 'gone', cfg)
    assert plain['redistributable'] and plain['license'].startswith('this project')
    assert not real['redistributable'] and real['license'] == registry.LIACI_LICENSE
    assert not gone['redistributable'] and gone['license'].startswith('unknown')


def test_scores_must_come_from_the_model_and_promotion_follows_the_rules(tmp_path, cfg):
    path, _ = _classifier(tmp_path, [_hull(tmp_path, 'h1', False)])
    c = registry.add(path, 'rc', cfg)
    with pytest.raises(ValueError, match='not all scored'):
        registry.attach('rc@1', _score(tmp_path, 'someone else'), cfg)
    with pytest.raises(ValueError, match='no score'):
        registry.promote('rc@1', cfg)
    registry.attach('rc@1', _score(tmp_path, c['benchmark_id'], map_recall_confirmed=0.5), cfg)
    with pytest.raises(ValueError, match='map_recall_confirmed'):
        registry.promote('rc@1', cfg)
    registry.attach('rc@1', _score(tmp_path, c['benchmark_id'], libraries=('ab' * 32,)), cfg)
    with pytest.raises(ValueError, match='procedural defects'):
        registry.promote('rc@1', cfg)
    registry.attach('rc@1', _score(tmp_path, c['benchmark_id']), cfg)
    assert registry.promote('rc@1', cfg)['stage'] == 'production'
    c2 = registry.add(path, 'rc', cfg)
    registry.attach('rc@2', _score(tmp_path, c2['benchmark_id']), cfg)
    registry.promote('rc@2', cfg)
    assert registry.card('rc@1', cfg)['stage'] == 'archived' and registry.card('rc', cfg)['version'] == 2
    assert [e['event'] for e in registry.card('rc@1', cfg)['events']] == ['added', 'scored', 'scored', 'scored', 'promoted', 'archived']


def test_a_registered_model_is_checked_where_it_is_used_and_packaged_only_when_free(tmp_path, cfg):
    real, _ = _classifier(tmp_path, [_hull(tmp_path, 'h2', True)], 'r')
    plain, model = _classifier(tmp_path, [_hull(tmp_path, 'h1', False)], 'p')
    for name, path in (('real', real), ('plain', plain)):
        c = registry.add(path, name, cfg)
        registry.attach(f'{name}@1', _score(tmp_path, c['benchmark_id']), cfg)
        registry.promote(f'{name}@1', cfg)
    cfg['detector']['model'] = 'registry:plain'
    assert detector.load_model(cfg) == model
    with pytest.raises(ValueError, match='redistributed'):
        registry.package('real', cfg, dest_dir=str(tmp_path))
    dest = registry.package('plain', cfg, dest_dir=str(tmp_path))
    assert os.path.basename(dest) == 'region_classifier.json' and json.load(open(dest)) == model
    stored = registry.resolve('registry:plain', cfg)
    os.chmod(stored, 0o644)
    with open(stored, 'a') as f:
        f.write(' ')
    with pytest.raises(ValueError, match='digest'):
        detector.load_model(cfg)


def test_a_segmenter_trained_on_liaci_is_never_redistributable(tmp_path, cfg):
    torch = pytest.importorskip('torch')
    ds = tmp_path / 'liaci'
    ds.mkdir()
    (ds / 'annotations.json').write_text(json.dumps({'source': {'name': 'LIACi'}, 'frames': {}}))
    meta = {'classes': cfg['perception']['classes'], 'trained_on': [{'dataset': str(ds)}], 'backbone_sha256': 'b' * 64, 'epoch': 3}
    path = tmp_path / 'seg.pt'
    torch.save({'state': {}, 'meta': json.dumps(meta)}, path)
    c = registry.add(str(path), 'seg', cfg)
    assert c['kind'] == 'segmenter' and not c['redistributable'] and c['license'] == registry.LIACI_LICENSE
    assert c['benchmark_id'] == 'segmenter:' + hashlib.sha256(path.read_bytes()).hexdigest()
    assert c['provenance']['epoch'] == 3
