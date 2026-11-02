"""Perception datasets, the perception referee and the classical baseline."""
import json

import numpy as np
import pytest
from PIL import Image

from auvinspect import baseline, config, dataset, perception


def _obj(item, cls, box, required=True, size='medium'):
    return {'item': item, 'class': cls, 'box': box, 'required': required, 'size': size, 'pixels': 500, 'inspectable_share': 1.0}


def _ann(frames):
    return {'classes': ['corrosion', 'crack'], 'splits': {'train': list(frames), 'val': [], 'test': []},
            'frames': {fid: {'scenario_seed': seed, 'objects': objs} for fid, (seed, objs) in frames.items()}}


def test_referee_scores_hits_false_alarms_and_ignored_objects():
    cfg = config.load()
    ann = _ann({'a': (1, [_obj(0, 'corrosion', [10, 10, 50, 50]), _obj(1, 'corrosion', [100, 100, 120, 120], required=False)]),
                'b': (2, [_obj(0, 'corrosion', [10, 10, 50, 50], size='small')])})       # same item id, another hull
    perfect = {'a': [{'class': 'corrosion', 'score': 0.9, 'box': [11, 11, 50, 50]}],
               'b': [{'class': 'corrosion', 'score': 0.8, 'box': [10, 10, 49, 51]}]}
    m = perception.evaluate(ann, perfect, cfg)['per_class']['corrosion']
    assert m['ap'] == 1.0 and m['precision'] == 1.0 and m['recall'] == 1.0
    assert m['items'] == {'required': 2, 'found': 2, 'recall': 1.0, 'recall_by_size': {'medium': 1.0, 'small': 1.0}}
    noisy = {'a': perfect['a'] + [{'class': 'corrosion', 'score': 0.95, 'box': [300, 300, 340, 340]},   # false alarm
                                  {'class': 'corrosion', 'score': 0.7, 'box': [100, 100, 120, 120]},   # on the ignored object
                                  {'class': 'crack', 'score': 0.9, 'box': [10, 10, 50, 50]}],        # wrong class
             'b': []}
    r = perception.evaluate(ann, noisy, cfg)
    m = r['per_class']['corrosion']
    assert m['precision'] == 0.5 and m['recall'] == 0.5 and m['items']['found'] == 1
    assert m['ap'] == pytest.approx(0.25)          # the hit ranks second: precision 1/2 at recall 1/2
    assert r['false_alarms_per_frame'] == pytest.approx(1.0)   # the stray corrosion box and the crack box


@pytest.fixture(scope='module')
def small_dataset(small_survey, small_scn):
    root = small_survey
    summary = dataset.export([str(root / 'scn' / 'survey')], str(root / 'ds'), small_scn.cfg)
    return root, summary, json.load(open(root / 'ds' / 'annotations.json'))


def test_dataset_objects_match_the_ground_truth(small_dataset, small_scn):
    root, summary, ann = small_dataset
    cfg = small_scn.cfg
    split = dataset.split_of(small_scn.seed, cfg['perception']['splits'])
    assert summary['frames'][split] == len(ann['frames']) > 0
    classes = ann['classes']
    for fid, fr in list(ann['frames'].items())[:12]:
        gt = np.load(root / 'scn' / 'survey' / 'gt' / (fr['frame'] + '.npz'))
        mask = np.asarray(Image.open(root / 'ds' / 'masks' / split / (fid + '.png')))
        for o in fr['objects']:
            m = gt['label'] == o['item']
            rows, cols = np.nonzero(m)
            assert o['pixels'] == m.sum() and o['box'] == [cols.min(), rows.min(), cols.max() + 1, rows.max() + 1]
            assert np.all(mask[m] == classes.index(o['class']) + 1)
            share = gt['quality'][m].mean()
            assert o['required'] == (m.sum() >= cfg['perception']['min_visible_px'] and share >= cfg['perception']['min_inspectable_share'])
        lines = (root / 'ds' / 'labels' / split / (fid + '.txt')).read_text().splitlines()
        assert len(lines) == sum(o['pixels'] >= cfg['perception']['min_visible_px'] for o in fr['objects'])


def test_baseline_finds_corrosion_on_the_small_hull(small_dataset, small_scn):
    root, _, ann = small_dataset
    det = baseline.run(str(root / 'ds'), small_scn.cfg)
    m = perception.evaluate(ann, det, small_scn.cfg)
    assert m['per_class']['corrosion']['items']['recall'] >= 0.5


def _paint(h=240, w=320):
    yy, xx = np.mgrid[0:h, 0:w]
    lamp = np.exp(-(((xx - w / 2) / (0.5 * w)) ** 2 + ((yy - h / 2) / (0.5 * h)) ** 2))      # hot spot and falloff
    rng = np.random.default_rng(0)
    paint = np.array([0.45, 0.42, 0.25])
    return np.clip(paint * (0.35 + 0.65 * lamp)[..., None] * (1 + rng.normal(0, 0.02, (h, w, 3))), 0, 1)


def test_baseline_reports_a_patch_but_not_a_weld_seam():
    cfg = config.load()
    img = _paint()
    img[60:62, :] *= 0.6                                         # weld seam: a thin dark line across the image
    img[150:156, 40:120] *= 0.6                                  # a short thin seam fragment along the image axis
    bright = np.array([1.4, 1.8, 2.0])                            # flaked coating: brighter in every channel
    img[100:150, 180:200] *= bright                              # an L-shaped (not rectangular) patch
    img[130:150, 200:240] *= bright
    det = baseline.detect((img * 255).astype(np.uint8), cfg)
    assert [d['class'] for d in det] == ['coating_damage']
    assert perception.iou(det[0]['box'], [180, 100, 240, 150]) > 0.6
