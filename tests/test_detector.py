"""The region classifier of the detector: labels from the ground truth, the network, the packaged model."""
import copy

import numpy as np
import pytest

from auvinspect import baseline, config, dataset, detector


def test_regions_are_labelled_from_the_ground_truth(small_survey, small_cfg):
    X, y = detector.regions(str(small_survey / 'scn' / 'survey'), small_cfg)
    names = detector.classes(small_cfg)
    assert X.shape == (len(y), 26) and np.isfinite(X).all()
    found = {names[k] for k in set(y.tolist())}
    assert 'none' in found and found & {'corrosion', 'coating_damage', 'biofouling'}   # defects and everything else


def test_the_network_learns_and_is_deterministic():
    cfg = config.load()
    rng = np.random.default_rng(1)
    centres = rng.normal(0, 3, (3, 26))
    y = np.repeat([0, 1, 2], 60)
    X = centres[y] + rng.normal(0, 1, (len(y), 26))
    cfg['detector']['iterations'] = 300
    net = detector.fit(X, y, 3, cfg)
    assert (detector.predict(net, X).argmax(1) == y).mean() > 0.95
    assert detector.fit(X, y, 3, cfg) == net                                # same data, same weights
    p = detector.predict(net, X[:5])
    assert np.allclose(p.sum(1), 1.0)


def test_the_packaged_classifier_is_trained_on_training_hulls_of_every_paint():
    cfg = config.load()
    model = detector.load_model(cfg)
    assert model['classes'] == detector.classes(cfg)
    assert all(dataset.split_of(t['seed'], cfg['perception']['splits']) == 'train' for t in model['trained_on'])
    paints = {tuple(t['paint_rgb']) for t in model['trained_on']}
    assert {tuple(p) for p in cfg['benchmark']['world']['paint_rgb']} <= paints
    other = copy.deepcopy(cfg)
    other['detector']['context_px'] += 1
    with pytest.raises(ValueError):
        detector.load_model(other)
    rules = copy.deepcopy(cfg)
    rules['detector']['model'] = None
    assert detector.load_model(rules) is None                              # null: the hand-written rules


def test_detect_takes_its_classes_from_the_classifier():
    cfg = config.load()
    h, w = 240, 320
    yy, xx = np.mgrid[0:h, 0:w]
    lamp = np.exp(-(((xx - w / 2) / (0.5 * w)) ** 2 + ((yy - h / 2) / (0.5 * h)) ** 2))
    img = np.clip(np.array([0.45, 0.42, 0.25]) * (0.35 + 0.65 * lamp)[..., None]
                  * (1 + np.random.default_rng(0).normal(0, 0.02, (h, w, 3))), 0, 1)
    img[100:150, 180:230] *= np.array([1.4, 1.8, 2.0])
    rgb = (img * 255).astype(np.uint8)

    class Always:
        context_px = 4

        def __init__(self, cls):
            self.cls = cls

        def classify(self, comps):
            return [(self.cls, 0.8) for _ in comps]
    assert [d['class'] for d in baseline.detect(rgb, cfg, classifier=Always('corrosion'))] == ['corrosion']
    assert baseline.detect(rgb, cfg, classifier=Always(None)) == []         # none: not reported
