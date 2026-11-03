"""The deep segmenter on a tiny dataset, with a randomly initialised encoder standing in for the ImageNet
weights (which are not in this repository)."""
import copy
import hashlib
import json

import numpy as np
import pytest
from PIL import Image

from auvinspect import config, perception, segmenter

torch = pytest.importorskip('torch')


@pytest.fixture(scope='module')
def setup(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('seg')
    from torchvision.models import mobilenet_v2
    torch.manual_seed(0)
    backbone = tmp / 'backbone.pth'
    torch.save(mobilenet_v2().state_dict(), backbone)
    cfg = copy.deepcopy(config.load())
    cfg['segmenter'].update({'backbone_sha256': hashlib.sha256(backbone.read_bytes()).hexdigest(), 'input_width': 64,
                             'input_height': 48, 'epochs': 2, 'batch': 2, 'threads': 2, 'min_region_px': 20})
    classes = cfg['perception']['classes']
    ds = tmp / 'ds'
    frames, splits = {}, {'train': [], 'val': [], 'test': []}
    rng = np.random.default_rng(0)
    for k, split in enumerate(['train'] * 4 + ['val', 'test']):
        for sub in ('images', 'masks'):
            (ds / sub / split).mkdir(parents=True, exist_ok=True)
        img = np.full((96, 128, 3), (100, 40, 30), np.uint8)
        mask = np.zeros((96, 128), np.uint8)
        x0, y0 = int(rng.integers(10, 80)), int(rng.integers(10, 50))
        img[y0:y0 + 30, x0:x0 + 30] = (200, 120, 40)
        mask[y0:y0 + 30, x0:x0 + 30] = classes.index('corrosion') + 1
        fid = f'f{k}'
        Image.fromarray(img).save(ds / 'images' / split / f'{fid}.png')
        Image.fromarray(mask).save(ds / 'masks' / split / f'{fid}.png')
        frames[fid] = {'split': split, 'scenario_seed': None, 'width': 128, 'height': 96, 'image': f'images/{split}/{fid}.png',
                       'objects': [{'item': 0, 'class': 'corrosion', 'box': [x0, y0, x0 + 30, y0 + 30], 'pixels': 900,
                                    'required': True, 'size': 'unknown'}]}
        splits[split].append(fid)
    (ds / 'annotations.json').write_text(json.dumps({'classes': classes, 'splits': splits, 'frames': frames}))
    return cfg, ds, backbone, tmp


def test_regions_become_scored_boxes():
    cfg = copy.deepcopy(config.load())
    cfg['segmenter'].update({'input_width': 64, 'input_height': 48, 'min_region_px': 20})
    classes = cfg['perception']['classes']
    prob = np.zeros((len(classes) + 1, 48, 64), np.float32)
    prob[0] = 1.0
    for (r0, r1, c0, c1, p) in ((4, 12, 4, 14, 0.9), (30, 40, 40, 60, 0.7), (44, 45, 2, 3, 0.95)):   # the last is too small
        prob[0, r0:r1, c0:c1] = 1 - p
        prob[1, r0:r1, c0:c1] = p
    dets, pred = segmenter.frame_detections(prob, 128, 96, classes, cfg)                       # twice the input size
    assert pred.shape == (96, 128)
    boxes = sorted((d['box'], d['score']) for d in dets)
    assert [b for b, _ in boxes] == [[8, 8, 28, 24], [80, 60, 120, 80]] and all(d['class'] == classes[0] for d in dets)
    assert 0.8 < boxes[0][1] <= 0.9 and 0.6 < boxes[1][1] <= 0.7                              # mean over the region, edges blurred


def test_training_keeps_the_best_val_epoch_and_the_model_reloads(setup):
    cfg, ds, backbone, tmp = setup
    model = segmenter.train([str(ds)], str(backbone), cfg, seed=1, log=lambda line: None)
    meta = model['meta']
    assert len(meta['history']) == 2 and meta['epoch'] in (1, 2)
    assert meta['val_map'] == max(h['val_map'] for h in meta['history'])
    assert meta['trained_on'][0]['frames'] == 4
    path = tmp / 'model.pt'
    segmenter.save(model, str(path))
    net, meta2, run_cfg = segmenter.load(str(path), config.load())
    assert meta2 == meta and run_cfg['segmenter']['input_width'] == 64
    det, conf = segmenter.detect(net, str(ds), 'test', run_cfg)
    assert list(det) == ['f5'] and conf.sum() == 96 * 128
    ann = json.loads((ds / 'annotations.json').read_text())
    assert perception.evaluate(ann, det, cfg, 'test')['per_class']['corrosion']['required'] == 1
    iou = segmenter.pixel_iou(conf, cfg['perception']['classes'])
    assert set(iou) >= {'background', 'corrosion', 'mean_over_present'}


def test_the_encoder_weights_must_match_their_digest(setup):
    cfg, ds, backbone, _ = setup
    bad = copy.deepcopy(cfg)
    bad['segmenter']['backbone_sha256'] = '0' * 64
    with pytest.raises(ValueError):
        segmenter.train([str(ds)], str(backbone), bad, log=lambda line: None)
