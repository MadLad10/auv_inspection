"""The LIACi import, on a small stand-in with LIACi's layout (the real images are not redistributable)."""
import copy
import csv
import json

import numpy as np
import pytest
from PIL import Image

from auvinspect import baseline, config, liaci, perception

CATEGORIES = ['sea_chest_grating', 'paint_peel', 'over_board_valve', 'defect', 'corrosion', 'propeller', 'anode',
              'bilge_keel', 'marine_growth', 'ship_hull']


def _square(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


@pytest.fixture
def root(tmp_path):
    images = [  # file, size, official split, [(category, polygon)]
        ('image_0001.jpg', (1280, 720), 'train', [('ship_hull', _square(0, 0, 1280, 720)), ('paint_peel', _square(100, 100, 500, 400)),
                                                  ('corrosion', _square(200, 200, 260, 260)), ('defect', _square(800, 100, 900, 200))]),
        ('image_0002.jpg', (640, 480), 'test', [('marine_growth', _square(10, 10, 210, 110)), ('anode', _square(400, 300, 404, 304))]),
        ('image_0003.jpg', (640, 480), 'not_used', [('corrosion', _square(10, 10, 100, 100))]),
        ('image_0301.jpg', (1920, 1080), 'train', [('anode', _square(900, 500, 1100, 600))]),
    ]
    (tmp_path / 'images').mkdir()
    coco = {'images': [], 'annotations': [], 'categories': [{'id': k + 1, 'name': n} for k, n in enumerate(CATEGORIES)]}
    rows = []
    for k, (name, (W, H), split, objs) in enumerate(images, 1):
        Image.fromarray(np.full((H, W, 3), 90, np.uint8)).save(tmp_path / 'images' / name, format='PNG')   # LIACi's .jpg are PNG
        coco['images'].append({'id': k, 'width': W, 'height': H, 'file_name': name})
        for cat, poly in objs:
            x0, y0, x1, y1 = poly[0][0], poly[0][1], poly[0][4], poly[0][5]
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': k, 'category_id': CATEGORIES.index(cat) + 1,
                                        'segmentation': poly, 'area': float((x1 - x0) * (y1 - y0)), 'bbox': [x0, y0, x1 - x0, y1 - y0], 'iscrowd': 0})
        rows.append({'': k - 1, 'file_name': name, 'split': split})
    (tmp_path / 'coco-labels.json').write_text(json.dumps(coco))
    with open(tmp_path / 'train_test_split.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['', 'file_name', 'split'])
        w.writeheader()
        w.writerows(rows)
    return tmp_path


def test_the_import_keeps_the_official_split_and_maps_the_classes(root, tmp_path):
    cfg = config.load()
    out = tmp_path / 'out'
    s = liaci.import_dataset(str(root), str(out), cfg)
    ann = json.loads((out / 'annotations.json').read_text())
    assert ann['splits'] == {'train': ['liaci_image_0001'], 'val': ['liaci_image_0301'], 'test': ['liaci_image_0002']}   # not_used left out
    assert s['objects'] == {'coating_damage': 1, 'corrosion': 1, 'biofouling': 1, 'anode': 2}               # hull and `defect` are background
    assert ann['source']['license'] == 'CC BY-NC-SA 4.0' and len(ann['source']['labels_sha256']) == 64

    fr = ann['frames']['liaci_image_0001']
    assert (fr['width'], fr['height']) == (640, 360)
    img = Image.open(out / fr['image'])
    assert img.size == (640, 360)
    objs = {o['class']: o for o in fr['objects']}
    assert objs['coating_damage']['box'] == [50, 50, 251, 201]                                             # half scale, inclusive edges
    mask = np.asarray(Image.open(out / 'masks' / 'train' / 'liaci_image_0001.png'))
    k = cfg['perception']['classes'].index
    assert mask[115, 115] == k('corrosion') + 1 and mask[60, 60] == k('coating_damage') + 1                 # the smaller on top
    assert mask[100, 425] == 0                                                                             # the `defect`: background
    tiny = [o for o in ann['frames']['liaci_image_0002']['objects'] if o['class'] == 'anode'][0]
    assert tiny['pixels'] < cfg['perception']['min_visible_px'] and not tiny['required']


def test_perfect_detections_score_full_marks_and_the_baseline_refuses_real_images(root, tmp_path):
    cfg = config.load()
    out = tmp_path / 'out'
    liaci.import_dataset(str(root), str(out), cfg)
    ann = json.loads((out / 'annotations.json').read_text())
    det = {fid: [{'class': o['class'], 'score': 0.9, 'box': o['box']} for o in fr['objects']] for fid, fr in ann['frames'].items()}
    m = perception.evaluate(ann, det, cfg)
    assert m['map'] == 1.0 and m['false_alarms_per_frame'] == 0
    with pytest.raises(ValueError):
        baseline.run(str(out), cfg)


def test_validation_blocks_and_class_mapping_checked():
    cfg = config.load()
    assert [liaci.split_of(f'image_{n:04d}.jpg', 'train', cfg) for n in (1, 300, 301, 350, 351)] == ['train', 'train', 'val', 'val', 'train']
    assert liaci.split_of('image_0301.jpg', 'test', cfg) == 'test' and liaci.split_of('image_0005.jpg', 'not_used', cfg) is None
    bad = copy.deepcopy(config.load())
    bad['real_images']['liaci']['classes']['defect'] = 'dent'
    with pytest.raises(ValueError):
        config.validate(bad)
