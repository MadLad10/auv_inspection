"""Real defects for the simulator, on a stand-in with LIACi's layout: defects of known reflectance relative
to the paint, under a lamp whose light falls off across the image."""
import copy
import csv
import json

import numpy as np
import pytest
from PIL import Image
from scipy import ndimage

from auvinspect import config, evolution, liaci, patches, scenario

CATEGORIES = ['sea_chest_grating', 'paint_peel', 'over_board_valve', 'defect', 'corrosion', 'propeller', 'anode',
              'bilge_keel', 'marine_growth', 'ship_hull']
PAINT = np.array([0.25, 0.12, 0.08])                         # linear reflectance of their hull paint
REL = {'corrosion': (2.0, 1.5, 1.0), 'paint_peel': (2.5, 3.0, 3.5), 'marine_growth': (0.5, 0.8, 0.6), 'anode': (1.5, 1.5, 1.5)}
W, H = 600, 400


def _square(x0, y0, x1, y1):
    return [[x0, y0, x1, y0, x1, y1, x0, y1]]


def _srgb(v):
    v = np.clip(v, 0, 1)
    return (np.where(v <= 0.0031308, 12.92 * v, 1.055 * v ** (1 / 2.4) - 0.055) * 255 + 0.5).astype(np.uint8)


@pytest.fixture(scope='module')
def root(tmp_path_factory):
    tmp = tmp_path_factory.mktemp('liaci')
    good = [('corrosion', _square(300, 100, 360, 160)), ('paint_peel', _square(60, 100, 120, 160)),
            ('marine_growth', _square(450, 100, 510, 160))]
    images = [  # file, official split, [(category, polygon)] drawn in order (later on top)
        ('image_0001.jpg', 'train', good + [
            ('paint_peel', _square(0, 300, 80, 360)),                                   # cut by the border
            ('marine_growth', _square(200, 300, 230, 330)),                             # small
            ('corrosion', _square(300, 260, 360, 320)), ('anode', _square(300, 260, 360, 300))]),   # overlapped
        ('image_0002.jpg', 'test', [('corrosion', _square(300, 100, 360, 160))]),
        ('image_0301.jpg', 'train', [('corrosion', _square(300, 100, 360, 160))]),     # a validation block
    ]
    (tmp / 'images').mkdir()
    coco = {'images': [], 'annotations': [], 'categories': [{'id': k + 1, 'name': n} for k, n in enumerate(CATEGORIES)]}
    rows = []
    light = np.exp(0.3 * np.arange(W) / W)[None, :, None]                              # the lamp falls off across the image
    for k, (name, split, objs) in enumerate(images, 1):
        lin = np.broadcast_to(PAINT, (H, W, 3)).copy()
        for cat, poly in [('ship_hull', _square(0, 0, W - 1, H - 1))] + objs:
            if cat != 'ship_hull':
                lin[liaci._raster(poly, 1.0, (W, H))] = PAINT * REL[cat]
            x0, y0, x1, y1 = poly[0][0], poly[0][1], poly[0][4], poly[0][5]
            coco['annotations'].append({'id': len(coco['annotations']) + 1, 'image_id': k, 'category_id': CATEGORIES.index(cat) + 1,
                                        'segmentation': poly, 'area': float((x1 - x0) * (y1 - y0)), 'bbox': [x0, y0, x1 - x0, y1 - y0], 'iscrowd': 0})
        Image.fromarray(_srgb(lin * light)).save(tmp / 'images' / name, format='PNG')
        coco['images'].append({'id': k, 'width': W, 'height': H, 'file_name': name})
        rows.append({'': k - 1, 'file_name': name, 'split': split})
    (tmp / 'coco-labels.json').write_text(json.dumps(coco))
    with open(tmp / 'train_test_split.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['', 'file_name', 'split'])
        w.writeheader()
        w.writerows(rows)
    return tmp


@pytest.fixture(scope='module')
def library(root, tmp_path_factory):
    out = tmp_path_factory.mktemp('patches')
    s = patches.build(str(root), str(out), config.load())
    return patches.Library(str(out)), s


@pytest.fixture(scope='module')
def flat_cfg():
    return config.load(overrides={'hull': {'length_m': 6.0}, 'appearance': {'slime_strength': [0.0, 0.0]}})


def test_library_keeps_whole_defects_on_plain_hull_from_the_training_split_only(library, root, tmp_path):
    lib, s = library
    assert s['patches'] == {'corrosion': 1, 'coating_damage': 1, 'biofouling': 1}      # not the test or validation images
    assert s['rejected']['coating_damage'] == {'cut by the border': 1}
    assert s['rejected']['biofouling'] == {'small': 1} and s['rejected']['corrosion'] == {'overlapped': 1}
    for cls, cat in (('corrosion', 'corrosion'), ('coating_damage', 'paint_peel'), ('biofouling', 'marine_growth')):
        (e,) = lib.index['classes'][cls]
        assert e['file'] == 'image_0001.jpg' and e['pixels'] == 61 * 61
        rel, mask = lib.patch(cls, e)
        np.testing.assert_allclose(np.median(rel[mask], 0), REL[cat], rtol=0.03)        # paint, lamp colour and falloff divided out
    again = patches.build(str(root), str(tmp_path), config.load())
    assert again['library_sha256'] == lib.digest == s['library_sha256']


def test_a_hull_with_real_defects_is_reproducible_with_exact_labels(library, flat_cfg, tmp_path):
    lib, _ = library
    scn = scenario.generate(3, flat_cfg, lib)
    assert scenario.generate(3, flat_cfg, lib).digest() == scn.digest()
    assert scenario.generate(3, flat_cfg).digest() != scn.digest()
    assert 'patch_library' not in scenario.generate(3, flat_cfg).spec()                 # procedural hulls keep their digests
    assert scn.patch_library['library_sha256'] == lib.digest and scn.patch_library['license'] == 'CC BY-NC-SA 4.0'
    by_class = {}
    for it in scn.defects():
        by_class.setdefault(it['class'], []).append(it)
        assert ('patch' in it) == (it['class'] in patches.PAINTED)                      # cracks, missing anodes: procedural
    tex = scn.texture.astype(float)
    for cls, cat in (('corrosion', 'corrosion'), ('coating_damage', 'paint_peel'), ('biofouling', 'marine_growth')):
        it = max(by_class[cls], key=lambda i: i['area_m2'])
        m = scn.labels == it['id']
        assert abs(m.sum() * scn.mm_per_px ** 2 / 1e6 - it['area_m2']) < 1e-9                # the label is the painted patch
        inner = ndimage.binary_erosion(m, iterations=3)
        ring = ndimage.binary_dilation(m, iterations=12) & ~ndimage.binary_dilation(m, iterations=4) & (scn.labels < 0)
        ratio = np.median(tex[inner], 0) / np.median(tex[ring], 0)
        np.testing.assert_allclose(ratio, np.minimum(REL[cat], 1 / np.asarray(flat_cfg['appearance']['paint_rgb'])), rtol=0.1)
    scn.save(str(tmp_path))
    back = scenario.Scenario.load(str(tmp_path), flat_cfg)
    assert back.digest() == scn.digest() and back.patch_library == scn.patch_library


def test_a_hull_with_real_defects_evolves_with_its_own_library_only(library, flat_cfg):
    lib, _ = library
    scn = scenario.generate(3, flat_cfg, lib)
    with pytest.raises(ValueError):
        evolution.evolve(scn, 12, flat_cfg)
    with pytest.raises(ValueError):
        evolution.evolve(scenario.generate(3, flat_cfg), 12, flat_cfg, lib)
    later = evolution.evolve(scn, 12, flat_cfg, lib)
    assert evolution.evolve(scn, 12, flat_cfg, lib).digest() == later.digest() and later.patch_library == scn.patch_library
    grown = 0
    for it in scn.defects():
        if 'patch' not in it:
            continue
        old, new = scn.labels == it['id'], (later.labels == it['id']) & (scn.labels != it['id'])
        assert (later.texture[old] == scn.texture[old]).all()                             # old pixels untouched
        colours = {tuple(c) for c in scn.texture[old]}
        assert all(tuple(c) in colours for c in later.texture[new])                       # growth continues the patch's own texture
        grown += int(new.sum())
    assert grown > 0
    assert all('patch' in it for it in later.defects() if it.get('new_after_months') and it['class'] in patches.PAINTED)


def test_patch_settings_are_validated():
    bad = copy.deepcopy(config.load())
    bad['real_images']['liaci']['patches']['reflectance'] = [5.0, 1.0]
    with pytest.raises(ValueError):
        config.validate(bad)
