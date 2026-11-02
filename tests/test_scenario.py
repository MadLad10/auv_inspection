import numpy as np
from PIL import Image, ImageFilter

from auvinspect import config, scenario
from auvinspect.scenario import Scenario


def test_same_seed_same_digest_and_different_seed_differs(small_cfg, small_scn):
    assert scenario.generate(3, small_cfg).digest() == small_scn.digest()
    assert scenario.generate(4, small_cfg).digest() != small_scn.digest()


def test_defect_areas_match_their_label_pixels(small_scn):
    px2 = (small_scn.mm_per_px / 1000) ** 2
    ids, counts = np.unique(small_scn.labels[small_scn.labels >= 0], return_counts=True)
    pixels = dict(zip(ids.tolist(), counts.tolist()))
    defects = small_scn.defects()
    assert defects, 'scenario should contain defects'
    for d in defects:
        assert abs(pixels[d['id']] * px2 - d['area_m2']) < 1e-6
    # anodes are geometry, never painted into the label raster
    assert not set(a['id'] for a in small_scn.anodes()) & set(pixels)


def test_defects_lie_inside_the_inspection_band_and_keep_their_gap(small_scn):
    cfg = small_scn.cfg
    g0, g1 = cfg['hull']['inspect_girth_m']
    m = cfg['hull']['inspect_margin_m']
    for d in small_scn.defects():
        x0, s0, x1, s1 = d['bbox_xs']
        if d['class'] != 'missing_anode':
            assert g0 - 1e-6 <= s0 and s1 <= g1 + 1e-6 and m - 1e-6 <= x0 and x1 <= small_scn.hull.length - m + 1e-6
    # no other defect's pixels within min_gap of a defect's pixels (Chebyshev distance, in pixels)
    gap = int(cfg['defects']['min_gap_m'] / (small_scn.mm_per_px / 1000)) - 1
    lab = small_scn.labels
    for d in small_scn.defects():
        rr, cc = np.nonzero(lab == d['id'])
        r0, r1, c0, c1 = max(0, rr.min() - gap), rr.max() + gap + 1, max(0, cc.min() - gap), cc.max() + gap + 1
        win = lab[r0:r1, c0:c1]
        near = Image.fromarray(((win == d['id']) * 255).astype(np.uint8)).filter(ImageFilter.MaxFilter(2 * gap + 1))
        others = set(np.unique(win[np.asarray(near) > 0]).tolist()) - {-1, d['id']}
        assert not others, (d['id'], others)


def test_save_load_round_trip_and_run_settings_do_not_change_the_world(tmp_path, small_scn):
    digest = small_scn.save(str(tmp_path))
    run_cfg = config.load(overrides={'hull': {'length_m': 6.0}, 'survey': {'standoff_m': 1.2}})
    loaded = Scenario.load(str(tmp_path), run_cfg)
    assert loaded.digest() == digest
    assert loaded.cfg['survey']['standoff_m'] == 1.2
    assert np.array_equal(loaded.labels, small_scn.labels)
