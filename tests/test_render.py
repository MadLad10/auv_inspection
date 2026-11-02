import numpy as np
import pytest

from auvinspect import optics, scene
from auvinspect.coverage import Coverage
from auvinspect.render import LABEL_SOUND, InspectionRenderer
from auvinspect.rng import stream


def _face(rend, scn, x, s, standoff=1.0):
    h, cfg = scn.hull, scn.cfg
    cam = cfg['survey']['camera']
    pos = h.point(x, s) + standoff * h.normal(s) - np.asarray(cfg['cameras']['rig'][cam]['offset'])
    rend.set_pose(pos, 0.0, {cam: float(np.degrees(h.turn_angle(s)))})
    return cam


def _texture_correlation(scn, frame):
    px = scn.mm_per_px / 1000
    rows = np.clip((frame.s / px).astype(int), 0, scn.texture.shape[0] - 1)
    cols = np.clip((frame.x / px).astype(int), 0, scn.texture.shape[1] - 1)
    looked_up = scn.texture[rows, cols].astype(float).mean(-1)
    m = frame.on_hull
    return np.corrcoef(looked_up[m], frame.albedo.astype(float).mean(-1)[m])[0, 1]


@pytest.fixture(scope='module')
def rend(small_scn):
    r = InspectionRenderer(small_scn)
    yield r
    r.close()


def test_labels_align_with_rendered_texture(small_scn, rend):
    """Per-pixel ground truth (back-projected onto the analytic hull) matches what the camera renders."""
    d = max(small_scn.defects(), key=lambda i: i['area_m2'])
    cam = _face(rend, small_scn, *d['centroid_xs'])
    f = rend.capture(cam, stream(0, 'test'))
    assert f.on_hull.mean() > 0.5
    assert _texture_correlation(small_scn, f) > 0.95        # the unlit albedo pass is the texture itself
    assert (f.label == d['id']).sum() > 0


def test_the_other_texture_orientation_would_misalign(small_scn, monkeypatch):
    monkeypatch.setattr(scene, 'TEXTURE_FLIP_V', not scene.TEXTURE_FLIP_V)
    r = InspectionRenderer(small_scn)
    try:
        d = max(small_scn.defects(), key=lambda i: i['area_m2'])
        f = r.capture(_face(r, small_scn, *d['centroid_xs']), stream(0, 'test'))
        assert _texture_correlation(small_scn, f) < 0.5
    finally:
        r.close()


def test_anodes_are_labelled_from_geometry(small_scn, rend):
    a = small_scn.anodes()[0]
    cam = _face(rend, small_scn, *a['centroid_xs'])
    f = rend.capture(cam, stream(0, 'test'))
    m = f.label == a['id']
    assert m.sum() > 100 and f.on_anode[m].all() and not f.on_hull[m].any()


def test_capture_is_deterministic(small_scn, rend):
    cam = _face(rend, small_scn, 3.0, 2.0)
    a = rend.capture(cam, stream(5, 'noise'))
    b = rend.capture(cam, stream(5, 'noise'))
    assert np.array_equal(a.rgb, b.rgb) and np.array_equal(a.label, b.label)
    assert (a.label[a.on_hull] >= LABEL_SOUND).all()


@pytest.mark.parametrize('girth', [1.5, 3.6, 4.3])     # flat side, bilge turn, near the bottom
def test_analytic_raycast_matches_the_rendered_range(small_scn, rend, girth):
    """The onboard footprint and the standoff sounder use the analytic hull; the renderer uses the mesh."""
    cam = _face(rend, small_scn, 3.0, girth, standoff=1.5)
    f = rend.capture(cam, stream(0, 'test'))
    t = small_scn.hull.raycast(f.cam_pos, (rend.rays / rend.ray_norm[..., None]) @ f.cam_mat.T)
    m = f.on_hull
    assert np.isfinite(t[m]).all()
    assert np.percentile(np.abs(t[m] - f.range[m]), 99) < 0.01    # mesh chords on the bilge, exact on flats


def test_a_blacked_out_frame_keeps_its_ground_truth_but_covers_nothing(small_scn, rend):
    cam = _face(rend, small_scn, 3.0, 2.0)
    good = rend.capture(cam, stream(1, 'test'))
    black = rend.capture(cam, stream(1, 'test'), blackout=True)
    assert not black.rgb.any() and not black.signal
    assert np.array_equal(black.label, good.label)
    cov = Coverage(small_scn)
    assert not cov.quality(black).any() and cov.quality(good).any()


def _surface_points(scn, rend, f):
    """Each pixel's ray meeting the analytic hull (the surface the ground truth is computed on)."""
    d = (rend.rays / rend.ray_norm[..., None]) @ f.cam_mat.T
    return f.cam_pos + scn.hull.raycast(f.cam_pos, d)[..., None] * d


@pytest.mark.parametrize('girth', [1.5, 3.6])
def test_pixel_footprint_matches_finite_differences_on_the_hull(small_scn, rend, girth):
    """The exact footprint (camera.footprint_mm) against finite differences of neighbouring pixels'
    surface points: the larger singular value of [dP/du, dP/dv]."""
    cam = _face(rend, small_scn, 3.0, girth, standoff=1.5)
    f = rend.capture(cam, stream(0, 'test'))
    P = _surface_points(small_scn, rend, f)
    ok = np.isfinite(P).all(-1)
    P = np.where(ok[..., None], P, 0.0)
    du = (P[1:-1, 2:] - P[1:-1, :-2]) / 2
    dv = (P[2:, 1:-1] - P[:-2, 1:-1]) / 2
    numeric = np.linalg.svd(np.stack([du, dv], -1), compute_uv=False)[..., 0] * 1000
    inner = f.on_hull[1:-1, 1:-1] & ok[1:-1, 2:] & ok[1:-1, :-2] & ok[2:, 1:-1] & ok[:-2, 1:-1]
    inner &= f.incidence_deg[1:-1, 1:-1] <= small_scn.cfg['coverage']['max_incidence_deg']
    rel = np.abs(f.gsd_mm[1:-1, 1:-1][inner] / numeric[inner] - 1)
    assert inner.sum() > 10000 and np.percentile(rel, 99) < 0.01     # rendered depth vs analytic: below 1%
    if girth < small_scn.hull.side_depth:                       # a surface parallel to the image: the same everywhere
        assert np.allclose(f.gsd_mm[f.on_hull], 1500.0 / rend.focal, rtol=0.01)


def test_hull_frames_are_exposed_to_the_target(small_scn, rend):
    cfg = small_scn.cfg
    cam = _face(rend, small_scn, 3.0, 2.0, standoff=cfg['survey']['standoff_m'])
    f = rend.capture(cam, stream(2, 'test'))
    sensor = cfg['cameras']['sensor']
    H, W = f.rgb.shape[:2]
    fh, fw = int(H * sensor['auto_exposure']['metering_fraction'] / 2), int(W * sensor['auto_exposure']['metering_fraction'] / 2)
    meter = f.rgb[H // 2 - fh:H // 2 + fh, W // 2 - fw:W // 2 + fw].mean()
    assert meter == pytest.approx(sensor['auto_exposure']['target'] * 255, rel=0.1)
    assert f.gain < sensor['max_gain']


def test_turbid_water_washes_out_the_contrast_until_nothing_is_covered(small_scn, rend):
    cfg = small_scn.cfg
    cam = _face(rend, small_scn, 3.0, 2.0, standoff=cfg['survey']['standoff_m'])
    cov = Coverage(small_scn)
    medians, covered = [], []
    for vis in (None, 4.0, 2.0):
        w = optics.with_visibility(cfg['water'], vis) if vis else None
        f = rend.capture(cam, stream(3, 'test'), water=w)
        medians.append(float(np.median(f.cnr[f.on_hull])))
        covered.append(float(cov.quality(f)[f.on_hull].mean()))
        if vis:
            share = f.radiance['backscatter'][f.on_hull].sum() / sum(r[f.on_hull].sum() for r in f.radiance.values())
            assert share > 0.3                                  # haze, not the hull, fills the image
    assert medians[0] > medians[1] > medians[2]
    assert covered[0] > 0.9 and covered[2] == 0.0


def test_motion_blur_follows_the_camera_motion(small_scn, rend):
    cam = _face(rend, small_scn, 3.0, 2.0, standoff=1.5)
    still = rend.capture(cam, stream(4, 'test'))
    v = np.array([1.5, 0.0, 0.0])                                # along the hull, across the image
    moving = rend.capture(cam, stream(4, 'test'), motion=(v, np.zeros(3)))
    expect = rend.focal * 1.5 / 1.5 * moving.exposure_s       # f v / Z t, in pixels
    assert moving.blur_px == pytest.approx(expect, rel=1e-6) and still.blur_px == 0.0
    sharp = np.abs(np.diff(still.rgb.astype(float), axis=1)).mean()
    smeared = np.abs(np.diff(moving.rgb.astype(float), axis=1)).mean()
    assert smeared < sharp
