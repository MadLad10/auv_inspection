"""Image physics checked against closed forms and independent numerical references."""
import math

import numpy as np
import pytest

from auvinspect import config, optics
from auvinspect.rng import stream


@pytest.fixture(scope='module')
def cfg():
    return config.load()


def _sphere_integral(f, n=200000):
    """Integral over the sphere of f(cos psi) (axisymmetric)."""
    psi = np.linspace(0, math.pi, n)
    return np.trapz(f(np.cos(psi)) * 2 * math.pi * np.sin(psi), psi)


def test_phase_function_is_normalised_with_mean_cosine_g():
    for g in (0.0, 0.5, 0.924):
        assert _sphere_integral(lambda c: optics.hg_phase(c, g)) == pytest.approx(1.0, rel=1e-4)
        assert _sphere_integral(lambda c: c * optics.hg_phase(c, g)) == pytest.approx(g, abs=1e-4)
    g = 0.924
    theta = math.radians(10)
    psi = np.linspace(0, theta, 200000)
    within = np.trapz(optics.hg_phase(np.cos(psi), g) * 2 * math.pi * np.sin(psi), psi)
    assert optics.hg_fraction_within(theta, g) == pytest.approx(within, rel=1e-4)


def test_lamp_radiates_its_power_and_halves_at_its_half_power_angle(cfg):
    lc = cfg['cameras']['lamps']
    lamp = optics.Lamps(np.zeros((1, 3)), np.array([[0.0, 0.0, 1.0]]), lc)
    theta = np.linspace(0, math.pi / 2, 100000)
    dirs = np.stack([np.sin(theta), np.zeros_like(theta), np.cos(theta)], 1)
    I, _, _ = lamp.toward(dirs)
    power = np.trapz(I[:, 0, :] * 2 * math.pi * np.sin(theta)[:, None], theta, axis=0)
    assert np.allclose(power, lc['power_w'] * np.asarray(lc['spectrum']), rtol=1e-4)
    half = math.radians(lc['half_power_angle_deg'])
    Ih, _, _ = lamp.toward(np.array([[math.sin(half), 0.0, math.cos(half)]]))
    assert np.allclose(Ih[0, 0] / lamp.I0, 0.5)


def test_irradiance_on_a_facing_plane_matches_the_closed_form(cfg):
    w = cfg['water']
    lc = dict(cfg['cameras']['lamps'], offsets_m=[[0.0, 0.0, 0.0]])
    lamp = optics.Lamps.mounted(np.zeros(3), np.eye(3), lc)          # at the origin, aimed along -z
    d = 1.7
    eta = optics.forward_eta(w, cfg['optics'])
    E = optics.irradiance(np.array([[0.0, 0.0, -d]]), np.array([[0.0, 0.0, 1.0]]), lamp, w, eta)[0]
    _, b, c = optics.iops(w)
    assert np.allclose(E, lamp.I0 / d ** 2 * np.exp(-c * d) * (1 + eta * np.expm1(b * d)), rtol=1e-12)


def test_backscatter_quadrature_matches_a_dense_reference(cfg):
    w, o = cfg['water'], cfg['optics']
    lamps = optics.Lamps.mounted(np.zeros(3), np.eye(3), cfg['cameras']['lamps'])
    dirs = np.array([[0.0, 0.0, -1.0], [0.6, 0.3, -1.0], [-0.8, -0.6, -1.0]])
    dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
    r = np.array([1.5, 2.2, 40.0])
    got = optics.interpolate_along(optics.backscatter_cumulative(np.zeros(3), dirs, lamps, w, optics.path_grid(o)),
                                   optics.path_grid(o), r)
    _, b, c = optics.iops(w)
    for k in range(3):                                                # brute force: uniform midpoint rule
        s = (np.arange(400000) + 0.5) * min(r[k], o['path_max_m']) / 400000
        X = s[:, None] * dirs[k]
        I, u, D = lamps.toward(X)
        p = optics.hg_phase(-(u * dirs[k]).sum(-1), w['phase_g'])
        f = (b * p[..., None] * I / D[..., None] ** 2 * np.exp(-c * D[..., None])).sum(1) * np.exp(-np.outer(s, c))
        ref = f.sum(0) * (s[1] - s[0])
        assert np.allclose(got[k], ref, rtol=0.01)


def test_backscatter_needs_scattering_and_grows_with_it(cfg):
    o = cfg['optics']
    lamps = optics.Lamps.mounted(np.zeros(3), np.eye(3), cfg['cameras']['lamps'])
    dirs, r = np.array([[0.2, 0.1, -1.0]]) / math.sqrt(1.05), np.array([2.0])

    def back(scale):
        w = dict(cfg['water'], scattering_per_m=(np.asarray(cfg['water']['scattering_per_m']) * scale).tolist())
        return optics.interpolate_along(optics.backscatter_cumulative(np.zeros(3), dirs, lamps, w, optics.path_grid(o)),
                                        optics.path_grid(o), r)[0]
    assert np.allclose(back(0.0), 0.0)
    assert np.all(back(0.02) < back(0.04)) and np.allclose(back(0.04) / back(0.02), 2.0, rtol=0.05)


def test_forward_scatter_conserves_energy(cfg):
    w = cfg['water']
    H, W, r = 240, 320, 1.6
    direct = np.zeros((H, W, 3))
    direct[110:130, 150:170] = 1.0                                    # a bright patch away from the borders
    valid = np.ones((H, W), bool)
    fwd = optics.forward_scatter(direct, np.full((H, W), r), valid, w, 381.0, cfg['optics'])
    _, b, _ = optics.iops(w)
    assert np.allclose(fwd.sum((0, 1)), direct.sum((0, 1)) * np.expm1(b * r), rtol=0.02)
    shape, inside = optics.forward_kernel(r, w['phase_g'], 381.0, cfg['optics']['forward_kernel_radius_px'])
    R = cfg['optics']['forward_kernel_radius_px']
    assert shape.sum() == pytest.approx(1.0) and shape[R, R] == shape.max() and 0 < inside <= 1
    assert np.all(np.diff(shape[R, R:]) <= 0)                          # the halo falls off with angle


def test_sensor_counts_photons_with_shot_and_read_noise(cfg):
    sensor = dict(cfg['cameras']['sensor'], bits=12)
    wb = optics.white_balance(cfg['cameras'])
    rate = np.full((400, 400, 3), 2.0e5)
    t, gain = 0.004, 2.0
    img = optics.expose(rate, t, gain, stream(0, 'sensor_test'), sensor, wb).astype(float)
    k = optics.dn_per_electron(sensor, gain) * wb
    e = rate[0, 0, 0] * t
    assert np.allclose(img.mean((0, 1)), e * k, rtol=0.003)
    var_expected = (e + sensor['read_noise_e'] ** 2) * k ** 2 + 1 / 12
    assert np.allclose(img.var((0, 1)), var_expected, rtol=0.03)


def test_white_balance_renders_a_white_surface_under_the_lamps_neutral(cfg):
    cams = cfg['cameras']
    wb = optics.white_balance(cams)
    radiance = np.asarray(cams['lamps']['spectrum'])                 # a white surface reflects the lamp spectrum
    signal = radiance * optics.electrons_per_radiance(cams['sensor'], np.ones(1))[0] * wb
    assert np.allclose(signal / signal[1], 1.0) and wb[1] == 1.0


def test_auto_exposure_meets_its_target_or_its_limits(cfg):
    sensor = cfg['cameras']['sensor']
    wb = np.ones(3)
    want = sensor['auto_exposure']['target'] * sensor['full_well_e']
    t, gain = optics.auto_exposure(np.full((48, 64, 3), 3.0e6), sensor, wb)
    assert gain == pytest.approx(1.0) and t == pytest.approx(want / 3.0e6)
    t, gain = optics.auto_exposure(np.full((48, 64, 3), 1.0e5), sensor, wb)
    assert t == sensor['exposure_s'][1] and gain == pytest.approx(want / (1.0e5 * t))
    t, gain = optics.auto_exposure(np.full((48, 64, 3), 1.0), sensor, wb)
    assert t == sensor['exposure_s'][1] and gain == sensor['max_gain']


def test_cell_cnr_matches_its_definition(cfg):
    cv, sensor = cfg['coverage'], cfg['cameras']['sensor']
    s1, bg, t, gain, gsd = 4.0e5, 9.0e4, 0.008, 3.0, 4.0
    got = optics.cell_cnr(np.full((1, 3), s1), np.full((1, 3), bg), t, gain, np.array([gsd]), cfg)[0]
    q = 1 / (optics.dn_per_electron(sensor, gain) * optics.white_balance(cfg['cameras'])[cv['cnr_channel']])
    expect = cv['reflectance_step'] * s1 * t / math.sqrt(bg * t + sensor['read_noise_e'] ** 2 + q * q / 12) * cv['cell_m'] * 1000 / gsd
    assert got == pytest.approx(expect)


def test_visibility_maps_to_attenuation(cfg):
    w = cfg['water']
    assert optics.visibility_m(w) == pytest.approx(w['visibility_constant'] / (w['absorption_per_m'][1] + w['scattering_per_m'][1]))
    for v in (2.0, 4.0, 20.0):
        assert optics.visibility_m(optics.with_visibility(w, v)) == pytest.approx(v)
