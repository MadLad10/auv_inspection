import numpy as np
import pytest

from auvinspect import config
from auvinspect.hull import HullGeometry


def hull():
    return HullGeometry.from_config(config.load())


def test_inverse_round_trips_every_region():
    h = hull()
    s = np.linspace(0, h.girth, 997)
    x = np.linspace(0, h.length, 997)
    x2, s2, d = h.inverse(h.point(x, s))
    assert np.allclose(x2, x) and np.allclose(s2, s, atol=1e-9) and np.allclose(d, 0, atol=1e-9)
    # points off the surface along the normal report that distance
    _, s3, d3 = h.inverse(h.point(x, s) + 0.7 * h.normal(s))
    assert np.allclose(d3, 0.7, atol=1e-9) and np.allclose(s3, s, atol=1e-9)


def test_normals_are_unit_and_continuous_at_the_joins():
    h = hull()
    for s in (h.side_depth, h.bilge_end):
        assert np.allclose(h.normal(s - 1e-9), h.normal(s + 1e-9), atol=1e-6)
    assert np.allclose(np.linalg.norm(h.normal(np.linspace(0, h.girth, 50)), axis=-1), 1)


def test_mesh_faces_point_outward_and_texture_coordinates_are_in_range():
    h = hull()
    v, uv, f = h.mesh(40, 30, flip_v=False)
    n = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    centre = v[f].mean(1)
    _, s, _ = h.inverse(centre)
    assert ((n * h.normal(s)).sum(1) > 0).all()
    assert uv.min() >= 0 and uv.max() <= 1


def test_raycast_meets_every_hull_region_at_the_standoff():
    h = hull()
    s = np.linspace(0.05, h.girth - 0.05, 301)
    x = np.full_like(s, h.length / 2)
    n = h.normal(s)
    t = h.raycast(h.point(x, s) + 1.3 * n, -n)
    assert np.allclose(t, 1.3, atol=1e-9)
    # past the hull's ends the ray misses, unless the section is extended
    off = h.point(np.full_like(s, -0.2), s) + 1.3 * n
    assert np.isinf(h.raycast(off, -n)).all()
    assert np.allclose(h.raycast(off, -n, bounded=False), 1.3, atol=1e-9)
    # rays pointing away from the hull miss
    assert np.isinf(h.raycast(h.point(x, s) + 1.3 * n, n)).all()


@pytest.mark.parametrize('override', [
    {'anodes': {'girth_m': 3.5}},                      # off the flat side
    {'hull': {'inspect_girth_m': [0.0, 99.0]}},        # beyond the hull
    {'survey': {'camera': 'periscope'}},
    {'survey': {'overlap_along': 1.0}},
    {'faults': [{'type': 'warp_drive', 't_start': 1.0}]},   # not a known fault type
    {'faults': [{'type': 'dvl_dropout', 't_start': 1.0, 'duration': 0.0}]},
    {'vehicle': {'speed_mps': 0.0}},
    {'onboard': {'claim_pixel_stride': 0}},
    {'water': {'phase_g': 1.0}},                       # not a phase function
    {'water': {'scattering_per_m': [0.4, -0.1, 0.4]}},
    {'cameras': {'lamps': {'half_power_angle_deg': 95.0}}},
    {'vehicle': {'model': {'mass_kg': 0.0}}},
    {'vehicle': {'integration_dt_s': 0.1}},            # longer than the control tick
    {'coverage': {'cnr_channel': 3}},
])
def test_invalid_configurations_are_rejected(override):
    with pytest.raises(config.ConfigError):
        config.load(overrides=override)
