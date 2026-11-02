"""Vehicle dynamics and sensors checked against analytic results."""
import math

import numpy as np
import pytest

from auvinspect import config
from auvinspect.dynamics import (Contact, RigidBody, VehicleModel, added_mass_ellipsoid, box_quadratic_damping,
                                 coriolis_times, quat_to_rpy, yaw_quat)
from auvinspect.hull import HullGeometry
from auvinspect.vehicle import Vehicle, seabed_z


def _cfg(**model):
    return config.load(overrides={'hull': {'length_m': 6.0}, 'vehicle': {'model': model}} if model else {'hull': {'length_m': 6.0}})


def _free_body(cfg, pos=(3.0, -5.0, -3.0), current=(0.0, 0.0, 0.0)):
    """A vehicle far from the hull and the seabed."""
    hull = HullGeometry.from_config(cfg)
    m = VehicleModel(cfg)
    return RigidBody(m, Contact(hull, seabed_z(hull, cfg), cfg['vehicle']['contact']), current, pos, yaw_quat(0.0)), m


def test_added_mass_matches_lamb():
    rho = 1.0
    sphere = added_mass_ellipsoid(1, 1, 1, rho) / (4 / 3 * math.pi)
    assert np.allclose(sphere, [0.5, 0.5, 0.5, 0, 0, 0], atol=1e-4)
    mf = 4 / 3 * math.pi * 2.0                                   # prolate spheroid, length/diameter 2
    a = added_mass_ellipsoid(2, 1, 1, rho)
    assert a[0] / mf == pytest.approx(0.210, abs=1e-3) and a[1] / mf == pytest.approx(0.704, abs=1e-3)
    assert a[3] == 0 and a[4] / (mf * (4 + 1) / 5) == pytest.approx(0.240, abs=1e-3)   # Lamb k' about the minor axis


def test_strip_theory_drag_of_a_box():
    L, W, H, rho, cd = 0.5, 0.3, 0.2, 1025.0, 1.0
    d = box_quadratic_damping((L, W, H), cd, rho, n=200)
    assert np.allclose(d[:3], [0.5 * rho * cd * W * H, 0.5 * rho * cd * L * H, 0.5 * rho * cd * L * W])
    assert d[5] == pytest.approx(rho * cd * H * (W ** 4 + L ** 4) / 64, rel=1e-3)


def test_coriolis_terms_do_no_work():
    m = VehicleModel(_cfg())
    rng = np.random.default_rng(0)
    for _ in range(20):
        nu = rng.normal(size=6)
        for M in (m.M_RB, m.M_A):
            assert abs(nu @ coriolis_times(M, nu)) < 1e-9


def test_kinetic_energy_is_conserved_without_damping_gravity_or_thrust():
    cfg = _cfg(damping_linear=[0.0] * 6, drag_coefficient=0.0, centre_of_gravity_m=[0.0, 0.0, 0.0], mass_kg=30.75)
    body, m = _free_body(cfg)
    m.weight = m.buoyancy                                         # neutral: no net hydrostatic force or moment
    body.nu = np.array([0.3, -0.2, 0.1, 0.8, -0.5, 1.2])
    M = m.M_RB + m.M_A
    e0 = 0.5 * body.nu @ M @ body.nu
    for _ in range(500):
        body.step(0.01, np.zeros(6))
    assert 0.5 * body.nu @ M @ body.nu == pytest.approx(e0, rel=1e-6)


def test_terminal_surge_speed_balances_thrust_and_drag():
    body, m = _free_body(_cfg())
    m.weight = m.buoyancy                                         # stay at depth
    thrust = np.array([5.0, 5.0, 0, 0, 0, 0])                    # both surge thrusters
    for _ in range(3000):
        body.step(0.01, thrust)
    u = body.nu[0]
    assert m.d_lin[0] * u + m.d_quad[0] * u * u == pytest.approx(10.0, rel=1e-3)


def test_added_mass_slows_the_first_acceleration():
    # centre of gravity on the body origin: a surge force then does not also pitch the vehicle
    cfg = _cfg(thruster_tau_s=1e-4, centre_of_gravity_m=[0.0, 0.0, 0.0])
    body, m = _free_body(cfg)
    m.weight = m.buoyancy
    body.step(0.001, np.array([10.0, 10.0, 0, 0, 0, 0]))
    accel = body.nu[0] / 0.001
    assert accel == pytest.approx(20.0 / (m.mass + m.added[0]), rel=0.02)
    assert m.added[0] > 0.05 * m.mass                             # not negligible for this vehicle


def test_buoyant_vehicle_rises_at_its_terminal_speed_and_rights_itself():
    body, m = _free_body(_cfg(), pos=(3.0, -5.0, -6.5))          # deep enough to stay submerged for the test
    roll0 = math.radians(20)
    body.q = np.array([math.cos(roll0 / 2), math.sin(roll0 / 2), 0.0, 0.0])
    for _ in range(1500):
        body.step(0.01, np.zeros(6))
    w = body.velocity_world()[2]
    assert m.d_lin[2] * w + m.d_quad[2] * w * w == pytest.approx(m.buoyancy - m.weight, rel=1e-3)
    assert abs(quat_to_rpy(body.q)[0]) < math.radians(0.5)


def test_vehicle_drifts_with_the_current():
    body, m = _free_body(_cfg(), current=(0.2, -0.1, 0.0))
    m.weight = m.buoyancy
    for _ in range(4000):
        body.step(0.01, np.zeros(6))
    assert np.allclose(body.velocity_world()[:2], [0.2, -0.1], atol=2e-3)


def test_contact_stops_the_vehicle_at_the_hull():
    cfg = _cfg()
    hull = HullGeometry.from_config(cfg)
    start = hull.point(3.0, 1.5) + 0.6 * hull.normal(1.5)
    v = Vehicle(hull, cfg, start, seed=0)
    push = np.array([0, 0, 0, 0, 30.0, 30.0])                     # both sway thrusters toward the hull (+y)
    for _ in range(200):
        v.step(push)
    assert v.contacts, 'the vehicle should have touched the hull'
    assert max(c['penetration_m'] for c in v.contacts) < 0.03
    assert v.clearance() > -0.03 and float(hull.inverse(v.body.pos)[2]) > 0


def test_sensors_at_rest_and_dvl_faults():
    cfg = config.load(overrides={'hull': {'length_m': 6.0}, 'faults': [{'type': 'dvl_dropout', 't_start': 1.0}],
                                 'sensors': {'imu': {'accel_noise': 0.0, 'gyro_noise': 0.0, 'gyro_bias_walk': 0.0}}})
    hull = HullGeometry.from_config(cfg)
    m = VehicleModel(cfg)
    v = Vehicle(hull, cfg, (3.0, -4.0, -3.0), seed=0)
    v.step(np.zeros(6))
    imu = v.readings()['imu']
    expected_z = m.g + (m.buoyancy - m.weight) / (m.mass + m.added[2])   # specific force of a body starting to rise
    assert np.allclose(imu['accel'], [0, 0, expected_z], atol=2e-3) and np.allclose(imu['gyro'], 0, atol=1e-9)
    dvl = v.readings()['dvl']
    assert dvl['valid'] and dvl['altitude'] == pytest.approx(-3.0 - 0.1 - seabed_z(hull, cfg), abs=0.05)
    for _ in range(25):
        v.step(np.zeros(6))
    assert not v.readings()['dvl']['valid']
    assert v.faults.log[0]['event'] == 'onset' and v.faults.log[0]['type'] == 'dvl_dropout'


def test_a_dead_thruster_produces_nothing():
    cfg = config.load(overrides={'hull': {'length_m': 6.0}, 'faults': [{'type': 'thruster_dead', 'params': {'name': 'surge_left'}}]})
    v = Vehicle(HullGeometry.from_config(cfg), cfg, (3.0, -4.0, -3.0), seed=0)
    for _ in range(20):
        v.step(np.array([20.0, 20.0, 0, 0, 0, 0]))
    assert v.body.thrust[0] == 0.0 and v.body.thrust[1] > 15.0
    assert v.body.nu[5] > 0.01        # the right thruster alone turns the vehicle to port
