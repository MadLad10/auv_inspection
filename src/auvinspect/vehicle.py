"""The vehicle in the loop (simulator side).

Rigid-body dynamics in water (dynamics.py), sensors (sensors.py), faults (faults.py), the camera tilt
unit (rate-limited) and the battery, all in the scenario's own world: the hull and seabed the
renderer draws are the ones the vehicle can touch and its DVL sees.

Everything here is the simulator. Onboard software sees only `readings()`; the referee may also read
the truth methods.
"""
from __future__ import annotations

import math

import numpy as np

from .camera import axes
from .dynamics import Contact, RigidBody, VehicleModel, quat_to_rpy, yaw_quat
from .faults import FaultInjector
from .hull import HullGeometry
from .optics import with_visibility
from .sensors import Sensors


def seabed_z(hull: HullGeometry, cfg: dict) -> float:
    return -(hull.side_depth + hull.bilge_radius + cfg['water']['seabed_clearance_m'])


class Vehicle:
    def __init__(self, hull: HullGeometry, cfg: dict, start, seed: int):
        self.hull, self.cfg = hull, cfg
        v = cfg['vehicle']
        self.model = VehicleModel(cfg)
        self.z_bed = seabed_z(hull, cfg)
        self.body = RigidBody(self.model, Contact(hull, self.z_bed, v['contact']), v['current_mps'], start, yaw_quat(0.0))
        self.body.nu_dot, _ = self.body.accel(self.body.pos, self.body.q, self.body.nu, self.body.thrust)
        self.dt = float(v['control_dt_s'])
        self.substeps = max(1, round(self.dt / v['integration_dt_s']))
        self.faults = FaultInjector(cfg['faults'])
        self.thrust_cmd = np.zeros(len(self.model.thruster_names))
        cam = cfg['cameras']['rig'][cfg['survey']['camera']]
        self.look, self.offset = cam['look'], np.asarray(cam['offset'], float)
        self.tilt_deg = 0.0          # actual tilt of the inspection camera (view pitched up)
        self.tilt_cmd = 0.0          # commanded by the onboard software
        self.energy_wh = float(v['model']['battery_wh'])
        self._drops: set[int] = set()
        self.contacts: list[dict] = []
        self.t = 0.0
        self.sensors = Sensors(hull, cfg, seed, self.z_bed)
        self.sensors.sample(self, force=True)

    # ------------------------------------------------------------------ stepping
    def step(self, thrust_cmd: np.ndarray):
        """Apply thruster requests (N) for one control tick (the only actuation path)."""
        m = self.model
        self.thrust_cmd = np.clip(np.asarray(thrust_cmd, float), -m.max_thrust, m.max_thrust)
        health = np.ones(len(m.thruster_names))
        for f in self.faults.all('thruster_dead', self.t):
            if f.params.get('name') in m.thruster_names:
                health[m.thruster_names.index(f.params['name'])] = 0.0
        deepest = 0.0
        for _ in range(self.substeps):
            self.body.step(self.dt / self.substeps, self.thrust_cmd * health)
            deepest = max(deepest, self.body.penetration)
        self.t = round(self.t + self.dt, 9)
        self.faults.update(self.t)
        rate = self.cfg['vehicle']['tilt_rate_deg_s'] * self.dt
        self.tilt_deg += float(np.clip(self.tilt_cmd - self.tilt_deg, -rate, rate))
        self._battery()
        if deepest > 0:
            self.contacts.append({'t': round(self.t, 3), 'penetration_m': round(deepest, 4)})
        self.sensors.sample(self)

    def _battery(self):
        mdl = self.cfg['vehicle']['model']
        power = mdl['hotel_power_w'] + self.model.power_per_thrust15 * float((np.abs(self.body.thrust) ** 1.5).sum())
        self.energy_wh = max(0.0, self.energy_wh - power * self.dt / 3600.0)
        for i, f in enumerate(self.faults.faults):
            if f.type == 'battery_drop' and f.active(self.t) and i not in self._drops:
                self._drops.add(i)
                self.energy_wh = max(0.0, self.energy_wh - mdl['battery_wh'] * f.params.get('drop_pct', 0.0) / 100)

    def readings(self) -> dict:
        """What the onboard software receives."""
        return dict(self.sensors.readings)

    # ------------------------------------------------------------------ truth (simulator / referee)
    def truth_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Body origin (world) and attitude quaternion (w, x, y, z)."""
        return self.body.pos.copy(), self.body.q.copy()

    def truth_rpy(self) -> tuple[float, float, float]:
        return quat_to_rpy(self.body.q)

    def camera_truth(self) -> tuple[np.ndarray, np.ndarray]:
        """Inspection camera position and axes (columns, world) at the actual tilt."""
        R = self.body.R()
        return self.body.pos + R @ self.offset, R @ axes(self.look, math.radians(self.tilt_deg))

    def camera_motion(self) -> tuple[np.ndarray, np.ndarray]:
        """Velocity and angular velocity (world) of the inspection camera."""
        R = self.body.R()
        w = self.body.omega_world()
        return self.body.velocity_world() + np.cross(w, R @ self.offset), w

    def clearance(self) -> float:
        """Smallest distance from the vehicle's box surface to the hull or the seabed (negative: contact)."""
        pts = self.body.pos + self.model.contact_points @ self.body.R().T
        x, _, d = self.hull.inverse(pts)
        d = np.where((x >= 0) & (x <= self.hull.length), d, np.inf)
        return float(min(d.min(), (pts[:, 2] - self.z_bed).min()))

    def water(self) -> dict:
        """The water now: the scenario's, clouded by an active turbidity fault."""
        f = self.faults.get('turbidity', self.t)
        return with_visibility(self.cfg['water'], f.params['visibility_m']) if f else self.cfg['water']

    def camera_blackout(self) -> bool:
        return self.faults.get('camera_blackout', self.t) is not None

    def battery_percent(self) -> float:
        return 100.0 * self.energy_wh / self.cfg['vehicle']['model']['battery_wh']

    def close(self):
        pass
