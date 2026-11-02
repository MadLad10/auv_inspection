"""The vehicle's sensors, sampled from the simulated truth at their own rates, with noise and faults.

Readings use the record format auv_navigation_fusion consumes (imu, dvl, pressure, compass) plus the
inspection payload: the standoff sounder on the camera tilt unit, a water-quality sensor
(attenuation and scattering per colour channel), the tilt encoder, thruster telemetry and battery.

  imu        gyro = body angular rate + bias (random walk, plus an imu_bias fault) + noise; accelerometer =
             specific force at the body origin + noise; AHRS roll and pitch + noise
  dvl        four Janus beams at beam_angle_deg from the body's down axis, cast against the hull and the
             seabed; valid only when every beam returns within range. Velocity of the DVL head relative
             to the (static) surface it sees, with scale error and noise
  compass    yaw + noise (+ compass_disturbance)
  pressure   depth of the body origin + noise (pressure_stuck holds the last value)
  standoff   range to the hull along the inspection camera's optical axis + noise (standoff_dropout)
  water      attenuation c and scattering b per channel, with relative noise
"""
from __future__ import annotations

import math

import numpy as np

from .optics import iops
from .rng import stream


class Sensors:
    def __init__(self, hull, cfg: dict, seed: int, z_bed: float):
        self.hull, self.cfg, self.c = hull, cfg, cfg['sensors']
        self.z_bed = z_bed
        self.rng = {k: stream(seed, f'sensor_{k}') for k in self.c}
        self.readings: dict[str, dict] = {}
        self._next: dict[str, float] = {}
        self.gyro_bias = np.zeros(3)
        self._stuck_depth = None
        a = math.radians(self.c['dvl']['beam_angle_deg'])
        sa, ca = math.sin(a), math.cos(a)
        self.dvl_dirs = np.array([[sa, 0, -ca], [-sa, 0, -ca], [0, sa, -ca], [0, -sa, -ca]], float)

    def _due(self, name: str, t: float, force: bool) -> bool:
        if force or t + 1e-9 >= self._next.get(name, 0.0):
            self._next[name] = t + 1.0 / self.c[name]['rate_hz']
            return True
        return False

    def sample(self, veh, force: bool = False):
        t, c, body = veh.t, self.c, veh.body
        roll, pitch, yaw = veh.truth_rpy()
        if self._due('imu', t, force):
            ci, r = c['imu'], self.rng['imu']
            dt = 1.0 / ci['rate_hz']
            self.gyro_bias += r.normal(0, ci['gyro_bias_walk'] * math.sqrt(dt), 3)
            bias = self.gyro_bias.copy()
            f = veh.faults.get('imu_bias', t)
            if f:
                bias[2] += f.params.get('gyro_bias_z', 0.0)
            an = math.radians(ci['attitude_noise_deg'])
            self.readings['imu'] = {'t': t, 'gyro': (body.nu[3:] + bias + r.normal(0, ci['gyro_noise'], 3)).tolist(),
                                    'accel': (body.specific_force() + r.normal(0, ci['accel_noise'], 3)).tolist(),
                                    'roll': roll + float(r.normal(0, an)), 'pitch': pitch + float(r.normal(0, an))}
        if self._due('dvl', t, force):
            self.readings['dvl'] = self._dvl(veh, t)
        if self._due('compass', t, force):
            cc = c['compass']
            f = veh.faults.get('compass_disturbance', t)
            off = math.radians(f.params.get('offset_deg', 0.0)) if f else 0.0
            y = yaw + off + float(self.rng['compass'].normal(0, math.radians(cc['noise_deg'])))
            self.readings['compass'] = {'t': t, 'yaw': (y + math.pi) % (2 * math.pi) - math.pi}
        if self._due('pressure', t, force):
            depth = -float(body.pos[2]) + float(self.rng['pressure'].normal(0, c['pressure']['noise_m']))
            if veh.faults.get('pressure_stuck', t):
                if self._stuck_depth is None:
                    self._stuck_depth = self.readings.get('pressure', {}).get('depth', depth)
                depth = self._stuck_depth
            else:
                self._stuck_depth = None
            self.readings['pressure'] = {'t': t, 'depth': round(depth, 4)}
        if self._due('standoff', t, force):
            cs = c['standoff']
            cp, A = veh.camera_truth()
            rng_m = float(self.hull.raycast(cp, -A[:, 2]))
            ok = math.isfinite(rng_m) and rng_m <= cs['max_range_m'] and veh.faults.get('standoff_dropout', t) is None
            self.readings['standoff'] = {'t': t, 'valid': ok,
                                         'range': round(rng_m + float(self.rng['standoff'].normal(0, cs['noise_m'])), 4) if ok else None}
        if self._due('water_quality', t, force):
            _, b, cc = iops(veh.water())
            k = c['water_quality']['noise_frac']
            r = self.rng['water_quality']
            self.readings['water_quality'] = {'t': t, 'attenuation_per_m': (cc * (1 + r.normal(0, k, 3))).round(5).tolist(),
                                              'scattering_per_m': (b * (1 + r.normal(0, k, 3))).round(5).tolist()}
        if self._due('thrusters', t, force):
            self.readings['thrusters'] = {'t': t, 'names': veh.model.thruster_names,
                                          'cmd_n': [round(float(x), 3) for x in veh.thrust_cmd],
                                          'thrust_n': [round(float(x), 3) for x in body.thrust]}
        if self._due('battery', t, force):
            self.readings['battery'] = {'t': t, 'percent': round(veh.battery_percent(), 3)}
        self.readings['tilt'] = {'t': t, 'deg': veh.tilt_deg}

    def _dvl(self, veh, t: float) -> dict:
        cd, body = self.c['dvl'], veh.body
        R = body.R()
        head = body.pos + R @ np.asarray(cd['offset_m'], float)
        dirs = self.dvl_dirs @ R.T
        r_hull = self.hull.raycast(head, dirs)
        with np.errstate(divide='ignore', invalid='ignore'):
            r_bed = np.where(dirs[:, 2] < 0, (self.z_bed - head[2]) / dirs[:, 2], np.inf)
        ranges = np.minimum(r_hull, r_bed)
        ok = np.isfinite(ranges) & (ranges >= cd['min_range_m']) & (ranges <= cd['max_range_m'])
        valid = bool(ok.all()) and veh.faults.get('dvl_dropout', t) is None
        alt = float(np.mean(ranges[ok]) * math.cos(math.radians(cd['beam_angle_deg']))) if ok.any() else -1.0
        if not valid:
            return {'t': t, 'valid': False, 'vel': None, 'altitude': round(alt, 3)}
        v = body.nu[:3] + np.cross(body.nu[3:], np.asarray(cd['offset_m'], float))
        vel = v * (1 + cd['scale_error']) + self.rng['dvl'].normal(0, cd['vel_noise_mps'], 3)
        return {'t': t, 'valid': True, 'vel': vel.tolist(), 'altitude': round(alt, 3)}
