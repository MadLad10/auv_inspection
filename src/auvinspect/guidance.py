"""Guidance and control: fly legs laid out on the hull, using only the vehicle's own estimate.

A leg is a polyline in hull coordinates (x along the ship, s down the girth) along which the
inspection camera stays on the hull normal at the survey standoff, so every point of every path keeps
that clearance, including transits round the bilge. The reference point advances along the leg at the
leg's speed but never more than `vehicle.lead_m` ahead of the vehicle's projection onto it. A lane
planned to end past a hull end carries an extension of `vehicle.edge_search_m`: if the standoff
sounder still sees the hull when the lane's planned end is reached (the vehicle has drifted along the
ship), the vehicle keeps going until the echo stops, which gives the navigation its along-ship fix.

A world-frame PID turns the position error into a force, plus a feed-forward of the vehicle model's
own drag at the reference velocity, so the integrator only meets constant disturbances (buoyancy,
current) and does not wind up while the vehicle cruises. PD loops hold yaw and pitch (roll is
passively stable and not actuated). The pseudo-inverse of the thruster geometry, read from the
vehicle model, allocates the wrench to the six thrusters; saturation scales the whole command so its
direction is kept. The camera tilt unit faces the hull normal at the reference girth and cancels the
measured roll.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

from .hull import HullGeometry
from .planner import lane_girths, lane_xs, capture_step

YAW_REF = 0.0    # the inspection camera looks to port and the hull is to port: hold heading along +x


@dataclass
class Leg:
    kind: str                         # 'lane' (frames are taken) or 'transit'
    phase: str                        # 'survey' or 'revisit_<round>'
    x: list[float]                    # hull coordinates of the control points
    s: list[float]
    speed: float
    lane: int | None = None
    capture_x: list[float] = field(default_factory=list)
    edge_search: bool = False         # the lane ends past a hull end: keep going until the echo stops


class Path:
    """Dense world polyline of the rig origin along a leg, its arc length, tilt and frame positions."""

    def __init__(self, leg: Leg, hull: HullGeometry, cfg: dict):
        self.leg = leg
        step = cfg['vehicle']['path_step_m']
        xs, ss = list(leg.x), list(leg.s)
        if leg.edge_search:                                   # extension beyond the planned end, along the lane
            xs.append(xs[-1] + math.copysign(cfg['vehicle']['edge_search_m'], xs[-1] - xs[-2]))
            ss.append(ss[-1])
        X, S = [xs[0]], [ss[0]]
        planned_end = 0
        for k, (x0, s0, x1, s1) in enumerate(zip(xs, ss, xs[1:], ss[1:])):
            n = max(1, math.ceil(math.hypot(x1 - x0, s1 - s0) / step))
            u = np.linspace(0, 1, n + 1)[1:]
            X.extend(x0 + u * (x1 - x0))
            S.extend(s0 + u * (s1 - s0))
            if k == len(leg.x) - 2:
                planned_end = len(X) - 1
        X, S = np.asarray(X), np.asarray(S)
        cam = cfg['cameras']['rig'][cfg['survey']['camera']]
        self.p = hull.point(X, S) + cfg['survey']['standoff_m'] * hull.normal(S) - np.asarray(cam['offset'], float)
        self.tilt = np.degrees(hull.turn_angle(S))
        self.arc = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(self.p, axis=0), axis=1))])
        self.length = float(self.arc[-1])
        self.nominal = float(self.arc[planned_end])          # arc length of the planned end
        self.X, self.S = X, S
        if leg.capture_x:
            order = np.argsort(X)
            self.capture_arc = np.interp(leg.capture_x, X[order], self.arc[order])
        else:
            self.capture_arc = np.zeros(0)

    def at(self, arc: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Position, unit tangent and tilt (deg) at an arc length."""
        i = int(np.clip(np.searchsorted(self.arc, arc) - 1, 0, len(self.arc) - 2))
        a0, a1 = self.arc[i], self.arc[i + 1]
        u = 0.0 if a1 <= a0 else float(np.clip((arc - a0) / (a1 - a0), 0, 1))
        d = self.p[i + 1] - self.p[i]
        n = float(np.linalg.norm(d))
        return self.p[i] + u * d, (d / n if n > 0 else np.zeros(3)), float(self.tilt[i] + u * (self.tilt[i + 1] - self.tilt[i]))

    def project(self, p: np.ndarray, hint: int) -> tuple[float, int]:
        """Arc length of the point of the path nearest p, searched forward from segment `hint`."""
        lo, hi = max(0, hint - 2), len(self.p) - 1
        a, b = self.p[lo:hi], self.p[lo + 1:hi + 1]
        d = b - a
        u = np.clip(((p - a) * d).sum(1) / np.maximum((d * d).sum(1), 1e-12), 0, 1)
        k = int(np.argmin(np.linalg.norm(a + u[:, None] * d - p, axis=1)))
        i = lo + k
        return float(self.arc[i] + u[k] * (self.arc[i + 1] - self.arc[i])), i


class LegFollower:
    def __init__(self, path: Path, cfg: dict):
        self.path, self.v = path, cfg['vehicle']
        self.ref = 0.0
        self.proj, self._i = 0.0, 0
        self.next_capture = 0
        self.limit = path.nominal                             # the reference stops here unless searching
        self.searching = False

    def step(self, p_est: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray, float]:
        """Advance the reference; return reference position, feed-forward velocity and tilt (deg)."""
        self.proj, self._i = self.path.project(p_est, self._i)
        new = min(self.ref + self.path.leg.speed * dt, self.proj + self.v['lead_m'], self.limit)
        new = max(new, self.ref)
        p, tangent, tilt = self.path.at(new)
        v = tangent * (new - self.ref) / dt
        self.ref = new
        return p, v, tilt

    def due_captures(self) -> list[int]:
        """Frames whose position the vehicle (by its own estimate) has reached."""
        out = []
        while self.next_capture < len(self.path.capture_arc) and self.proj >= self.path.capture_arc[self.next_capture]:
            out.append(self.next_capture)
            self.next_capture += 1
        return out

    def at_limit(self, p_est: np.ndarray) -> bool:
        """The reference has reached its limit, every frame is taken and the vehicle has arrived."""
        return (self.ref >= self.limit - 1e-9 and self.next_capture >= len(self.path.capture_arc)
                and float(np.linalg.norm(p_est - self.path.at(self.limit)[0])) <= self.v['arrive_tol_m'])

    def search(self):
        """Continue into the extension past the planned end."""
        self.searching, self.limit = True, self.path.length

    def nominal_duration(self) -> float:
        return self.path.length / self.path.leg.speed


def survey_legs(hull: HullGeometry, cfg: dict) -> list[Leg]:
    """The planned survey as legs: an approach, then back-and-forth lanes joined by girth transits."""
    sv, v = cfg['survey'], cfg['vehicle']
    xa, xb = -sv['end_margin_m'], hull.length + sv['end_margin_m']
    xs = lane_xs(xa, xb, capture_step(cfg))
    legs = []
    girths = [float(g) for g in lane_girths(hull, cfg)]
    legs.append(Leg('transit', 'survey', [xa - v['start_lead_m'], xa], [girths[0]] * 2, v['transit_speed_mps']))
    for k, s in enumerate(girths):
        x0, x1 = (xa, xb) if k % 2 == 0 else (xb, xa)
        if k:
            legs.append(Leg('transit', 'survey', [x0, x0], [girths[k - 1], s], v['transit_speed_mps']))
        legs.append(Leg('lane', 'survey', [x0, x1], [s, s], v['speed_mps'], lane=k,
                        capture_x=[float(x) for x in (xs if k % 2 == 0 else xs[::-1])],
                        edge_search=not 0.0 <= x1 <= hull.length))
    return legs


class Controller:
    def __init__(self, cfg: dict, model):
        """model: the vehicle's dynamics.VehicleModel (thruster geometry about the body origin, drag)."""
        self.c = cfg['vehicle']['control']
        self.names = list(model.thruster_names)
        self.alloc = np.linalg.pinv(model.B[[0, 1, 2, 4, 5]])   # wrench (Fx Fy Fz My Mz); roll is not actuated
        self.max_thrust = float(model.max_thrust)
        self.drag_lin = model.d_lin[:3] * self.c['drag_feedforward']
        self.drag_quad = model.d_quad[:3] * self.c['drag_feedforward']
        self.integral = np.zeros(3)

    def command(self, est: dict, p_ref: np.ndarray, v_ref: np.ndarray, dt: float) -> np.ndarray:
        """Thruster requests (N) from the estimated state (world position/velocity, yaw, yaw rate,
        roll, pitch, pitch rate)."""
        c = self.c
        e = p_ref - np.array([est['x'], est['y'], est['z']])
        ev = v_ref - np.array([est['vx'], est['vy'], est['vz']])
        lim = np.asarray(c['i_limit_n'], float)
        self.integral = np.clip(self.integral + np.asarray(c['ki']) * e * dt, -lim, lim)
        f_world = np.asarray(c['kp']) * e + np.asarray(c['kd']) * ev + self.integral
        cy, sy = math.cos(est['yaw']), math.sin(est['yaw'])
        to_body = np.array([[cy, sy, 0.0], [-sy, cy, 0.0], [0.0, 0.0, 1.0]])
        v_body = to_body @ v_ref
        f_body = to_body @ f_world + self.drag_lin * v_body + self.drag_quad * np.abs(v_body) * v_body
        yaw_err = (YAW_REF - est['yaw'] + math.pi) % (2 * math.pi) - math.pi
        m_z = c['kp_yaw'] * yaw_err - c['kd_yaw'] * est['r']
        m_y = -c['kp_pitch'] * est['pitch'] - c['kd_pitch'] * est['q']
        u = self.alloc @ np.array([*f_body, m_y, m_z])
        peak = float(np.abs(u).max())
        return u * (self.max_thrust / peak) if peak > self.max_thrust else u
