"""Six-degree-of-freedom vehicle dynamics in water (Fossen's marine-craft equations).

Body frame at the centre of buoyancy CO: x forward, y port, z up. World frame z up, z = 0 at the
surface. nu = [v, w] is the linear and angular velocity in the body frame, and nu_r = nu - [R^T v_c, 0]
the velocity relative to the water (v_c: a uniform, constant current in the world frame).

  (M_RB + M_A) dnu/dt = tau_thrusters + tau_hydrostatic + tau_contact
                        - C_RB(nu) nu - C_A(nu_r) nu_r - D(nu_r) nu_r + M_A [-w x R^T v_c, 0]
  dp/dt = R v,  dq/dt = 1/2 q (x) (0, w)

  M_RB     rigid body about CO, centre of gravity at r_g (box inertia, shifted by parallel axes)
  M_A      added mass of the ellipsoid with the box's volume and proportions (Lamb's coefficients)
  C        Coriolis-centripetal matrices in the Sagatun-Fossen form; skew-symmetric, so they do no work
  D        linear damping (configured) plus quadratic drag from a drag coefficient on the box faces
           (strip theory: only the faces' advancing parts push water)
  hydrostatic  weight at r_g, buoyancy at CO, reduced as the box breaks the surface
  thrusters    first-order lag on the commanded thrust, scaled by thruster health
  contact  penalty springs and dampers at points on the box surface against the hull and the
           seabed, along the surface normal, with Coulomb friction
Integration is fourth-order Runge-Kutta with the thrust held over each step.
"""
from __future__ import annotations

import math

import numpy as np

from .hull import HullGeometry


def skew(a) -> np.ndarray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def quat_mul(a, b) -> np.ndarray:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([aw * bw - ax * bx - ay * by - az * bz, aw * bx + ax * bw + ay * bz - az * by,
                     aw * by - ax * bz + ay * bw + az * bx, aw * bz + ax * by - ay * bx + az * bw])


def quat_to_rpy(q) -> tuple[float, float, float]:
    w, x, y, z = q
    return (math.atan2(2 * (w * x + y * z), 1 - 2 * (x * x + y * y)),
            math.asin(max(-1.0, min(1.0, 2 * (w * y - z * x)))),
            math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z)))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.array([math.cos(yaw / 2), 0.0, 0.0, math.sin(yaw / 2)])


def lamb_coefficients(a: float, b: float, c: float, n: int = 6000) -> tuple[float, float, float]:
    """Lamb's alpha0, beta0, gamma0 for an ellipsoid with semi-axes a, b, c (they sum to 2)."""
    L2 = max(a, b, c) ** 2
    lam = np.concatenate([[0.0], np.geomspace(1e-9 * L2, 1e9 * L2, n)])
    D = np.sqrt((a * a + lam) * (b * b + lam) * (c * c + lam))
    out = []
    for e in (a, b, c):
        f = 1.0 / ((e * e + lam) * D)
        out.append(a * b * c * float(((f[1:] + f[:-1]) / 2 * np.diff(lam)).sum()))
    return tuple(out)


def added_mass_ellipsoid(a: float, b: float, c: float, rho: float) -> np.ndarray:
    """Added mass and inertia (surge, sway, heave, roll, pitch, yaw) of an ellipsoid in an ideal fluid."""
    al, be, ga = lamb_coefficients(a, b, c)
    mf = rho * 4.0 / 3.0 * math.pi * a * b * c

    def rot(p2, q2, kp, kq):          # Lamb: rotation in the plane of semi-axes p, q
        den = 2 * (p2 - q2) + (p2 + q2) * (kp - kq)
        return 0.0 if abs(p2 - q2) < 1e-12 or abs(den) < 1e-15 else mf / 5 * (p2 - q2) ** 2 * (kq - kp) / den
    return np.array([al / (2 - al) * mf, be / (2 - be) * mf, ga / (2 - ga) * mf,
                     rot(b * b, c * c, be, ga), rot(c * c, a * a, ga, al), rot(a * a, b * b, al, be)])


def box_quadratic_damping(size, cd: float, rho: float, n: int = 24) -> np.ndarray:
    """Quadratic drag coefficients (N per (m/s)^2, N m per (rad/s)^2) of a box in the six body axes:
    the sum of 1/2 rho Cd |v_n|^3 dA over the advancing parts of its faces, for unit velocity."""
    L, W, H = size
    half = np.array([L, W, H]) / 2
    pts, nrm, area = [], [], []
    for k in range(3):
        i, j = [a for a in range(3) if a != k]
        u = (np.arange(n) + 0.5) / n * 2 - 1
        U, V = np.meshgrid(u * half[i], u * half[j], indexing='ij')
        for sgn in (1.0, -1.0):
            p = np.zeros((n * n, 3))
            p[:, i], p[:, j], p[:, k] = U.ravel(), V.ravel(), sgn * half[k]
            nn = np.zeros(3)
            nn[k] = sgn
            pts.append(p)
            nrm.append(np.repeat(nn[None], n * n, 0))
            area.append(np.full(n * n, 4 * half[i] * half[j] / (n * n)))
    p, nn, dA = np.concatenate(pts), np.concatenate(nrm), np.concatenate(area)
    out = []
    for axis in range(6):
        e = np.zeros(3)
        e[axis % 3] = 1.0
        vn = (nn @ e) if axis < 3 else ((np.cross(e, p) * nn).sum(1))
        lever = np.ones(len(p)) if axis < 3 else np.abs(vn)          # moment arm of the face element about the axis
        adv = vn > 0
        out.append(0.5 * rho * cd * float((np.abs(vn[adv]) ** 2 * lever[adv] * dA[adv]).sum()))
    return np.array(out)


def coriolis_times(M: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """C(nu) nu for the Sagatun-Fossen parametrisation of a symmetric mass matrix M."""
    v, w = nu[:3], nu[3:]
    a = M[:3, :3] @ v + M[:3, 3:] @ w
    b = M[3:, :3] @ v + M[3:, 3:] @ w
    return np.concatenate([np.cross(w, a), np.cross(v, a) + np.cross(w, b)])


class VehicleModel:
    """Mass properties, hydrodynamics, thrusters and contact points of a box-shaped vehicle."""

    def __init__(self, cfg: dict):
        m, w = cfg['vehicle']['model'], cfg['water']
        self.rho, self.g = w['density_kg_m3'], w['gravity_m_s2']
        self.size = np.asarray(m['size_m'], float)
        L, W, H = self.size
        self.volume = float(L * W * H)
        self.mass = float(m['mass_kg'])
        self.r_g = np.asarray(m['centre_of_gravity_m'], float)
        Ig = np.diag([W * W + H * H, L * L + H * H, L * L + W * W]) * self.mass / 12
        S = skew(self.r_g)
        self.M_RB = np.block([[self.mass * np.eye(3), -self.mass * S], [self.mass * S, Ig - self.mass * S @ S]])
        semi = (6 / math.pi) ** (1 / 3) * self.size / 2          # ellipsoid of equal volume and proportions
        self.added = added_mass_ellipsoid(*semi, self.rho)
        self.M_A = np.diag(self.added)
        self.M_inv = np.linalg.inv(self.M_RB + self.M_A)
        self.d_lin = np.asarray(m['damping_linear'], float)
        self.d_quad = box_quadratic_damping(self.size, m['drag_coefficient'], self.rho)
        self.weight = self.mass * self.g
        self.buoyancy = self.rho * self.g * self.volume
        self.thruster_names = list(m['thrusters'])
        geo = np.array([m['thrusters'][n] for n in self.thruster_names], float)
        self.thruster_pos, self.thruster_dir = geo[:, :3], geo[:, 3:] / np.linalg.norm(geo[:, 3:], axis=1, keepdims=True)
        self.B = np.vstack([self.thruster_dir.T, np.cross(self.thruster_pos, self.thruster_dir).T])   # about CO
        self.max_thrust = float(m['max_thrust_n'])
        self.thrust_tau = float(m['thruster_tau_s'])
        # ideal actuator-disk power T^1.5 / sqrt(2 rho A), divided by the thruster efficiency
        area = math.pi * m['thruster_diameter_m'] ** 2 / 4
        self.power_per_thrust15 = 1.0 / (m['thruster_efficiency'] * math.sqrt(2 * self.rho * area))
        g = np.linspace(-1, 1, 3)
        corners = np.array([[x, y, z] for x in g for y in g for z in g if max(abs(x), abs(y), abs(z)) == 1.0])
        self.contact_points = corners * self.size / 2              # 26 points on the box surface
        self.radius = float(np.linalg.norm(self.size) / 2)

    def hydrostatic(self, R: np.ndarray, z: float) -> np.ndarray:
        H = self.size[2]
        frac = min(1.0, max(0.0, 0.5 - z / H))
        fw = R.T @ np.array([0.0, 0.0, -self.weight])
        fb = R.T @ np.array([0.0, 0.0, self.buoyancy * frac])
        return np.concatenate([fw + fb, np.cross(self.r_g, fw)])

    def damping(self, nu_r: np.ndarray) -> np.ndarray:
        return (self.d_lin + self.d_quad * np.abs(nu_r)) * nu_r


class Contact:
    """Penalty contact of the vehicle's surface points with the hull and the seabed."""

    def __init__(self, hull: HullGeometry, z_bed: float, cfg_contact: dict):
        self.hull, self.z_bed, self.c = hull, z_bed, cfg_contact

    def near(self, pos: np.ndarray, radius: float) -> bool:
        x, _, d = self.hull.inverse(pos)
        near_hull = -radius <= float(x) <= self.hull.length + radius and float(d) < radius + 0.05
        return near_hull or float(pos[2]) - radius < self.z_bed + 0.05

    def force(self, pos, R, v_body, w_body, pts_body) -> tuple[np.ndarray, float]:
        """Generalised body-frame force (6,) and the deepest penetration (m)."""
        pw = pos + pts_body @ R.T
        vw = (v_body + np.cross(w_body, pts_body)) @ R.T
        x, s, d = self.hull.inverse(pw)
        pen_h = np.where((x >= 0) & (x <= self.hull.length), -d, -1.0)
        pen_b = self.z_bed - pw[:, 2]
        F = np.zeros_like(pw)
        deepest = 0.0
        for pen, n in ((pen_h, self.hull.normal(s)), (pen_b, np.tile([0.0, 0.0, 1.0], (len(pw), 1)))):
            hit = pen > 0
            if not hit.any():
                continue
            deepest = max(deepest, float(pen[hit].max()))
            vn = (vw[hit] * n[hit]).sum(1)
            fn = np.maximum(self.c['stiffness_n_per_m'] * pen[hit] - self.c['damping_n_per_mps'] * vn, 0.0)
            vt = vw[hit] - vn[:, None] * n[hit]
            ft = -self.c['friction'] * fn[:, None] * vt / (np.linalg.norm(vt, axis=1, keepdims=True) + self.c['slip_speed_mps'])
            F[hit] += fn[:, None] * n[hit] + ft
        fb = F @ R
        return np.concatenate([fb.sum(0), np.cross(pts_body, fb).sum(0)]), deepest


class RigidBody:
    """State (position, attitude quaternion, body velocity, thrust) and its integration."""

    def __init__(self, model: VehicleModel, contact: Contact, current_world, pos, quat):
        self.m, self.contact = model, contact
        self.current = np.asarray(current_world, float)
        self.pos = np.asarray(pos, float).copy()
        self.q = np.asarray(quat, float).copy()
        self.nu = np.zeros(6)
        self.thrust = np.zeros(len(model.thruster_names))
        self.penetration = 0.0
        self.nu_dot = np.zeros(6)

    def accel(self, pos, q, nu, thrust) -> tuple[np.ndarray, float]:
        m = self.m
        R = quat_to_mat(q)
        v, w = nu[:3], nu[3:]
        vc = R.T @ self.current
        nu_r = nu - np.concatenate([vc, np.zeros(3)])
        tau = m.B @ thrust + m.hydrostatic(R, float(pos[2]))
        pen = 0.0
        if self.contact.near(pos, m.radius):
            fc, pen = self.contact.force(pos, R, v, w, m.contact_points)
            tau = tau + fc
        rhs = (tau - coriolis_times(m.M_RB, nu) - coriolis_times(m.M_A, nu_r) - m.damping(nu_r)
               + m.M_A @ np.concatenate([-np.cross(w, vc), np.zeros(3)]))
        return m.M_inv @ rhs, pen

    def _deriv(self, pos, q, nu, thrust):
        nd, pen = self.accel(pos, q, nu, thrust)
        return quat_to_mat(q) @ nu[:3], 0.5 * quat_mul(q, np.concatenate([[0.0], nu[3:]])), nd, pen

    def step(self, dt: float, thrust_cmd: np.ndarray):
        """Advance dt: thrust lag (exact), then RK4 on position, attitude and velocity."""
        self.thrust += (thrust_cmd - self.thrust) * (1.0 - math.exp(-dt / self.m.thrust_tau))
        p0, q0, n0 = self.pos, self.q, self.nu
        k1 = self._deriv(p0, q0, n0, self.thrust)
        k2 = self._deriv(p0 + dt / 2 * k1[0], q0 + dt / 2 * k1[1], n0 + dt / 2 * k1[2], self.thrust)
        k3 = self._deriv(p0 + dt / 2 * k2[0], q0 + dt / 2 * k2[1], n0 + dt / 2 * k2[2], self.thrust)
        k4 = self._deriv(p0 + dt * k3[0], q0 + dt * k3[1], n0 + dt * k3[2], self.thrust)
        self.pos = p0 + dt / 6 * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        q = q0 + dt / 6 * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        self.q = q / np.linalg.norm(q)
        self.nu = n0 + dt / 6 * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        self.nu_dot, self.penetration = self.accel(self.pos, self.q, self.nu, self.thrust)

    def R(self) -> np.ndarray:
        return quat_to_mat(self.q)

    def velocity_world(self) -> np.ndarray:
        return self.R() @ self.nu[:3]

    def omega_world(self) -> np.ndarray:
        return self.R() @ self.nu[3:]

    def specific_force(self) -> np.ndarray:
        """What an accelerometer at CO measures (body frame): acceleration minus gravity."""
        v, w = self.nu[:3], self.nu[3:]
        return self.nu_dot[:3] + np.cross(w, v) + self.R().T @ np.array([0.0, 0.0, self.m.g])
