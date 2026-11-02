"""Underwater image formation (Jaffe-McGlamery single scattering) and the camera sensor.

Water, per colour channel: absorption a, scattering b, attenuation c = a + b, and a Henyey-Greenstein
phase function p(psi) with asymmetry g. The vehicle's lamps sit on the inspection camera's tilt unit.
Lamp k radiates I_k(theta) = I0 cos^m(theta) about its axis (zero behind it); m follows from the
half-power angle and I0 = P (m + 1) / (2 pi) gives the lamp its total power P.

For a camera ray of unit direction d meeting a Lambertian surface (albedo rho, unit normal n) at
range r:
  irradiance   E  = sum_k I_k(theta_k) max(0, -n.w_k) / D_k^2 * e^{-c D_k} (1 + eta (e^{b D_k} - 1))
                    + daylight E_d(z) [(1 + n_z) / 2 + R (1 - n_z) / 2]
               (w_k: unit vector lamp -> surface, D_k its length; eta: share of the light scattered on
               the way that stays in a broad beam, the scattering within optics.forward_cone_deg)
  direct       L_D = rho E / pi * e^{-c r}
  forward      L_F = in (S * K) + (1 - in) mean(S),  S = L_D (e^{b r} - 1): light scattered at small
               angles on the way to the camera. Each surface point's scattered power (the multiply
               scattered e^{-a r} - e^{-c r}) is spread by the single-scattering halo K of a point at
               the frame's median range; `in` is the halo's share inside the kernel, the rest is
               spread as a uniform veil.
  backscatter  L_B = int_0^r b p(psi_k(s)) sum_k I_k(theta_k(s)) / D_k(s)^2 e^{-c D_k(s)} e^{-c s} ds
                    + L_inf (1 - e^{-c r}),  L_inf = b p(90 deg) E_d(z) / c (daylight veiling light)
Sensor (lens of f-number N, square pixels of pitch p): photo-electrons per second
  e = (L_D + L_F + L_B) pi / (4 N^2) cos^v(alpha) p^2 QE lambda / (h c0)
with v = 4 for a thin lens (natural vignetting). Auto exposure sets the exposure time (up to the
motion-blur limit) and then the gain so the centre-weighted mean of the white-balanced image reaches
its target. The image is Poisson shot noise on the electrons, Gaussian read noise, gain, white
balance (per-channel gains that render a white surface under the lamps neutral) and quantization,
clipped at the full well.

Seeing a defect: a reflectance step dr on the reference background changes a pixel by dr S1 t
electrons (S1: direct signal per unit albedo). With the background's electrons (direct, forward and
backscatter), the read noise and the quantization step, the per-pixel contrast-to-noise ratio is
dr S1 t / sqrt(N_bg + sigma_read^2 + q^2 / 12). Over one coverage cell of (cell / gsd)^2 pixels it is
that times cell / gsd, which the referee compares with the Rose criterion.
"""
from __future__ import annotations

import math

import numpy as np

H_PLANCK = 6.62607015e-34
C_LIGHT = 2.99792458e8


# ------------------------------------------------------------------ water
def iops(water: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Absorption, scattering and attenuation coefficients per channel (1/m)."""
    a = np.asarray(water['absorption_per_m'], float)
    b = np.asarray(water['scattering_per_m'], float)
    return a, b, a + b


def visibility_m(water: dict) -> float:
    """Horizontal visibility of a black target, visibility_constant / c, middle channel."""
    return water['visibility_constant'] / float(iops(water)[2][1])


def with_visibility(water: dict, visibility: float) -> dict:
    """The same water with its particle scattering scaled to give this visibility (a turbidity event)."""
    a, b, _ = iops(water)
    k = max(0.0, (water['visibility_constant'] / visibility - a[1]) / b[1])
    return {**water, 'scattering_per_m': (b * k).tolist()}


def hg_phase(cos_psi, g: float):
    """Henyey-Greenstein phase function (1/sr), normalised over the sphere; psi is the scattering angle."""
    return (1 - g * g) / (4 * math.pi * (1 + g * g - 2 * g * np.asarray(cos_psi)) ** 1.5)


def hg_fraction_within(theta: float, g: float) -> float:
    """Share of the scattered power within angle theta of the forward direction."""
    return (1 - g * g) / (2 * g) * (1 / (1 - g) - 1 / math.sqrt(1 + g * g - 2 * g * math.cos(theta)))


def forward_eta(water: dict, cfg_optics: dict) -> float:
    return hg_fraction_within(math.radians(cfg_optics['forward_cone_deg']), water['phase_g'])


# ------------------------------------------------------------------ lamps
class Lamps:
    """Lamps with a cos^m beam; positions and unit axes in the world frame."""

    def __init__(self, positions: np.ndarray, axes_: np.ndarray, lamps_cfg: dict):
        self.pos = np.asarray(positions, float).reshape(-1, 3)
        self.axis = np.asarray(axes_, float).reshape(-1, 3)
        half = math.radians(lamps_cfg['half_power_angle_deg'])
        self.m = math.log(0.5) / math.log(math.cos(half))
        power = lamps_cfg['power_w'] * np.asarray(lamps_cfg['spectrum'], float)
        self.I0 = power * (self.m + 1) / (2 * math.pi)            # W/sr per channel on the axis

    @classmethod
    def mounted(cls, cam_pos: np.ndarray, cam_axes: np.ndarray, lamps_cfg: dict) -> 'Lamps':
        """Lamps on the tilt unit: offsets in the camera frame (x right, y up, z back), aimed along the view."""
        off = np.asarray(lamps_cfg['offsets_m'], float)
        pos = cam_pos + off @ cam_axes.T
        return cls(pos, np.repeat(-cam_axes[:, 2][None], len(off), 0), lamps_cfg)

    def toward(self, x: np.ndarray):
        """For points x (..., 3): intensity toward them (..., k, 3), unit lamp->x vectors (..., k, 3) and
        distances (..., k)."""
        v = x[..., None, :] - self.pos
        d = np.maximum(np.linalg.norm(v, axis=-1), 1e-9)
        u = v / d[..., None]
        cos_t = np.clip((u * self.axis).sum(-1), 0.0, 1.0)
        return self.I0 * (cos_t ** self.m)[..., None], u, d


# ------------------------------------------------------------------ light at surfaces and in the water
def daylight_irradiance(water: dict, depth_m) -> np.ndarray:
    """Downwelling daylight irradiance per channel at a depth (W/m^2), shape (..., 3)."""
    dl = water['daylight']
    return np.asarray(dl['surface_irradiance_w_m2'], float) * np.exp(
        -np.asarray(dl['diffuse_attenuation_per_m'], float) * np.maximum(np.asarray(depth_m, float), 0.0)[..., None])


def irradiance(points: np.ndarray, normals: np.ndarray, lamps: Lamps, water: dict, eta: float) -> np.ndarray:
    """Irradiance (W/m^2, per channel) on Lambertian surface points with unit normals, shape (..., 3)."""
    _, b, c = iops(water)
    I, u, D = lamps.toward(points)
    cos_i = np.clip(-(u * normals[..., None, :]).sum(-1), 0.0, 1.0)
    D3 = D[..., None]
    E = (I * cos_i[..., None] / D3 ** 2 * np.exp(-c * D3) * (1 + eta * np.expm1(b * D3))).sum(-2)
    nz = normals[..., 2]
    up = water['daylight']['upwelling_ratio']
    return E + daylight_irradiance(water, -points[..., 2]) * ((1 + nz) / 2 + up * (1 - nz) / 2)[..., None]


def path_grid(cfg_optics: dict) -> np.ndarray:
    """Ranges along a camera ray at which backscatter is accumulated (geometric spacing)."""
    o = cfg_optics
    return np.geomspace(o['path_start_m'], o['path_max_m'], o['backscatter_samples'])


def backscatter_cumulative(origin: np.ndarray, dirs: np.ndarray, lamps: Lamps, water: dict, s_grid: np.ndarray) -> np.ndarray:
    """Lamp backscatter radiance accumulated from the camera up to each range in s_grid, for unit ray
    directions dirs (M, 3): shape (M, S, 3). Trapezoid rule; the integrand is zero at the camera
    (the lamps sit beside it and point forward)."""
    _, b, c = iops(water)
    X = origin + s_grid[None, :, None] * dirs[:, None, :]                     # (M, S, 3)
    I, u, D = lamps.toward(X)                                                  # (M, S, k, 3), (M, S, k, 3), (M, S, k)
    cos_psi = -(u * dirs[:, None, None, :]).sum(-1)                            # incident direction vs back to the camera
    p = hg_phase(cos_psi, water['phase_g'])
    D3 = D[..., None]
    f = (b * p[..., None] * I / D3 ** 2 * np.exp(-c * D3)).sum(-2) * np.exp(-c * s_grid[None, :, None])
    s = np.concatenate([[0.0], s_grid])
    f = np.concatenate([np.zeros_like(f[:, :1]), f], axis=1)
    return np.cumsum((f[:, 1:] + f[:, :-1]) / 2 * np.diff(s)[None, :, None], axis=1)


def daylight_veil(water: dict, cam_depth: float, ranges: np.ndarray) -> np.ndarray:
    """Daylight veiling radiance along rays of the given ranges (inf: open water), shape (..., 3)."""
    _, b, c = iops(water)
    L_inf = b * float(hg_phase(0.0, water['phase_g'])) * daylight_irradiance(water, np.asarray(cam_depth)) / c
    return L_inf * -np.expm1(-c * np.asarray(ranges, float)[..., None])


def interpolate_along(table: np.ndarray, s_grid: np.ndarray, ranges: np.ndarray) -> np.ndarray:
    """Value of cumulative tables (M, S, 3) at per-ray ranges (M,), linear in range (clamped)."""
    r = np.clip(np.nan_to_num(ranges, posinf=s_grid[-1]), 0.0, s_grid[-1])
    k = np.clip(np.searchsorted(s_grid, r), 1, len(s_grid) - 1)
    s0, s1 = s_grid[k - 1], s_grid[k]
    w = np.clip((r - s0) / (s1 - s0), 0.0, 1.0)[:, None]
    idx = np.arange(len(r))
    lo = np.where((r < s_grid[0])[:, None], table[idx, 0] * (r / s_grid[0])[:, None], table[idx, k - 1])
    return np.where((r < s_grid[0])[:, None], lo, (1 - w) * table[idx, k - 1] + w * table[idx, k])


def backscatter_image(origin: np.ndarray, rays: np.ndarray, ranges: np.ndarray, lamps: Lamps, water: dict,
                      cfg_optics: dict) -> np.ndarray:
    """Lamp backscatter for a regular grid of unit rays (H, W, 3) with ranges (H, W): exact along each
    ray, computed on every `backscatter_grid_px`-th ray and interpolated bilinearly across the image."""
    H, W = ranges.shape
    step = cfg_optics['backscatter_grid_px']
    rows = np.unique(np.r_[0:H:step, H - 1])
    cols = np.unique(np.r_[0:W:step, W - 1])
    s_grid = path_grid(cfg_optics)
    table = backscatter_cumulative(origin, rays[np.ix_(rows, cols)].reshape(-1, 3), lamps, water, s_grid)
    table = table.reshape(len(rows), len(cols), len(s_grid), 3)
    fr = np.interp(np.arange(H), rows, np.arange(len(rows)))
    fc = np.interp(np.arange(W), cols, np.arange(len(cols)))
    i0 = np.minimum(fr.astype(int), len(rows) - 2)
    j0 = np.minimum(fc.astype(int), len(cols) - 2)
    wr, wc = (fr - i0)[:, None], (fc - j0)[None, :]
    r = np.clip(np.nan_to_num(ranges, posinf=s_grid[-1]), 0.0, s_grid[-1])
    k = np.clip(np.searchsorted(s_grid, r), 1, len(s_grid) - 1)
    ws = np.clip((r - s_grid[k - 1]) / (s_grid[k] - s_grid[k - 1]), 0.0, 1.0)[..., None]
    below = (r < s_grid[0])[..., None]
    out = np.zeros((H, W, 3))
    for di, dj, w in ((0, 0, (1 - wr) * (1 - wc)), (1, 0, wr * (1 - wc)), (0, 1, (1 - wr) * wc), (1, 1, wr * wc)):
        I, J = (i0 + di)[:, None], (j0 + dj)[None, :]
        v = (1 - ws) * table[I, J, k - 1] + ws * table[I, J, k]
        v = np.where(below, table[I, J, 0] * (r / s_grid[0])[..., None], v)
        out += v * w[..., None]
    return out


_KERNELS: dict = {}


def forward_kernel(r: float, g: float, focal_px: float, radius_px: int, samples: int = 160) -> tuple[np.ndarray, float]:
    """Shape (normalised to 1) of the single-scattering halo around a point at range r on a pixel grid
    of (2 radius + 1)^2, and the share of the forward-scattered light that falls inside it.

    Light leaving a surface point at angle theta from a pixel's line of sight is scattered into that
    line of sight at distance s from the camera through psi(theta, s); per unit solid angle of the
    camera the halo is K(theta) = int_0^r p(psi) r^2 / D^2 ds, which integrates to r over the sphere."""
    key = (round(r, 2), round(g, 4), round(focal_px, 3), radius_px)
    if key in _KERNELS:
        return _KERNELS[key]
    yy, xx = np.mgrid[-radius_px:radius_px + 1, -radius_px:radius_px + 1].astype(float)
    rho = np.hypot(xx, yy)
    rho[radius_px, radius_px] = 0.383                       # mean distance of a unit pixel's area from its centre
    theta_px = np.arctan(rho / focal_px)
    th = np.geomspace(theta_px.min(), theta_px.max(), 256)
    u = np.geomspace(1e-5 * r, r, samples)                  # distance of the scattering point from the surface
    s = r - u
    T, S = np.meshgrid(th, s, indexing='ij')
    D = np.sqrt((r * np.sin(T)) ** 2 + (r * np.cos(T) - S) ** 2)
    K = hg_phase((r * np.cos(T) - S) / D, g) * r * r / D ** 2
    du = np.diff(u)
    Kint = ((K[:, 1:] + K[:, :-1]) / 2 * du).sum(1)          # integral over u = r - s
    weights = np.interp(theta_px, th, Kint) * np.cos(theta_px) ** 3 / focal_px ** 2   # times the pixel's solid angle
    total = float(weights.sum())
    _KERNELS[key] = (weights / total, min(1.0, total / r))
    return _KERNELS[key]


def forward_scatter(direct: np.ndarray, ranges: np.ndarray, valid: np.ndarray, water: dict, focal_px: float,
                    cfg_optics: dict) -> np.ndarray:
    """Forward-scattered radiance for a direct-radiance image (H, W, 3). Each surface pixel at range r
    sends (e^{b r} - 1) times its direct radiance into the halo of a point at the frame's median
    surface range; the halo's share inside the kernel is spread by convolution, the rest uniformly over
    the frame. Energy is conserved, and the halo reaches pixels without geometry too."""
    if not valid.any():
        return np.zeros_like(direct)
    _, b, _ = iops(water)
    src = np.where(valid[..., None], direct * np.expm1(b * np.where(valid, ranges, 0.0)[..., None]), 0.0)
    R = cfg_optics['forward_kernel_radius_px']
    shape, inside = forward_kernel(float(np.median(ranges[valid])), water['phase_g'], focal_px, R)
    H, W, _ = direct.shape
    ph, pw = H + 2 * R, W + 2 * R
    kf = np.fft.rfft2(shape, s=(ph, pw))
    out = np.empty_like(direct)
    for ch in range(3):
        conv = np.fft.irfft2(np.fft.rfft2(src[..., ch], s=(ph, pw)) * kf, s=(ph, pw))[R:R + H, R:R + W]
        out[..., ch] = inside * conv + (1 - inside) * src[..., ch].mean()
    return np.maximum(out, 0.0)


# ------------------------------------------------------------------ sensor
def electrons_per_radiance(sensor: dict, cos_alpha: np.ndarray) -> np.ndarray:
    """Photo-electrons per second per unit scene radiance (W/m^2/sr) for each pixel and channel."""
    lam = np.asarray(sensor['wavelength_nm'], float) * 1e-9
    k = (math.pi / (4 * sensor['f_number'] ** 2) * (sensor['pixel_pitch_um'] * 1e-6) ** 2
         * np.asarray(sensor['quantum_efficiency'], float) * lam / (H_PLANCK * C_LIGHT))
    return np.asarray(cos_alpha, float)[..., None] ** sensor['vignetting_exponent'] * k


def white_balance(cfg_cameras: dict) -> np.ndarray:
    """Per-channel gains (green = 1) that render a white surface under the lamps neutral."""
    sensor, lamps = cfg_cameras['sensor'], cfg_cameras['lamps']
    resp = (np.asarray(lamps['spectrum'], float) * np.asarray(sensor['quantum_efficiency'], float)
            * np.asarray(sensor['wavelength_nm'], float))
    return resp[1] / resp


def dn_per_electron(sensor: dict, gain: float) -> float:
    return (2 ** sensor['bits'] - 1) / sensor['full_well_e'] * gain


def auto_exposure(rate: np.ndarray, sensor: dict, wb: np.ndarray) -> tuple[float, float]:
    """Exposure time and gain bringing the centre-weighted mean of the white-balanced signal
    (electrons/s times the channel gains, all channels) to the target."""
    ae = sensor['auto_exposure']
    H, W = rate.shape[:2]
    fh, fw = max(1, int(H * ae['metering_fraction'] / 2)), max(1, int(W * ae['metering_fraction'] / 2))
    meter = max(float((rate[H // 2 - fh:H // 2 + fh, W // 2 - fw:W // 2 + fw] * wb).mean()), 1e-9)
    want = ae['target'] * sensor['full_well_e']          # electrons at gain 1 for the target level
    t = float(np.clip(want / meter, *sensor['exposure_s']))
    gain = float(np.clip(want / (meter * t), 1.0, sensor['max_gain']))
    return t, gain


def expose(rate: np.ndarray, t: float, gain: float, rng: np.random.Generator, sensor: dict, wb: np.ndarray) -> np.ndarray:
    """Digital image: Poisson photo-electrons, full-well clipping, read noise, gain, white balance,
    quantization."""
    e = np.minimum(rng.poisson(rate * t).astype(np.float64), sensor['full_well_e'])
    e += rng.normal(0.0, sensor['read_noise_e'], e.shape)
    top = 2 ** sensor['bits'] - 1
    dn = np.clip(np.rint(e * dn_per_electron(sensor, gain) * wb), 0, top)
    return dn.astype(np.uint8 if sensor['bits'] <= 8 else np.uint16)


def motion_blur(img: np.ndarray, blur_px: np.ndarray) -> np.ndarray:
    """Average of the image shifted along the blur vector (pixels) over the exposure (edges clamp)."""
    n = int(math.ceil(float(np.hypot(*blur_px)))) + 1
    if n <= 1:
        return img
    H, W = img.shape[:2]
    out = np.zeros_like(img)
    yy, xx = np.mgrid[0:H, 0:W].astype(float)
    for f in np.linspace(-0.5, 0.5, n):
        x = np.clip(xx + f * blur_px[0], 0, W - 1)
        y = np.clip(yy + f * blur_px[1], 0, H - 1)
        x0, y0 = np.minimum(x.astype(int), W - 2), np.minimum(y.astype(int), H - 2)
        wx, wy = (x - x0)[..., None], (y - y0)[..., None]
        out += ((1 - wx) * (1 - wy) * img[y0, x0] + wx * (1 - wy) * img[y0, x0 + 1]
                + (1 - wx) * wy * img[y0 + 1, x0] + wx * wy * img[y0 + 1, x0 + 1])
    return out / n


def cell_cnr(s1_rate: np.ndarray, bg_rate: np.ndarray, t: float, gain: float, gsd_mm: np.ndarray, cfg: dict) -> np.ndarray:
    """Contrast-to-noise ratio of coverage.reflectance_step over one coverage cell (see module notes).
    s1_rate, bg_rate: electrons/s per channel for unit albedo direct signal and for the background."""
    cv, sensor = cfg['coverage'], cfg['cameras']['sensor']
    ch = cv['cnr_channel']
    q = 1.0 / (dn_per_electron(sensor, gain) * white_balance(cfg['cameras'])[ch])   # electrons per digital step
    signal = cv['reflectance_step'] * s1_rate[..., ch] * t
    noise = np.sqrt(bg_rate[..., ch] * t + sensor['read_noise_e'] ** 2 + q * q / 12)
    return signal / noise * (cv['cell_m'] * 1000.0 / np.maximum(gsd_mm, 1e-6))


def background_rate(s1_rate: np.ndarray, back_rate: np.ndarray, ranges: np.ndarray, water: dict, cfg: dict) -> np.ndarray:
    """Electrons/s of the reference background: its direct and forward-scattered light, plus backscatter."""
    _, b, _ = iops(water)
    rho = np.asarray(cfg['coverage']['reference_albedo'], float)
    return rho * s1_rate * np.exp(b * np.nan_to_num(ranges, posinf=0.0)[..., None]) + back_rate
