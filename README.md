# AUV Hull Inspection (simulation)

Underwater ship-hull inspection software built and tested entirely in simulation. The world has
exact ground truth: every defect's position, outline and area is known, and every rendered pixel
knows which defect it shows. The pipeline can therefore be scored by an independent referee, not
by its own claims.

**Status:** milestone 2 of 6 (vehicle in the loop), on a physically based environment: Fossen
6-DOF vehicle dynamics with added mass and contact, Jaffe-McGlamery underwater image formation with
a photon-counting camera, and a contrast-to-noise coverage criterion.

## What it does now

- **Seeded hull scenarios.** A hull section (flat side, bilge turn, flat bottom) with painted plates
  and staggered weld seams, a slime film and a sea-chest grating as a decoy. Defects: corrosion,
  coating damage, biofouling with barnacles, cracks, and missing-anode footprints. Anodes are 3-D
  geometry, either intact or depleted. All sizes, counts and colours come from
  [`config/default.yaml`](src/auvinspect/config/default.yaml).
- **Exact ground truth.** Defects live in a label raster on the hull's analytic surface coordinates
  (x along the ship, s down the girth). A rendered pixel's label comes from back-projecting its depth
  onto that surface, so labels do not depend on the render mesh.
- **Physically based underwater images** ([`optics.py`](src/auvinspect/optics.py)). MuJoCo only
  rasterises geometry: depth, segmentation and unlit albedo. Two lamps on the camera's tilt unit
  light the hull through water described by its absorption, scattering and phase function. The
  image is direct light, forward-scattered light and backscatter, recorded by a sensor that counts
  photons, with auto exposure and white balance (see [Physics](#physics)).
- **Survey planning against the referee's limits.** Lanes run back and forth at a fixed standoff;
  the inspection camera tilts to face the local hull normal. Each candidate lane's covered strip is
  computed on the curved hull with the same physics the referee uses, including midway between
  frames, and lanes are placed greedily with the requested overlap and a girth margin.
- **Coverage referee.** A 2 cm hull cell is covered when an inspection-camera pixel lands in it
  within the range, incidence and footprint limits and with a contrast-to-noise ratio that makes a
  faint defect detectable over the cell. A defect or anode counts as inspected when at least 80% of
  its area is covered.

With the default configuration (water visibility 10 m), seed 7 has 7 anodes, 8 corrosion patches,
5 coating-damage patches, 4 fouling patches and 3 cracks. The survey plans 4 lanes (96 frames),
covers 100% of the inspection band and inspects every item.

### Vehicle in the loop

- **Dynamics** ([`dynamics.py`](src/auvinspect/dynamics.py)). The vehicle, a 0.5 x 0.3 x 0.2 m box
  with six thrusters, follows Fossen's marine-craft equations in the scenario's own world. The hull
  and seabed the camera sees are the ones it can touch and its DVL ranges against.
- **Navigation from sensors only.** [`auv_navigation_fusion`](../auv_navigation_fusion)'s EKF fuses
  the simulated IMU, DVL, depth and compass. Two hull-relative updates are added. A standoff sounder
  on the camera tilt unit measures the range to the hull, which pins the distance from the hull and,
  with depth, the girth position. Lanes run past both hull ends, so the sounder's echo starts or
  stops where its beam crosses a hull end; each crossing is an along-ship position fix. If the echo
  is still there at a lane's planned end, the vehicle keeps going until it stops (edge search).
- **Guidance and control.** Paths are laid out on the hull at the survey standoff, so transits round
  the bilge keep that clearance too. A world-frame PID with drag feed-forward drives the vehicle; PD
  loops hold yaw and pitch. Thrust allocation is a pseudo-inverse of the thruster geometry. The tilt
  unit (rate-limited) faces the hull normal and cancels the measured roll. Frames are triggered when
  the vehicle, by its own estimate, reaches each planned frame position.
- **Faults** ([`faults.py`](src/auvinspect/faults.py)): camera blackout, turbidity, DVL dropout,
  IMU bias, compass disturbance, stuck depth sensor, dead thruster, standoff-sounder dropout and
  battery loss, scheduled with onset and clear times and logged as ground truth.
- **Claimed coverage against the referee.** The vehicle judges each frame with the same image
  physics, from its estimated pose, the water its water-quality sensor measures, and the exposure
  and gain the camera used. A noiseless frame means a dead camera. Frames taken while the position
  uncertainty along the ship or the girth is above a limit count as seen but unverified. The referee
  compares this claim with what the camera truly saw: a cell claimed but not seen is a false claim.
- **Replanning.** After the survey, unclaimed cells are clustered into holes. Each hole gets revisit
  lanes placed with the planner's own strip rules, for up to two rounds. Revisits are skipped while
  the vehicle's own position uncertainty is too large to claim anything.

On the full seed-7 hull the mission flies 96 frames in 353 s of simulated time (about three minutes
of wall-clock time). It covers 100% of the band, claims exactly that with no false claims, never
touches the hull and inspects all 27 items. The largest along-ship camera position error is 10 cm.

Fault cases on the 6 m test hull (seed 3, 19 items; blackout and turbidity from 100 to 120 s, inside
the last lane; DVL dropout from 60 to 120 s; simulation results):

| Run | After survey: true / claimed | After revisits: true / claimed | False claims | Largest along-ship error | Items inspected |
|---|---|---|---|---|---|
| Nominal | 100% / 100% | (none needed) | 0 | 0.05 m | 19 / 19 |
| Camera blackout, 20 s | 93.0% / 92.9% | 100% / 100% | 0 | 0.05 m | 19 / 19 |
| Turbidity (2 m visibility), 20 s | 93.0% / 92.9% | 100% / 100% | 0 | 0.05 m | 19 / 19 |
| DVL dropout, 60 s | 100% / 84.5% | 100% / 100% | 0 | 0.68 m | 19 / 19 |
| DVL dropout, 60 s, no hull-end fixes | 100% / 73.7% | revisits skipped (26% unverified) | 0 | 1.37 m | 19 / 19 |

In 2 m visibility the frames are milky haze: backscatter and forward scatter swamp the hull's
contrast, so no pixel reaches the contrast-to-noise limit. The vehicle measures the water and
agrees with the referee. During the DVL dropout the along-ship error grows by dead reckoning
(attitude noise leaks gravity into the accelerometer integration) until the next hull end;
without hull-end fixes the vehicle cannot verify what it saw and reports it as unverified.

## Physics

### Water and light ([`optics.py`](src/auvinspect/optics.py))

Water is described per colour channel by its absorption a and scattering b (attenuation c = a + b)
and a Henyey-Greenstein phase function with asymmetry g = 0.924 (backscatter ratio about 0.018,
Petzold-like). Horizontal visibility is 4.8 / c (Zaneveld and Pegau 2003); a turbidity event scales
the particle scattering to a requested visibility. Each lamp radiates I(θ) = I0 cos^m θ about its
axis, with m from the half-power angle and I0 = P (m + 1) / 2π so its total power is P.

For a pixel seeing a Lambertian surface (albedo ρ, normal n) at range r, following Jaffe (1990) and
McGlamery (1980):

| Component | Model |
|---|---|
| Irradiance | Σ lamps I(θ) cos(i) / D² · e^(-cD) (1 + η (e^(bD) - 1)), plus optional daylight E_d(z) with diffuse attenuation; η is the scattered light that stays in a broad beam (HG power within 10°) |
| Direct | ρ E / π · e^(-cr) |
| Forward scatter | each surface pixel sends (e^(br) - 1) times its direct radiance into a single-scattering halo, computed from the phase function and the camera-surface geometry; the halo's share inside a 48-pixel kernel is convolved, the rest spread as a veil; energy is conserved |
| Backscatter | ∫ b p(ψ) Σ I(θ) / D² e^(-cD) e^(-cs) ds along the camera ray (single scattering from the lamps), integrated on every 16th ray (error below 0.1%), plus optional daylight veiling light |

### Camera ([`optics.py`](src/auvinspect/optics.py), [`camera.py`](src/auvinspect/camera.py))

Photo-electrons per second are L · π / (4N²) · cos⁴α · p² · QE · λ / (h c), for f-number N, pixel
pitch p, quantum efficiency QE and band centre λ. Auto exposure sets the exposure time (up to 8 ms,
which keeps motion blur under a pixel at survey speed) and then the gain so the centre-weighted mean
reaches its target. The image is Poisson shot noise, read noise, gain, white balance for the lamps
and 8-bit quantization, clipped at the full well; camera motion during the exposure blurs it.

The footprint of a pixel on the surface is exact: the larger singular value of the Jacobian of the
image-to-surface map, (Z / f) (e - q (n·e) / (n·q)) for the image axes e. A surface parallel to the
image has the same footprint, Z / f, in every pixel.

### Coverage criterion ([`coverage.py`](src/auvinspect/coverage.py), [`predict.py`](src/auvinspect/predict.py))

A pixel counts when it is within range, incidence and footprint limits, and when a faint defect (an
albedo step of 0.05 on hull paint) would reach a contrast-to-noise ratio of 5 over one 2 cm cell
(the Rose criterion). The signal is the unscattered direct light; the noise is the shot noise of
the paint's direct, forward-scattered and backscattered light, read noise and the quantization
step. The referee evaluates this per pixel from the render; the planner and the vehicle evaluate the
same function on the analytic hull from the pose, water and exposure they know.

### Vehicle ([`dynamics.py`](src/auvinspect/dynamics.py), [`sensors.py`](src/auvinspect/sensors.py))

(M_RB + M_A) ν̇ + C_RB(ν)ν + C_A(ν_r)ν_r + D(ν_r)ν_r + g(η) = τ_thrusters + τ_contact (Fossen 2011),
with ν_r the velocity relative to a uniform current and the current's frame rotation term included.

| Term | Model |
|---|---|
| M_RB | box of the configured mass, centre of gravity 4 cm below the centre of buoyancy |
| M_A | added mass and inertia of the ellipsoid with the box's volume and proportions (Lamb's coefficients) |
| C_RB, C_A | Sagatun-Fossen parametrisation (skew-symmetric, no work) |
| D | configured linear damping plus quadratic drag from a drag coefficient on the box faces (strip theory) |
| g(η) | weight at the centre of gravity, buoyancy at the centre of buoyancy (reduced at the surface); trimmed 3 N positive in seawater |
| Thrusters | six, first-order lag and saturation; power from actuator-disk theory |
| Contact | spring-damper and Coulomb friction at 26 points of the box against the hull and the seabed |

Integration is fourth-order Runge-Kutta at 100 Hz; control and sensors run at 20 Hz. Sensors: IMU
(specific force, gyro bias random walk, AHRS attitude noise), a four-beam Janus DVL whose beams are
cast against the hull and the seabed, compass, pressure depth, the standoff sounder, a water-quality
sensor (attenuation and scattering), the tilt encoder, thruster telemetry and battery.

## Quick start

```bash
~/.local/bin/uv venv .venv --python /usr/bin/python3.10
~/.local/bin/uv pip install --python .venv/bin/python -e ".[dev]"
export MUJOCO_GL=egl
.venv/bin/auvinspect generate --seed 7 --out runs/s7
.venv/bin/auvinspect survey   --scenario runs/s7 --out runs/s7/survey
.venv/bin/auvinspect mission  --scenario runs/s7 --out runs/s7/mission
.venv/bin/auvinspect preview  --scenario runs/s7 --texture --out runs/s7/texture.png
.venv/bin/auvinspect preview  --scenario runs/s7 --x 10 --girth 3.9 --out runs/s7/rig.png
```

`--no-frames` scores coverage without writing images. `--config run.yaml` overrides run settings
(cameras, survey, coverage, vehicle, navigation, faults, ...) without changing the scenario, for
example a fault schedule:

```yaml
faults:
  - {type: camera_blackout, t_start: 200.0, duration: 20.0}
  - {type: dvl_dropout, t_start: 60.0, duration: 60.0}
  - {type: turbidity, t_start: 300.0, duration: 30.0, params: {visibility_m: 2.0}}
```

`mission --seed` changes the sensor noise and initial fix without changing the world.

The sibling package `auv_navigation_fusion` is installed from its built wheel in
`../auv_navigation_fusion/dist` (see `[tool.uv.sources]` in `pyproject.toml`), never by name from
an index.

## Design

- **World and run are separate.** The world sections (hull, appearance, defects, anodes, water)
  are saved with a scenario and are part of its SHA-256 digest (schema version 2). Run settings
  come from the run's configuration, so different missions can fly over the same frozen world.
  Loading checks the digest.
- **Deterministic.** Every random draw uses a named stream (`auvinspect.rng.stream(seed, name)`).
  The same seed and configuration give the same digest, images and trajectories.
- **No magic numbers.** Physical, sensor and sampling constants live in the YAML and are validated
  on load.
- **Simulator, onboard, referee.** `vehicle.py`, `dynamics.py`, `sensors.py` and `render.py` are the
  simulator. `onboard.py` and `guidance.py` are the vehicle's software: they read sensor readings,
  their own commands and the hull drawings, never the scenario's items, water or the simulator's
  truth. `coverage.py` is the referee. The mission reports where the vehicle's claim and the
  referee disagree.
- **Only the onboard controller actuates.** `Vehicle.step(thrust)` is the only path to the thrusters.

## Outputs

| File | Contents |
|---|---|
| `scenario.json` | Seed, hull, every item (class, area, bounding box, centroid, anode status), world config, digest |
| `texture.png`, `labels.npz` | Hull albedo and the per-pixel item label raster |
| `*/frames/*.png` | Camera images |
| `*/gt/*.npz` | Per-pixel label, depth, pixel footprint and contrast-to-noise ratio |
| `survey/frames.jsonl` | Pose, camera pose, exposure and gain, newly covered cells, visible items with pixel counts |
| `*/coverage.json`, `coverage.png` | Coverage fraction, per-item inspection verdict, coverage map |
| `mission/frames.jsonl` | Per frame: true camera pose, phase and leg, exposure, gain and blur, active faults, the onboard verdict (claimed, uncertain, black, below limits) with its estimated camera pose and uncertainty, camera position error along the ship, girth and normal |
| `mission/coverage.json` | Referee coverage and items, claimed against true coverage, rounds and legs, navigation error, EKF summary, minimum clearance, contacts, fault log |
| `mission/claimed.png`, `claim_check.png` | Claimed coverage; agreement map (green both, red claimed but not seen, blue seen but not claimed) |
| `mission/trajectory.npz` | Per control tick: true pose, estimate, reference, thrust, position uncertainty, camera tilt, clearance |

## Tests

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

`env -u PYTHONPATH` stops pytest loading ROS 2 plugins from a sourced ROS environment. The tests check:
- optics against closed forms and independent numerics: phase function normalisation and mean
  cosine, lamp power and beam, irradiance on a facing plane, backscatter against a 400,000-point
  quadrature, forward-scatter energy conservation, photon and read noise statistics, white balance,
  auto exposure limits, the contrast-to-noise definition, visibility;
- dynamics against analytic results: Lamb added mass (sphere, 2:1 spheroid), strip-theory drag,
  Coriolis terms doing no work, kinetic energy conservation, terminal surge speed, the first
  acceleration with added mass, buoyant rise speed and self-righting, drift with a current, contact
  stopping the vehicle at the hull, sensors at rest, DVL dropout, a dead thruster;
- rendering: labels and albedo aligned with the texture, anodes labelled from geometry, the exact
  pixel footprint against finite differences on the hull, the analytic ray cast against rendered
  range, exposure to the target, turbid water washing out the contrast until nothing is covered,
  motion blur, blackout frames, determinism;
- planning and the referee: full coverage of a planned survey, a plan with a missing lane caught,
  an impossible standoff refused, configuration validation, scenario determinism and exact areas;
- missions: nominal (full coverage, no false claims, no contact, hull-end fixes on every lane);
  blackout and turbidity holes found and revisited; DVL dropout with and without hull-end fixes,
  never over-claimed; determinism.

## Roadmap

1. ~~Scenarios, rendering, planning, coverage referee~~
2. ~~Vehicle in the loop: dynamics, navigation drift (`auv_navigation_fusion`), camera faults;
   replanning when coverage holes appear.~~
3. Perception: pretrained models (depth, segmentation, features) and a fine-tuned defect detector,
   scored against the ground truth per class and per size.
4. Defect map and report: defects placed on the hull with sizes and evidence frames; a second
   survey with grown defects for change detection; inspector review.
5. Production layer: simulation tests in CI that gate merges on coverage and detection metrics,
   containers, model registry, observability, fault injection.
6. Fleet: several vehicles sharing a structure under acoustic communication limits.

## Limitations

- Simulation only. Numbers describe this simulator and must not be read as real-world performance.
- Defect appearance is procedural until real defect imagery (for example a public hull-inspection
  dataset) is added.
- Light: single scattering for backscatter, a single-scattering halo scaled to the multiply
  scattered power for forward scatter, Lambertian surfaces, three colour bands rather than spectra,
  no shadows cast by anodes, no marine snow. Water is uniform.
- Hydrodynamics: added mass from an equivalent ellipsoid, quadratic drag from strip theory, no
  interaction with the nearby hull (no wall effect), no waves.
- The standoff sounder is an ideal ray with Gaussian range noise. Hull-end fixes assume the echo only
  starts or stops at a hull end.
- The controller's drag feed-forward uses the vehicle model's own damping (scaled by
  `vehicle.control.drag_feedforward`). The onboard assumes the phase function and daylight it is
  configured with; it measures attenuation and scattering.
- Revisits are planned for the nominal water; persistent turbidity leaves holes, which are reported.
