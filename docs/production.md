# Benchmark, CI and operations

Scoring the whole pipeline on held-out hulls, gating changes on it in CI, the container image, the model registry and observability.

## Benchmark

Most numbers on the other pages come from a fixed world (one hull, one paint, 10 m visibility) and
partly from the development hull, seed 7. The benchmark ([`benchmark.py`](../src/auvinspect/benchmark.py)) scores the whole
pipeline on hulls it was not tuned on, in varied worlds:

- **Suites** (`benchmark.suites`): `ci`, two 8 m hulls for a quick check of every change; `dev`,
  four hulls for tuning; `test`, eight hulls, frozen: scored to report a milestone, never tuned on.
- **Varied worlds.** Each hull draws from its seed a water visibility (4 to 15 m, by scaling the
  particle scattering) and one of three antifouling paints (red-brown, black, blue). The coverage
  criterion judges a faint defect against the hull's own paint. The region classifier and the
  station model are trained on training-split hulls only: seven in the fixed world and ten whose
  water and paint are drawn as the benchmark's are (`generate --vary-world`).
- **Per hull, end to end:** generate, fly the mission, detect, map, check the stations, report, and
  keep the referees' verdicts in `result.json`.
- **Score:** values pooled over the suite's hulls (items found over items required, stations right
  over stations inspected, mean coverage) with a 95% interval from resampling hulls, since the hull
  is the unit of independence; suites under five hulls get no interval.
- **Gate:** `benchmark gate` fails (exit status 1) when a hull is missing or aborted, or a value
  falls below its minimum (above its maximum) in `benchmark.gate`.
- **CI** ([`.github/workflows/ci.yml`](../.github/workflows/ci.yml)): on every push to `main` and every
  pull request, GitHub Actions installs the pinned versions (`constraints.txt`), runs the tests,
  flies the `ci` suite and applies its gate, then the `ci_fleet` suite and its gate; the suite's scores and reports are kept as a build
  artifact. The runner has no GPU and renders with Mesa's software EGL device.

```bash
.venv/bin/auvinspect benchmark run   --suite dev --out runs/bench/dev            # all hulls; --seed for one
.venv/bin/auvinspect benchmark score --suite dev --out runs/bench/dev            # writes benchmark.json
.venv/bin/auvinspect benchmark gate  --suite ci  --out runs/bench/ci
```

Test suite (simulation results; 8 hulls: 4 blue, 2 black, 2 red-brown; visibility 4.8 to 12.5 m;
95% intervals from resampling hulls), scored once with the rules and once with the region classifier
(trained on the training hulls and tuned on the dev hulls in between, never on these):

| Value | Rules | Region classifier |
|---|---|---|
| Coverage (referee), false claims | 100% on every hull, none | same |
| Stations right | 55 / 56 (98%, 95 to 100%); 9 of 9 missing anodes found | same |
| Defects found by confirmed findings | 66% (57 to 76%) | 86% (79 to 92%) |
| Defects found by any finding | 72% (64 to 80%) | 90% (84 to 95%) |
| Precision of confirmed findings | 49% (38 to 63%) | 98% (96 to 100%) |
| False confirmed findings per hull | 10.4 | 0.25 |
| Corrosion, coating damage, fouling found | 100%, 91%, 29% | 100%, 97%, 97% |
| Cracks found | 54% (7 of 13), with about 50 false crack findings per hull | 23% (3 of 13) |

Coverage and the anode stations carried over to new water and paints from the start. The rules'
detector did not: fouling was found on the two red-brown hulls (10 of 10) and on none of the six
others (0 of 25). The region classifier, trained on hulls of all three paints, finds fouling on
every paint and almost no false findings. It finds fewer cracks than the rules, which reported
every dark line: it learned from 42 crack regions only, and a crack 3 to 6 mm wide at 4 mm per
pixel stands out less from the paint's grain than the grain does from itself (see
[Limitations](limitations.md)).

### Real defects and the deep segmenter

`benchmark run --patches` paints the suite's hulls with real LIACi defects; `--segmenter` detects
with the deep segmenter; `--missions` reuses another run's flown missions, so detectors are
compared on the same flights. The test suite, flown again with real defects (same hulls, water,
paint, anodes and cracks), scored once with each detector. Confirmed findings, 95% intervals from
resampling hulls:

| Test suite, 8 hulls | Classical (region classifier) | Deep segmenter (`seg_all`) |
|---|---|---|
| **Real defects:** defects found | 16% (12 to 20%) | **70% (65 to 76%)** |
| Found by any finding | 34% (28 to 40%) | **93% (88 to 96%)** |
| Precision | 48% (37 to 59%) | **68% (63 to 74%)** |
| False confirmed findings per hull | 2.9 (2.0 to 4.0) | 4.8 (3.4 to 6.0) |
| Corrosion, coating damage, fouling, cracks | 5/39, 6/35, 7/35, 2/13 | 30/39, 18/35, 30/35, 8/13 |
| **Procedural defects:** defects found | **86% (79 to 92%)** | 66% (58 to 75%) |
| Precision | **98% (96 to 100%)** | 68% (59 to 77%) |
| False confirmed findings per hull | **0.25** | 4.9 |
| Corrosion, coating damage, fouling, cracks | 37/39, 32/35, 34/35, 2/13 | 15/39, 25/35, 31/35, 10/13 |

Coverage (100%) and stations (55 of 56) do not depend on the detector. Real defects take the
classical detector from 86% to 16%: its candidate regions and region classifier learned the
procedural defects' flat colours. The segmenter, trained on real defects (and never on these
hulls), finds 70%. It also finds most cracks (8 to 10 of 13, against 2), which are procedural on
both versions of the hulls. Its cost is false findings: about 5 confirmed per hull, mostly
corrosion and fouling, on either version. The map's confirmation settings were tuned with the
classical detector, on procedural defects. The dev suite (4 hulls) gave the same picture: 10%
against 64% of real defects found.

```bash
.venv/bin/auvinspect benchmark run --suite test --patches runs/liaci_patches --out runs/bench/test_real
.venv/bin/auvinspect benchmark run --suite test --patches runs/liaci_patches --segmenter runs/models/seg_all.pt \
    --missions runs/bench/test_real --out runs/bench/test_real_seg
```

## Model registry ([`registry.py`](../src/auvinspect/registry.py))

Models are registered as numbered versions in `runs/registry` (`registry.root`). Each version is a
read-only copy of the model file and a card that records:
- the file's SHA-256, and the identity the benchmark records for the model;
- what it was trained on, and its settings digest;
- its license, worked out from its training sources;
- the benchmark scores attached to it;
- its stage (candidate, production, archived), with a dated history.

The rules:
- **License.** A model trained on LIACi's images, or on hulls painted with LIACi's defects, is an
  adaptation of LIACi (CC BY-NC-SA 4.0). One whose sources are no longer on disk counts as
  unknown. Neither kind may be redistributed. Segmenters never are either: their encoder holds
  torchvision's ImageNet weights.
- **Scores.** A benchmark summary is attached only when every hull in it was scored with that
  version.
- **Promotion.** A version becomes `production` only with a complete score of the kind's promotion
  suite (`registry.promotion`): every hull present, none aborted, on real or procedural defects as
  configured, its values within limits. The version it replaces is archived.
- **Use.** `registry:<name>` (the production version) or `registry:<name>@<version>` works wherever
  a model file is taken (`detector.model`, `stations.model`, `--segmenter`, `segment --model`), and
  the file is checked against its digest.
- **Packaging.** `package` copies a production JSON model into the package (for the CI gate) only
  when it may be redistributed.

```bash
.venv/bin/auvinspect registry add --model runs/models/seg_all.pt --name segmenter
.venv/bin/auvinspect benchmark score --suite dev --out runs/bench/dev_real_seg
.venv/bin/auvinspect registry attach segmenter@1 --score runs/bench/dev_real_seg/benchmark.json
.venv/bin/auvinspect registry promote segmenter@1          # `registry check` lists what stops it
.venv/bin/auvinspect benchmark run --suite test --patches runs/liaci_patches --segmenter registry:segmenter ...
```

| Registered | License | Redistributable | Promoted on (dev suite) |
|---|---|---|---|
| `region_classifier@1` (packaged) | this project | yes | procedural: recall 0.82, precision 0.94, 0.25 false per hull |
| `anode_stations@1` (packaged) | this project | yes | procedural: stations 28 of 28 |
| `segmenter@1` (`seg_all`) | CC BY-NC-SA 4.0, LIACi adaptation | no | real defects: recall 0.64, precision 0.75, 3.25 false per hull |

## Observability ([`observe.py`](../src/auvinspect/observe.py))

What an operator sees of a mission, kept apart from what only the simulator knows:
- **Event stream** (`events.jsonl` in a mission directory). It is written and flushed as the
  vehicle flies, and holds only what the vehicle knows: no ground truth, and no fault injected
  unless its symptoms show. Events:
  - the mission starting and finishing, and each leg's outcome;
  - a status heartbeat every 10 simulated seconds: estimated hull position and its uncertainty,
    battery, frames by claim, claimed share;
  - revisits planned or skipped;
  - the DVL lost and restored, dead camera frames, sensors excluded by the health checks;
  - turbid water, from the water-quality sensor;
  - low battery, and an abort.

  Events carry simulated time only, so the same mission writes the same stream. `auvinspect watch`
  follows the stream as the mission flies.
- **Metrics** (`metrics.prom`, Prometheus text format for a textfile collector). The vehicle's own
  numbers are `auvinspect_*`: frames by claim, legs by outcome, events by level, claimed share,
  holes left, battery. The referee's verdicts, which exist only in simulation, are
  `auvinspect_referee_*`: coverage, false claims, navigation error, clearance, contacts, faults
  injected. `benchmark score` and `gate` write a suite's values and its gate result the same way.
- **Stage traces** (`trace.jsonl` beside a benchmark hull's result): generate, fly, detect and
  report, with start time, duration and status; their durations are metrics too.
- **Alert rules** (`observability.alerts`): conditions on a run's metrics, such as an abandoned
  mission, critical events, claimed coverage short of 99%, holes left, dead frames, legs timed out,
  false claims or contact. `auvinspect health --run` prints the alerts that fire and exits 1 on a
  critical one. CI applies them to the `ci` suite's missions and keeps the streams, metrics and
  traces with the scores.

A mission on the small hull with a 30 s DVL dropout and a 15 s camera blackout, as `watch` shows it
(status lines left out):

```
     0.00 s  INFO     mission_started   seed=3 scenario=6422cddef575 legs=8 lanes=4 cameras=['port']
    60.00 s  WARNING  dvl_lost
    62.90 s  INFO     leg_finished      phase=survey kind=lane lane=1 outcome=done seconds=28.0
    90.00 s  INFO     dvl_restored
   104.50 s  WARNING  camera_black
   118.25 s  INFO     camera_restored
   132.55 s  INFO     revisit_planned   round=1 holes=1 legs=4
   184.95 s  INFO     mission_finished  outcome=completed claimed_fraction=1.0 holes_remaining=0 frames={'claimed': 39, 'uncertain': 7, 'black': 4, 'below_limits': 0} battery_pct=98.9
```

`health` then raises one warning (dead frames) and no critical alert. The event stream does not
change the mission: the `ci` suite flown with it gives the same results.

## Container

The [`Dockerfile`](../Dockerfile) builds the pipeline headless:
- Ubuntu 22.04 and Python 3.10, as on the CI runner, with the base image pinned by digest;
- Mesa's software EGL renderer;
- the versions in `constraints.txt`;
- run as a non-root user, with no PyTorch.

By default the navigation package is built from its repository at the commit pinned in
`pyproject.toml`; `--build-context nav=<dir>` takes an already built wheel instead. Inside the
image the full test suite passes (the two PyTorch tests are skipped) and the `ci` gate passes. CI
builds the image and surveys a small hull in it.

```bash
docker build -t auvinspect .        # or: --build-context nav=../auv_navigation_fusion/dist
docker run --rm -v "$PWD/runs:/work/runs" auvinspect benchmark run --suite ci --out runs/bench/ci
docker run --rm -v "$PWD/runs:/work/runs" auvinspect benchmark gate --suite ci --out runs/bench/ci
```

Results are bit-identical on one renderer, not across renderers. A GPU's renderer and Mesa's
software renderer sample the texture slightly differently, and the sensor's photon noise, drawn
for each pixel's signal, then differs everywhere. The scenarios, the flown missions, the pixel
labels and the coverage verdicts stay the same, and the scores move within the noise. On the `ci`
suite, confirmed-findings recall is 0.83 on this machine's Intel GPU and 0.86 in software, as in
the image and on CI; both pass the gate.
