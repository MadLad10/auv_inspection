# Usage

Every command is a subcommand of `auvinspect` (`.venv/bin/auvinspect --help` lists them). Runs are
directories inside their scenario directory, for example `runs/s7/mission` inside `runs/s7`.

## Install

Linux, Python 3.10, [uv](https://docs.astral.sh/uv/). Rendering is headless through EGL.

```bash
git clone https://github.com/MadLad10/auv_inspection.git && cd auv_inspection
uv venv .venv --python 3.10
uv pip install --python .venv/bin/python -c constraints.txt -e ".[dev]"   # ".[dev,learn]" adds CPU PyTorch
export MUJOCO_GL=egl
```

`constraints.txt` holds the exact versions the tests and the CI gate last passed with. The
navigation filter, [`auv-navigation-fusion`](https://github.com/MadLad10/auv-navigation-fusion), is
installed from its repository at the commit pinned in `pyproject.toml`, never by name from a
package index: it is not on PyPI, and a package registered there under its name would be someone
else's code.

## Scenarios, surveys and missions

```bash
.venv/bin/auvinspect generate --seed 7 --out runs/s7       # --vary-world: water and paint drawn from the seed
.venv/bin/auvinspect survey   --scenario runs/s7 --out runs/s7/survey     # planned poses, no vehicle
.venv/bin/auvinspect mission  --scenario runs/s7 --out runs/s7/mission    # the vehicle in the loop
.venv/bin/auvinspect preview  --scenario runs/s7 --texture --out runs/s7/texture.png
.venv/bin/auvinspect preview  --scenario runs/s7 --x 10 --girth 3.9 --out runs/s7/rig.png
```

- `--no-frames` scores coverage without writing images.
- `mission --seed` changes the sensor noise and the initial fix without changing the world.
- `--config run.yaml` overrides run settings (cameras, survey, coverage, vehicle, navigation,
  faults, ...) without changing the scenario. A fault schedule, for example:

```yaml
faults:
  - {type: camera_blackout, t_start: 200.0, duration: 20.0}
  - {type: dvl_dropout, t_start: 60.0, duration: 60.0}
  - {type: turbidity, t_start: 300.0, duration: 30.0, params: {visibility_m: 2.0}}
```

Follow a mission's event stream while it flies, and apply the alert rules afterwards (exit status
1 on a critical alert):

```bash
.venv/bin/auvinspect watch  --run runs/s7/mission
.venv/bin/auvinspect health --run runs/s7/mission
```

## Defect map and report

```bash
.venv/bin/auvinspect report --run runs/s7/mission --out runs/s7/report     # detects with the classical baseline
```

An inspector reviews findings and stations by writing `runs/s7/report/review.json`, for example
`{"F012": {"status": "rejected", "note": "weld seam"}, "S03": {"status": "confirmed"}}`, and
running the report again.

The same hull 12 months later, a second mission over it, and the changes between the two reports:

```bash
.venv/bin/auvinspect evolve  --scenario runs/s7 --months 12 --out runs/s7_m12
.venv/bin/auvinspect mission --scenario runs/s7_m12 --out runs/s7_m12/mission
.venv/bin/auvinspect report  --run runs/s7_m12/mission --out runs/s7_m12/report
.venv/bin/auvinspect change  --before runs/s7/report --after runs/s7_m12/report --out runs/s7_m12/change
```

## Fleet

```bash
.venv/bin/auvinspect fleet  --scenario runs/s7 --vehicles 2 --out runs/s7/fleet2   # --config: fleet.vehicle_faults
.venv/bin/auvinspect report --run runs/s7/fleet2/merged --out runs/s7/fleet2/report
```

## Perception datasets and detectors

Export recorded runs as a labelled dataset, run the classical baseline and score it:

```bash
.venv/bin/auvinspect dataset  --run runs/s7/survey --run runs/s8/survey --out runs/dataset
.venv/bin/auvinspect detect   --dataset runs/dataset --split test --out runs/dataset/baseline_test.json
.venv/bin/auvinspect evaluate --dataset runs/dataset --detections runs/dataset/baseline_test.json --split test
```

Refit the region classifier and the anode station model on training-split hulls (the commands
refuse other hulls; a run configuration naming the file in `detector.model` or `stations.model`
uses it), and check the stations of a run:

```bash
.venv/bin/auvinspect train-detector --run runs/s101/survey --run runs/s102/survey --out runs/region_classifier.json
.venv/bin/auvinspect train-stations --run runs/s101/survey --run runs/s102/survey --out runs/anode_stations.json
.venv/bin/auvinspect stations --run runs/s7/mission
```

## Real images (LIACi) and the deep segmenter

The LIACi images come from <https://liaci.sintef.cloud> (`data.zip`, 1.0 GB, CC BY-NC-SA 4.0).
Unpack them outside the repository:

```bash
.venv/bin/auvinspect import-liaci  --root ~/datasets/liaci/LIACi_dataset_pretty --out runs/liaci
.venv/bin/auvinspect liaci-patches --root ~/datasets/liaci/LIACi_dataset_pretty --out runs/liaci_patches
.venv/bin/auvinspect generate --seed 7 --patches runs/liaci_patches --out runs/s7_real   # real defects
```

The deep segmenter (the `learn` extra) needs torchvision's MobileNetV2 ImageNet weights
(`mobilenet_v2-7ebf99e0.pth`, 13.6 MB, from
<https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth>), checked against
`segmenter.backbone_sha256`. The code never downloads anything.

```bash
.venv/bin/auvinspect train-segmenter --dataset runs/liaci --backbone ~/models/mobilenet_v2-7ebf99e0.pth --out runs/models/seg.pt
.venv/bin/auvinspect segment  --model runs/models/seg.pt --dataset runs/liaci --split test --out runs/liaci/seg_test.json
.venv/bin/auvinspect evaluate --dataset runs/liaci --detections runs/liaci/seg_test.json --split test
.venv/bin/auvinspect segment  --model runs/models/seg.pt --run runs/s7/mission --out runs/s7/mission/seg.json
.venv/bin/auvinspect report   --run runs/s7/mission --detections runs/s7/mission/seg.json --out runs/s7/report_seg
```

## Benchmark and model registry

```bash
.venv/bin/auvinspect benchmark run   --suite dev --out runs/bench/dev      # all hulls; --seed for one
.venv/bin/auvinspect benchmark score --suite dev --out runs/bench/dev      # writes benchmark.json
.venv/bin/auvinspect benchmark gate  --suite ci  --out runs/bench/ci       # exit 1 when the gate fails

.venv/bin/auvinspect registry add     --model runs/models/seg_all.pt --name segmenter
.venv/bin/auvinspect registry attach  segmenter@1 --score runs/bench/dev_real_seg/benchmark.json
.venv/bin/auvinspect registry promote segmenter@1                          # `registry check` lists what stops it
```

See [production.md](production.md) for the suites, the gate and the registry's rules.

## Container

```bash
docker build -t auvinspect .        # or: --build-context nav=../auv_navigation_fusion/dist
docker run --rm -v "$PWD/runs:/work/runs" auvinspect benchmark run  --suite ci --out runs/bench/ci
docker run --rm -v "$PWD/runs:/work/runs" auvinspect benchmark gate --suite ci --out runs/bench/ci
```

## Tests

```bash
env -u PYTHONPATH MUJOCO_GL=egl .venv/bin/python -m pytest -q
```

`env -u PYTHONPATH` stops pytest loading ROS 2 plugins from a sourced ROS environment. What the
tests check is listed in [testing.md](testing.md).
