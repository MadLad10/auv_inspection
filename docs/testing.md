# Tests

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
- mapping and report: perfect detections mapping every findable item in place with the right area;
  partial views fusing while neighbours stay apart; areas from whole views, or from the union of
  views cut by the image border; the report's files and review statuses;
- anode stations: the plan holding every generated anode, the station check right on an unseen
  hull with the packaged model, detections at a station making no findings, the classifier voting
  over views, never guessing an unseen station, calling unclear what looks like no status or has
  no model for its paint, the packaged model's settings, training split and paints;
- region classifier: regions labelled from the ground truth, the network learning and
  deterministic, the packaged model trained on training hulls of every paint and refused under
  other settings, detections taking their classes (or none) from it;
- benchmark: each hull's water and paint drawn from its seed, the coverage criterion following the
  paint, scores pooled over hulls with intervals only for enough hulls, the gate;
- change: evolution growing patches no faster than their rate, extending cracks by the configured
  length, keeping ids, old pixels and determinism, recording the base digest; change labels (new,
  grown, unchanged, anode lost, gone) and the change referee on known cases;
- perception: the referee's hits, false alarms, ignored objects, items across hulls and average
  precision on known cases; dataset objects against the per-pixel ground truth; the baseline
  rejecting weld seams, finding fouling that fills most of a frame (with the paint reference) and
  finding corrosion on the test hull;
- LIACi import (on a stand-in with LIACi's layout): the official split with validation blocks and
  near-duplicates left out, the class mapping with other categories as background, resizing,
  smaller objects drawn on top, small objects not required, perfect detections scoring full marks,
  the baseline refusing real images;
- real defects in the simulator (same stand-in, defects of known reflectance under a lamp falling
  off across the image): only whole defects on plain hull from the training split kept, the paint
  and the lamp divided out, the library's digest repeatable; a hull painted with them repeatable,
  its labels the painted patches, the paint under a patch times its reflectance, procedural hulls'
  digests unchanged; evolution needing the same library, keeping old pixels and growing a patch
  from its own texture;
- deep segmenter (a randomly initialised encoder in place of the ImageNet weights): regions to
  boxes and scores, the best validation epoch kept, the model reloading, the weights' digest
  enforced;
- benchmark: reused missions refused when missing or of another hull;
- model registry: read-only versions with the benchmark's identity, the license following the
  training sources (procedural, LIACi-derived, missing), scores refused unless made with the
  model, promotion refused without a score, under the limits or on the wrong defects, the replaced
  version archived, a registered model checked against its digest where it is used, packaging
  refused for what may not be redistributed;
- missions: nominal (full coverage, no false claims, no contact, hull-end fixes on every lane);
  blackout and turbidity holes found and revisited; DVL dropout with and without hull-end fixes,
  never over-claimed; the faults the vehicle absorbs (stuck depth caught by the frozen-value check,
  compass, gyro bias, sounder dropout, dead thruster); a low battery aborting the mission and
  surfacing along the hull; claims shrinking by the pose uncertainty except at band and hull edges;
  an empty battery driving nothing; determinism. Each fault's event at its time (DVL lost and
  restored, camera black and restored, turbid water, sensor excluded, battery low then abort), the
  nominal stream free of ground truth and warnings, the heartbeat's spacing, the metrics against
  the summary, the alert rules on each run;
- observability: the metrics format round-tripped (labels escaped, NaN), alert rules matching
  labels, stage spans kept when a stage fails, `watch` stopping at the mission's end;
- fleet: lanes shared in contiguous groups with bands between them; a fleet of one identical to
  the single-vehicle mission (frames and events); two vehicles covering the small hull in under
  60% of the time, each on its own lanes, never closer than the limit, the fleet's metrics; an
  aborted vehicle's lanes handed over and flown (coverage kept, no lane twice, apart throughout),
  and left undone without coordination; a silent vehicle keeping its share and ending its mission
  when it loses the link; the merged run (every vehicle's frames, the fleet's verdict, aborted
  only with lanes left unflown); fleet values in the benchmark only for fleet suites, and gated;
  three vehicles in stretches with fixes from the ship, every lane of every stretch flown once,
  each vehicle's frames in its stretch, nothing touched after an abort;
- acoustic link and coordinator: a message waiting for its slot, arriving after its time on the
  air and the sound's travel, acknowledged and not resent; a slot carrying what fits; lost packets
  resent until acknowledged, given up when no one hears, nothing heard beyond range; lanes handed
  over only to a vehicle done with its own, with their band; a silent vehicle's lanes never
  handed over, the others released only past its planned finish; the ship's fix of where a
  vehicle was when it transmitted, with its error growing with range.
