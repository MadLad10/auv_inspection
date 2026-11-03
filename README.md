# AUV Hull Inspection (simulation)

Underwater ship-hull inspection software built and tested entirely in simulation. The world has
exact ground truth: every defect's position, outline and area is known, and every rendered pixel
knows which defect it shows. The pipeline can therefore be scored by an independent referee, not
by its own claims.

**Status:** milestones 1 to 4 of 6 done, on a physically based environment: Fossen
6-DOF vehicle dynamics with added mass and contact, Jaffe-McGlamery underwater image formation with
a photon-counting camera, and a contrast-to-noise coverage criterion. Milestone 3 (perception) has
its dataset exporter, perception referee, a classical baseline and a learned region classifier
(numpy, no downloads) trained on hulls of every paint. It imports the real LIACi inspection images
for scoring and training, paints LIACi's real defects onto simulated hulls, and has a deep
segmenter (PyTorch, the `learn` extra) with an ImageNet-pretrained encoder that finds 70% of real
defects on the frozen test hulls, against 16% for the classical detector. Milestone 4 has the
defect map, anode stations checked against the hull plan, the inspection report with review
statuses, the hull months later (grown and new defects, aged anodes) and change detection between
two surveys. Milestone 5 has the benchmark and CI that gates on it.

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
- **Navigation from sensors only.** [`auv_navigation_fusion`](https://github.com/MadLad10/auv-navigation-fusion)'s EKF fuses
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
  battery loss, scheduled with onset and clear times and logged as ground truth. An empty battery
  drives no thruster.
- **Onboard health and safety.** The navigation filter's innovation gates reject a disturbed compass
  and readings inconsistent with the other sensors. A frozen-value check excludes the depth sensor
  while its readings repeat exactly (a live sensor always shows noise; a gate cannot see a sensor
  stuck at a plausible value). Below a battery reserve the vehicle abandons the mission and climbs
  along the hull at the standoff to the waterline, a path that keeps clear of the hull even from
  under the bilge.
- **Claimed coverage against the referee.** The vehicle judges each frame with the same image
  physics, from its estimated pose, the water its water-quality sensor measures, and the exposure
  and gain the camera used. A noiseless frame means a dead camera. Frames taken while the position
  uncertainty along the ship or the girth is above a limit count as seen but unverified; below it,
  each footprint shrinks by twice those uncertainties before it is claimed (on the hull extended
  past its ends and band, so only real edges shrink). The referee compares this claim with what the
  camera truly saw: a cell claimed but not seen is a false claim.
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
| Camera blackout, 20 s | 93.0% / 92.3% | 100% / 100% | 0 | 0.05 m | 19 / 19 |
| Turbidity (2 m visibility), 20 s | 93.0% / 92.3% | 100% / 100% | 0 | 0.05 m | 19 / 19 |
| DVL dropout, 60 s | 100% / 81.3% | 100% / 100% | 0 | 0.68 m | 19 / 19 |
| DVL dropout, 60 s, no hull-end fixes | 100% / 72.9% | revisits skipped (27% unverified) | 0 | 1.37 m | 19 / 19 |
| Stuck depth, compass offset 20°, gyro bias, sounder dropout, dead thruster | 100% / 100% | (none needed) | 0 | 0.05 m | 19 / 19 |
| 95% battery loss at 80 s | 82.0% / 81.0% | aborted, surfaced along the hull | 0 | 0.05 m | 13 / 19 |

In 2 m visibility the frames are milky haze: backscatter and forward scatter swamp the hull's
contrast, so no pixel reaches the contrast-to-noise limit. The vehicle measures the water and
agrees with the referee. During the DVL dropout the along-ship error grows by dead reckoning
(attitude noise leaks gravity into the accelerometer integration) until the next hull end, where
the edge search restores the fix; without hull-end fixes the vehicle cannot verify what it saw and
reports it as unverified. In the mixed-fault run the frozen-value check excludes the stuck depth
sensor 0.3 s after it freezes, keeping the girth error under 2 cm (22 cm without the check, while
the vehicle drifted on its buoyancy), and the filter rejects all compass readings while the
compass is disturbed.

### Perception (milestone 3, in progress)

- **Datasets** ([`dataset.py`](src/auvinspect/dataset.py)). Recorded frames become a labelled
  dataset: images, per-pixel class masks, YOLO box labels and an annotations file listing every
  visible object with its item id, box, pixel count, the share of its pixels the referee counts as
  inspectable, physical size and size bin. An object is *required* in a frame when it is big enough
  and inspectable enough to be found there; otherwise it is ignored. Frames go to train, val and test
  by scenario, so overlapping frames of one hull never straddle splits.
- **Perception referee** ([`perception.py`](src/auvinspect/perception.py)). Detections are matched to
  objects per frame and class by box overlap. It reports average precision, precision and recall at
  the reporting threshold, recall by size, false alarms per frame, and hull-level item recall: the
  share of the hull's defects found in at least one frame.
- **Classical baseline** ([`baseline.py`](src/auvinspect/baseline.py)). No learning, no downloads: a
  robust polynomial fit of log brightness removes the lamp pattern, and pixels whose colour departs
  from the paint by more than three median deviations become candidate regions. The paint colour is
  the run's own (the median over its frames), and pixels off it enter the fit with their own offset,
  so a frame mostly covered by fouling is still measured against paint. The water edge is rejected
  by shape. Classes come either from hand-written rules on the colour change against the paint and
  the shape (weld seams and gratings rejected), or from the region classifier.
- **Region classifier** ([`detector.py`](src/auvinspect/detector.py)). The rules were written for
  red-brown paint; on black paint fouling is brighter than the paint in every channel and the rules
  call it corrosion. Instead, each candidate region is described by 26 measures (its colour
  relative to the paint, the paint's albedo, the colour around it, its anomaly, size and shape) and
  a small ReLU network (two layers of 32, numpy) gives it a class or *none* (seams, gratings, glare,
  noise). It is trained on 10,295 regions from the 17 training hulls (red-brown, blue and black
  paint; `train-detector`), labelled from the per-pixel ground truth, with classes weighted towards
  balance; packaged as `models/region_classifier.json` with its training hulls and settings
  digest. `detector.model: null` restores the rules.

On 14 seeded hulls (1,344 frames, all red-brown paint and 10 m water; the 5 test hulls unseen while
tuning; simulation results), on the test split:

| Class | Rules: AP | Classifier: AP | Classifier: precision / recall at 0.5 | Hull items found |
|---|---|---|---|---|
| Corrosion | 0.81 | 0.77 | 0.85 / 0.81 | 25 / 26 |
| Coating damage | 0.30 | 0.49 | 0.53 / 0.65 | 13 / 13 |
| Biofouling | 0.58 | 0.50 | 0.44 / 0.74 | 15 / 15 |
| Crack | 0.02 | 0.20 | 0.20 / 0.25 | 4 / 6 |
| Missing anode | 0.00 | 0.83 | 0.71 / 0.83 | 2 / 2 |
| Anode | 0.01 | 0.06 | 0.08 / 0.18 | 14 / 33 |

Mean AP 0.29 with the rules and 0.48 with the classifier; false alarms per frame 1.4 and 0.74. On
red-brown paint the classifier mostly wins on what the rules confused (coating damage, cracks,
missing-anode footprints) and gives up a little on corrosion and fouling; on other paints it is
far ahead (see the [benchmark](#benchmark)). Anodes, 3-D fittings whose sides change their outline,
are checked differently, at the stations of the hull plan (below).

#### Real images: LIACi

The simulator draws its defects. To measure perception on real hulls, the pipeline reads
[LIACi](https://liaci.sintef.cloud): 1,893 images from ROV inspections of ships, with polygons
drawn by inspection experts in ten categories. LIACi is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial,
share-alike) and is **not part of this repository**. Download it from its site and cite:

> M. Waszak, A. Cardaillac, B. Elvesæter, F. Rødølen, M. Ludvigsen, "Semantic Segmentation in
> Underwater Ship Inspections: Benchmark and Data Set", *IEEE Journal of Oceanic Engineering*
> 48(2), 462-473, 2023. doi:10.1109/JOE.2022.3219129

`import-liaci` ([`liaci.py`](src/auvinspect/liaci.py)) writes LIACi in the dataset format above, so
the perception referee scores real and simulated images alike:

- **Splits.** LIACi's official split: 1,370 train and 191 test images. Of the training images,
  those in every 7th block of 50 consecutive image numbers are held out for validation (192
  images), since neighbouring images come from one video. The 332 images LIACi marks *not used* (near-duplicates) are left out.
- **Classes.** Corrosion, paint peel (coating damage), marine growth (biofouling) and anodes. LIACi
  has no cracks or missing anodes; its `defect` (mechanical damage, mostly on propellers), hull,
  propeller, bilge keel, sea chest grating and overboard valve categories are background.
- **Masks from the polygons**, smaller objects on top. LIACi's own per-class bitmaps hold one label
  per pixel: in 4% of class-and-image pairs they differ from the polygons. In the cases checked, the
  difference lies under another category drawn over it, for example corrosion on a propeller hub
  lost under the propeller; one image (`image_0205`) disagrees outright.
- **Resolution.** Images (640x480, 1280x720 and 1920x1080) are resized to 640 pixels wide, the
  simulator's camera width. Every object with 50 pixels or more is required: there are no
  coverage limits or physical sizes on a real image.

| Required objects (images) | Train | Val | Test | Median share of the image |
|---|---|---|---|---|
| Corrosion | 406 (127) | 55 (28) | 66 (23) | 0.37% (simulated: 0.41%) |
| Coating damage (paint peel) | 2,938 (554) | 369 (83) | 545 (95) | 0.25% (0.69%) |
| Biofouling (marine growth) | 1,834 (511) | 210 (75) | 338 (80) | 1.1% (3.5%) |
| Anode | 564 (337) | 115 (69) | 79 (52) | 0.57% (1.0%) |

What these images can and cannot say: they are real hulls, cameras and water, unlike the
simulator. But they come from a few inspections whose ships are not identified, the test images
come from the same inspections as the training images (LIACi only removed near-duplicates; a test
image is as similar to its nearest training image as training images are to each other), and many
show propellers, sea chests and valves rather than hull plating. Scores on them are scores on
LIACi's images, not field performance. The classical baseline does not run on them: it needs a
recorded run's paint reference and the hull's paint. They are for learned models.

#### Real defects in the simulator

`liaci-patches` ([`patches.py`](src/auvinspect/patches.py)) cuts LIACi's corrosion, paint peel and
marine growth out of its original images. It uses our training split only, so LIACi's test images
and our validation blocks stay unseen. `generate --patches` then paints these real defects on a
simulated hull in place of procedural ones. The rest of the world is unchanged (hull, anodes,
cracks, water, camera), and the labels stay exact, because the generator places every patch
itself.

- **Which objects.** An object is kept when it is whole (at least 2,000 pixels, clear of the image
  border, at most 5% of it under other labelled objects) and lies on plain hull. The ring around
  it must be at least 70% hull and no other object, neither murky nor glaring. At most 2% of the
  object's pixels may be saturated.
- **Relative reflectance.** The image is decoded from sRGB to linear and divided, per channel, by
  a plane fitted to the log of the ring. This removes the paint's colour, the lamp's colour and
  its falloff across the patch, and the water's colour. An object with more than 1% of its pixels
  outside the reflectance range [0.03, 10] is dropped: its ring was not the paint under it, as on
  a structure's edge.
- **Placement.** The defect's area is drawn as before. A patch of its class is drawn, turned,
  maybe mirrored and scaled to that area; patches are never enlarged more than 2.5 times when one
  big enough exists. Its relative reflectance multiplies the hull paint under it. The scenario
  records the library's digest and each item its patch: the same seed, configuration and library
  give the same digest. Procedural hulls keep their digests.
- **Change.** `evolve --patches` grows a real defect by continuing its own texture: each new pixel
  copies the pixel mirrored across the old outline. New defects come from the library.

| Patches (rejected) | Kept | Too small | Overlapped | Cut by border | Not on plain hull | Glare or murky | Out of range |
|---|---|---|---|---|---|---|---|
| Corrosion | 121 | 153 | 95 | 17 | 13 | 9 | 5 |
| Coating damage | 938 | 1,074 | 522 | 81 | 124 | 204 | 51 |
| Biofouling | 411 | 300 | 663 | 234 | 122 | 64 | 42 |

Limits: the haze in front of a patch, glare and the 3-D relief of fouling stay in it, painted
flat. Their ships' paint is divided out, so a patch on a paint unlike theirs carries their
contrast, not its own. LIACi has no scale, so sizes come from the configured ranges. Patches, and
every hull painted with them, are adaptations of LIACi under CC BY-NC-SA 4.0: they stay in `runs/`,
out of the repository and CI.

#### Deep segmenter

[`segmenter.py`](src/auvinspect/segmenter.py) (the `learn` extra) assigns a class to every pixel.
- **Encoder.** Torchvision's MobileNetV2 with its ImageNet weights. The 13.6 MB file is downloaded
  by hand and checked against its SHA-256; the code never downloads anything.
- **Decoder.** A light decoder merges the encoder's stride-8, -16 and -32 features.
- **Training.** 320x240 input, horizontal flips, brightness and contrast jitter, class-weighted
  cross-entropy, 20 epochs of AdamW. The epoch kept is the one with the best mean AP on the
  validation splits. An epoch over LIACi's training images takes about 75 s on this machine's CPU.
- **Scoring.** For the perception referee, each connected region of a class becomes a box, scored
  by its mean probability. Pixel IoU is counted against the masks.

Boxes understate a segmenter on LIACi. Neighbouring paint flakes merge into one region, which then
faces several drawn polygons. Pixel IoU is the fairer measure there.

On a recorded run, `segment --run` gives detections for the map and report.

Mean pixel IoU over corrosion, coating damage, biofouling and anodes on each test split (one run
each, seed 0):

| Trained on | LIACi test (real) | Simulated, procedural defects | Simulated, real defects |
|---|---|---|---|
| LIACi | **0.29** | 0.19 | 0.14 |
| Simulated, procedural defects | 0.05 | **0.78** | 0.22 |
| Simulated, real defects | 0.03 | 0.35 | **0.51** |
| LIACi + simulated, real defects | **0.29** | 0.38 | 0.48 |

The simulated datasets are the 14 hulls of the perception dataset (7 train, 2 val, 5 test),
red-brown paint, kinematic survey frames. Its procedural and real-defect versions share seeds, so
hulls, anodes and cracks are the same.

On LIACi's test images, the LIACi model scores per class:

| LIACi test | Corrosion | Coating damage | Biofouling | Anodes |
|---|---|---|---|---|
| Pixel IoU | 0.11 | 0.26 | 0.48 | 0.32 |
| Box AP | 0.01 | 0.05 | 0.09 | 0.41 |

Its box mAP is 0.14, with 3.0 false alarms per frame. These numbers are not comparable with the
LIACi paper's: the classes, splits and resolution differ.

What this shows:
- **Simulation alone does not transfer to real images,** with procedural or with real defects (IoU
  0.05 and 0.03 on LIACi). Real patches change the defects. The rest of a LIACi frame is still
  unlike the simulator's: cameras and lamps, haze, ship structures, propellers and sea chests, and
  the framing of a piloted ROV.
- **Procedural defects are easy.** The procedural model scores 0.78 on them, but 0.22 on the same
  hulls with real defects. Hulls with real defects are the harder test. The model trained on them
  is also the more general (0.35 on procedural defects).
- **One model can serve both.** Training on LIACi and on hulls with real defects together keeps
  LIACi's score (0.29) and nearly keeps the simulator's (0.48).

In the full pipeline, the segmenter replaces the classical detector; the map, multi-view
confirmation and referee are unchanged. The model used there, `seg_all`, is trained on LIACi and
on 17 training hulls with real defects in all three paints (7 from the table's dataset, 10 in
varied water and paint), and validated on LIACi and 4 validation hulls. It scores 0.29 on LIACi,
0.40 on procedural and 0.55 on real-defect hulls (the table's measure). Its results on the frozen
test suite are under [Benchmark](#benchmark).

Trained models stay in `runs/models/`. Those trained on LIACi or on hulls with real defects are
adaptations of LIACi (CC BY-NC-SA 4.0), so the `ci` gate keeps procedural defects and the
classical detector.

### Defect map, anode stations and report (milestone 4)

- **Placing** ([`mapping.py`](src/auvinspect/mapping.py)). Each detection's box is cast onto the
  hull from the camera pose the vehicle estimated for that frame (navigation, AHRS, tilt encoder),
  giving its box in hull coordinates and its area from the pixel count times the exact surface area
  of a pixel. A finding's area is the median over the views that saw it whole; a patch larger than
  a frame, cut by the image border in every view, gets the area of the union of its views times
  their fill.
- **Fusing and confirming.** Views of one class join a finding when they overlap most of the smaller
  box, so partial views of a large patch merge while neighbours stay apart. A finding's support is
  the share of the frames that saw its spot (by the vehicle's own poses) in which it was detected;
  findings with support of at least 0.5 are *confirmed*, the rest *possible*. None is dropped.
- **Anode stations** ([`stations.py`](src/auvinspect/stations.py)). The hull plan (the vessel's
  drawings: `anodes.spacing_m`, `girth_m`, `size_m`) says where every anode should be, so each
  station is asked what is fitted now: *intact*, *depleted* or *missing*; a station no frame saw is
  *not inspected*, never missing. Every frame that sees a station within the coverage limits (by the
  vehicle's own pose) measures the station window on the detector's paint-relative residual: the
  area of off-paint pixels and the colour of those pixels and of the plan footprint. A linear
  discriminant per paint scheme, trained on the training hulls with that paint (`train-stations`,
  packaged as `models/anode_stations.json` with the scenarios it was trained on and the digest of
  the settings its features depend on), scores each view; a station takes the status with the
  highest summed log likelihood. Colours relative to the paint do not carry over between paints (a
  slime film changes a black hull's colour far more than a red one's): one model for all paints
  confused depleted and missing anodes. A station whose views lie far from the status they chose
  (median squared Mahalanobis distance above `stations.max_distance2`), or on a hull whose paint has
  no model, is *unclear*: left to the inspector rather than given a confident guess. The defect map
  leaves anodes to the stations: detector anode classes, and detections
  centred within `stations.glare_margin_m` of a station window (the bright metal's glare, which the
  baseline takes for coating damage), make no findings, and the detector masks those zones so a
  defect beside an anode is not merged with it.
- **Report** ([`report.py`](src/auvinspect/report.py)). A hull photomosaic built from the vehicle's
  own frames at its own poses (lamp falloff divided out), with the findings boxed, the stations
  marked by status, the unverified areas hatched, tables of findings (location, extent, area,
  support, evidence crop) and stations (status, confidence, views, evidence crop), the vehicle's
  health log, and a review status per finding and station read from `review.json`. A separate
  appendix, marked simulation only, scores the map and the stations against the truth.
- **Map referee.** A finding matches an item of its class when their hull boxes overlap; further
  findings on one item are duplicates. Required items are those the coverage referee counted
  inspected and that some frame showed with enough pixels to find; inspected items that never did
  (cracks about one pixel wide) are listed as below the camera's resolution.

Seed 7 is the development hull: the map, glare margin and change-detection settings were tuned on
it (and on its evolved versions), so the seed-7 numbers below are optimistic; the
[benchmark](#benchmark) scores held-out hulls. With the classical baseline on the seed-7 survey
(simulation results), with the region classifier: all 23 findings cover 18 of the 19 findable
defects; the 17 confirmed findings cover 16 of them, none false (with the rules: 88 findings
covering all 19, 21 confirmed covering 17 at 86% precision). Matched findings sit within 1 to 8 cm
of the true centres; areas come out at 1.13x the truth for corrosion and 1.02x for biofouling, but
1.8x for coating damage, whose flaked edge the segmentation overestimates.

The station model is trained on 17 hulls: the 7 red-brown training hulls of the perception dataset
and 10 with varied water and paint (seeds 501 to 513, `generate --vary-world`; 3 red-brown, 3 blue,
4 black). It gets every inspected station right on the 7 held-out red-brown hulls (49 stations: 34
intact, 11 depleted, 4 missing; surveys at planned poses; with only 4 missing anodes the 95%
interval for finding one runs from 40% to 100%), on the seed-7 missions at 0, 12 and 24 months (21
stations, at the vehicle's estimated poses) and on the benchmark's dev hulls (28 stations), against
6 of 33 anodes and none of 2 missing anodes found by the baseline detector on the test hulls. A
model trained on red-brown hulls alone, given black hulls, got 19 of 21 stations right by luck and
the other 2 wrong with confidence 1.0, at distances of 140 to 970 from every status (6 to 23 on a
red-brown hull); such a station is now unclear.

### Change detection (milestone 4)

- **The hull months later** ([`evolution.py`](src/auvinspect/evolution.py)). `evolve` turns a
  scenario into the same hull some months on, at the rates in `evolution`: patch outlines advance
  until the area reaches (1 + rate)^months (by an amount that varies along the outline, stopping
  short of other items), cracks extend at both tips, new defects appear at a Poisson rate per square
  metre and month, intact anodes deplete and depleted ones fall off, leaving a footprint. Old pixels
  are untouched and every item keeps its id, so each change has exact ground truth. The evolved
  scenario records its base digest and the rates in its lineage, and has its own digest.
- **Comparing two surveys** ([`change.py`](src/auvinspect/change.py)). The later survey's findings,
  by score, pair with the earlier survey's findings of the same class when their hull boxes overlap;
  each pair or lone finding is *new*, *grown* (area ratio of at least `change.growth_ratio`, 1.25),
  *unchanged* or *gone*. Each station compares its two statuses: *anode depleted*, *anode lost*,
  *anode replaced*, *unchanged*, or *unknown* when a survey did not inspect it. The change referee
  checks each finding's label against the item the later finding matches, and each station's
  against the change in its true status.

Seed 7 (the development hull) evolved by 12 months (simulation results): corrosion grows 1.60x, coating damage 1.43x,
fouling 2.5x, cracks by 4.8 cm; 8 new defects appear and one anode depletes. Comparing the two
missions' maps (with the region classifier), the confirmed findings flag 4 of the 5 new defects they
found, with 1 false new label, and 10 of the 16 grown defects they found, with no false growth;
estimated growth ratios are within 5% of the truth (median). Growth is flagged for 6 of 8 corrosion
patches, 3 of 4 fouling patches and 1 of 4 coating patches, whose area is dominated by the
overestimated flaked edge. All findings together flag 6 of the 7 new defects found, with 11 false
new labels (38 with the rules). The station comparison flags the depleted anode and changes nothing
else.

At 24 months (fouling 6.3x, 11 new defects, two anodes depleted, one fallen off) the station
comparison is right at all 7 stations, flagging both depletions and the lost anode. Finding changes
are weaker once fouling patches cover square metres: the confirmed findings flag 8 of the 10 new
defects found (3 false new labels; 8 with the rules) and 11 of 13 grown ones (2 false; the rules: 7
of 13 with 4 false), with a median growth-ratio error of 19%.

## Benchmark

The numbers above come from a fixed world (one hull, one paint, 10 m visibility) and partly from the
development hull. The benchmark ([`benchmark.py`](src/auvinspect/benchmark.py)) scores the whole
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
- **CI** ([`.github/workflows/ci.yml`](.github/workflows/ci.yml)): on every push to `main` and every
  pull request, GitHub Actions installs the pinned versions (`constraints.txt`), runs the tests,
  flies the `ci` suite and applies its gate; the suite's scores and reports are kept as a build
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
[Limitations](#limitations)).

#### Real defects and the deep segmenter (milestone 3)

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
git clone https://github.com/MadLad10/auv_inspection.git && cd auv_inspection
~/.local/bin/uv venv .venv --python /usr/bin/python3.10
~/.local/bin/uv pip install --python .venv/bin/python -c constraints.txt -e ".[dev]"   # .[dev,learn] adds CPU PyTorch
export MUJOCO_GL=egl
.venv/bin/auvinspect generate --seed 7 --out runs/s7       # --vary-world: water and paint drawn from the seed
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

Perception dataset, baseline detections and their score (runs are directories inside their
scenario directory):

```bash
.venv/bin/auvinspect dataset  --run runs/s7/survey --run runs/s8/survey --out runs/dataset
.venv/bin/auvinspect detect   --dataset runs/dataset --split test --out runs/dataset/baseline_test.json
.venv/bin/auvinspect evaluate --dataset runs/dataset --detections runs/dataset/baseline_test.json --split test
```

The real LIACi images, downloaded from <https://liaci.sintef.cloud> (`data.zip`, 1.0 GB) and
unpacked outside the repository:

```bash
.venv/bin/auvinspect import-liaci  --root ~/datasets/liaci/LIACi_dataset_pretty --out runs/liaci
.venv/bin/auvinspect liaci-patches --root ~/datasets/liaci/LIACi_dataset_pretty --out runs/liaci_patches
.venv/bin/auvinspect generate --seed 7 --patches runs/liaci_patches --out runs/s7_real   # real defects
```

The deep segmenter (the `learn` extra) needs torchvision's MobileNetV2 ImageNet weights
(`mobilenet_v2-7ebf99e0.pth`, 13.6 MB, from
<https://download.pytorch.org/models/mobilenet_v2-7ebf99e0.pth>), checked against
`segmenter.backbone_sha256`. Nothing is downloaded by the code:

```bash
.venv/bin/auvinspect train-segmenter --dataset runs/liaci --backbone ~/models/mobilenet_v2-7ebf99e0.pth --out runs/models/seg.pt
.venv/bin/auvinspect segment  --model runs/models/seg.pt --dataset runs/liaci --split test --out runs/liaci/seg_test.json
.venv/bin/auvinspect evaluate --dataset runs/liaci --detections runs/liaci/seg_test.json --split test
```

Defect map and inspection report for a run (detections keyed by frame name; default: the baseline):

```bash
.venv/bin/auvinspect report --run runs/s7/mission --out runs/s7/report
```

An inspector reviews findings and stations by writing `runs/s7/report/review.json`, for example
`{"F012": {"status": "rejected", "note": "weld seam"}, "S03": {"status": "confirmed"}}`, and
re-running the report.

Anode stations of recorded runs, and the station model and region classifier refitted on
training-split hulls (the commands refuse others; a run configuration naming the file in
`stations.model` or `detector.model` uses it):

```bash
.venv/bin/auvinspect stations --run runs/s7/mission
.venv/bin/auvinspect train-stations --run runs/s101/survey --run runs/s102/survey --out runs/anode_stations.json
.venv/bin/auvinspect train-detector --run runs/s101/survey --run runs/s102/survey --out runs/region_classifier.json
```

The same hull 12 months later, a second mission over it, and the changes between the two reports:

```bash
.venv/bin/auvinspect evolve  --scenario runs/s7 --months 12 --out runs/s7_m12
.venv/bin/auvinspect mission --scenario runs/s7_m12 --out runs/s7_m12/mission
.venv/bin/auvinspect report  --run runs/s7/mission --out runs/s7/report_mission
.venv/bin/auvinspect report  --run runs/s7_m12/mission --out runs/s7_m12/report_mission
.venv/bin/auvinspect change  --before runs/s7/report_mission --after runs/s7_m12/report_mission --out runs/s7_m12/change
```

The sibling package [`auv_navigation_fusion`](https://github.com/MadLad10/auv-navigation-fusion)
is installed from its repository at a pinned commit (see `dependencies` in `pyproject.toml`), never
by name from an index: it is not on PyPI, and a package registered there under its name would be
someone else's code. `constraints.txt` holds the exact versions the tests and the `ci` gate last
passed with.

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
| `*/gt/*.npz` | Per-pixel label, depth, pixel footprint, contrast-to-noise ratio and the referee's inspectable mask |
| `survey/frames.jsonl` | Pose, camera pose, exposure and gain, newly covered cells, visible items with pixel counts |
| `*/coverage.json`, `coverage.png` | Coverage fraction, per-item inspection verdict, coverage map |
| `mission/frames.jsonl` | Per frame: true camera pose, phase and leg, exposure, gain and blur, active faults, the onboard verdict (claimed, uncertain, black, below limits) with its estimated camera pose and uncertainty, camera position error along the ship, girth and normal |
| `mission/coverage.json` | Referee coverage and items, claimed against true coverage, rounds and legs, navigation error, EKF summary, minimum clearance, contacts, fault log |
| `mission/claimed.png`, `claim_check.png` | Claimed coverage; agreement map (green both, red claimed but not seen, blue seen but not claimed) |
| `mission/trajectory.npz` | Per control tick: true pose, estimate, reference, thrust, position uncertainty, camera tilt, clearance |
| `dataset/images`, `masks`, `labels` | Per split: images, class masks (0 background, k = class k), YOLO box labels |
| `dataset/annotations.json` | Classes, splits, and per frame every visible object: item, class, box, pixels, inspectable share, size, required |
| `report/report.html` | Inspection report: coverage, hull map, findings table with evidence, health log, simulation-only appendix |
| `report/findings.json`, `metrics.json` | Findings (hull box, centre, area, views, support, tier, evidence, review, match) and the map referee |
| `report/mosaic.png`, `map.png`, `map_vs_truth.png`, `evidence/` | Photomosaic, findings map, findings against true items, evidence crops |
| `report/stations.json` | Each planned station: id, centre, plan box, status, confidence, views, evidence, review |
| `report/manifest.json` | The run, its scenario directory and the scenario digest |
| evolved `scenario.json` | As above, plus the lineage (base digest, months, rates); new items carry `new_after_months` |
| `change/changes.json`, `metrics.json` | Finding changes (label, class, earlier and later finding, tier, area ratio) and station changes (earlier and later status, label); the change referee for confirmed findings, all findings and stations |
| `src/auvinspect/models/region_classifier.json` | The region classifier: network weights, feature standardisation, classes, settings digest, training hulls (seed, scenario digest, paint, regions) |
| `src/auvinspect/models/anode_stations.json` | The station model: per paint, class means and pooled covariance; settings digest; training hulls (seed, scenario digest, paint) |
| `bench/<suite>/s<seed>/result.json` | One benchmark hull: drawn water and paint, the referees' verdicts (coverage, claims, map by tier and class, stations), versions |
| `bench/<suite>/benchmark.json` | The suite: pooled values with intervals, per-hull rows, missing hulls, versions |

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
- missions: nominal (full coverage, no false claims, no contact, hull-end fixes on every lane);
  blackout and turbidity holes found and revisited; DVL dropout with and without hull-end fixes,
  never over-claimed; the faults the vehicle absorbs (stuck depth caught by the frozen-value check,
  compass, gyro bias, sounder dropout, dead thruster); a low battery aborting the mission and
  surfacing along the hull; claims shrinking by the pose uncertainty except at band and hull edges;
  an empty battery driving nothing; determinism.

## Roadmap

1. ~~Scenarios, rendering, planning, coverage referee~~
2. ~~Vehicle in the loop: dynamics, navigation drift (`auv_navigation_fusion`), camera faults;
   replanning when coverage holes appear.~~
3. ~~Perception: pretrained models (depth, segmentation, features) and a fine-tuned defect
   detector, scored against the ground truth per class and per size.~~ Datasets and the
   perception referee, a classical baseline, a learned region classifier (numpy) trained on hulls
   of every paint, the real LIACi images, real LIACi defects painted on simulated hulls, and a
   deep segmenter (ImageNet-pretrained MobileNetV2) scored on real images, simulated images and
   the frozen test suite. No depth or feature model: the vehicle's range comes from its sounder
   and DVL.
4. ~~Defect map and report: defects placed on the hull with sizes and evidence frames; a second
   survey with grown defects for change detection; inspector review.~~ The map with multi-view
   confirmation and its referee, anode stations checked against the hull plan, the photomosaic
   report with review statuses, scenario evolution and change detection with its referee.
5. Production layer: simulation tests in CI that gate merges on coverage and detection metrics,
   containers, model registry, observability, fault injection. *Done so far: the benchmark (suites
   of hulls in varied water and paint, pooled scores with hull-resampling intervals, a gate), CI
   on GitHub Actions running the tests and the `ci` suite's gate with pinned versions, fault
   injection in missions, models packaged with their provenance and settings digest.*
6. Fleet: several vehicles sharing a structure under acoustic communication limits.

## Limitations

- Simulation only. Numbers describe this simulator and must not be read as real-world performance.
- Defect appearance in the simulator is procedural, or cut from LIACi's real images (corrosion,
  coating damage, fouling; cracks and missing anodes stay procedural). Real patches do not make
  the rest of the image real: a segmenter trained only on simulated hulls fails on LIACi's images.
  LIACi itself is small, from a few inspections, and not ship-disjoint between training and test.
- The detector's hand-written class rules were written for red-brown paint (on black paint they
  take fouling for corrosion); the region classifier, trained on all three paints, replaced them.
  Paints it has not seen are untested.
- Cracks: a crack 3 to 6 mm wide at 4 mm per pixel shows a pixel contrast (median 3.5 to 8 median
  deviations of a valley filter) within the paint's grain (its 99.9th percentile 8 to 16), and the
  candidate regions' one-pixel opening removes most crack pixels. The region classifier finds 23% of
  findable cracks on the test hulls; the deep segmenter finds 8 to 10 of 13.
- The deep segmenter makes about five false confirmed findings per test hull, against 0.25 for the
  classical detector on procedural defects. The map's confirmation settings were tuned with the
  classical detector. Its weights, trained on LIACi-derived data, cannot go into the repository or
  CI, so the CI gate does not cover it.
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
- The coverage criterion asks whether a faint stain over a 2 cm cell would be seen; it does not ask
  whether a feature narrower than a pixel is resolved. Cracks 3 to 6 mm wide at 4 mm per pixel can
  count as covered yet show too few pixels to find; the map referee lists them separately.
