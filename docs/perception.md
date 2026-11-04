# Perception

Three detectors (a classical baseline, a learned region classifier and a deep segmenter) and the perception referee that scores them per frame, on simulated hulls and on the real LIACi inspection images.

## Datasets and detectors

- **Datasets** ([`dataset.py`](../src/auvinspect/dataset.py)). Recorded frames become a labelled
  dataset: images, per-pixel class masks, YOLO box labels and an annotations file listing every
  visible object with its item id, box, pixel count, the share of its pixels the referee counts as
  inspectable, physical size and size bin. An object is *required* in a frame when it is big enough
  and inspectable enough to be found there; otherwise it is ignored. Frames go to train, val and test
  by scenario, so overlapping frames of one hull never straddle splits.
- **Perception referee** ([`perception.py`](../src/auvinspect/perception.py)). Detections are matched to
  objects per frame and class by box overlap. It reports average precision, precision and recall at
  the reporting threshold, recall by size, false alarms per frame, and hull-level item recall: the
  share of the hull's defects found in at least one frame.
- **Classical baseline** ([`baseline.py`](../src/auvinspect/baseline.py)). No learning, no downloads: a
  robust polynomial fit of log brightness removes the lamp pattern, and pixels whose colour departs
  from the paint by more than three median deviations become candidate regions. The paint colour is
  the run's own (the median over its frames), and pixels off it enter the fit with their own offset,
  so a frame mostly covered by fouling is still measured against paint. The water edge is rejected
  by shape. Classes come either from hand-written rules on the colour change against the paint and
  the shape (weld seams and gratings rejected), or from the region classifier.
- **Region classifier** ([`detector.py`](../src/auvinspect/detector.py)). The rules were written for
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
far ahead (see the [benchmark](production.md#benchmark)). Anodes, 3-D fittings whose sides change their outline,
are checked differently, at the stations of the hull plan (see [mapping.md](mapping.md)).

## Real images: LIACi

The simulator draws its defects. To measure perception on real hulls, the pipeline reads
[LIACi](https://liaci.sintef.cloud): 1,893 images from ROV inspections of ships, with polygons
drawn by inspection experts in ten categories. LIACi is licensed
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/) (non-commercial,
share-alike) and is **not part of this repository**. Download it from its site and cite:

> M. Waszak, A. Cardaillac, B. Elvesæter, F. Rødølen, M. Ludvigsen, "Semantic Segmentation in
> Underwater Ship Inspections: Benchmark and Data Set", *IEEE Journal of Oceanic Engineering*
> 48(2), 462-473, 2023. doi:10.1109/JOE.2022.3219129

`import-liaci` ([`liaci.py`](../src/auvinspect/liaci.py)) writes LIACi in the dataset format above, so
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

## Real defects in the simulator

`liaci-patches` ([`patches.py`](../src/auvinspect/patches.py)) cuts LIACi's corrosion, paint peel and
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

## Deep segmenter

[`segmenter.py`](../src/auvinspect/segmenter.py) (the `learn` extra) assigns a class to every pixel.
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
test suite are under [Benchmark](production.md#benchmark).

Trained models stay in `runs/models/`. Those trained on LIACi or on hulls with real defects are
adaptations of LIACi (CC BY-NC-SA 4.0), so the `ci` gate keeps procedural defects and the
classical detector.
