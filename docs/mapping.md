# Defect map, report and change detection

From per-frame detections to findings placed on the hull, anode stations checked against the hull plan, an inspection report, and changes between two surveys of the same hull.

## Defect map, anode stations and report

- **Placing** ([`mapping.py`](../src/auvinspect/mapping.py)). Each detection's box is cast onto the
  hull from the camera pose the vehicle estimated for that frame (navigation, AHRS, tilt encoder),
  giving its box in hull coordinates and its area from the pixel count times the exact surface area
  of a pixel. A finding's area is the median over the views that saw it whole; a patch larger than
  a frame, cut by the image border in every view, gets the area of the union of its views times
  their fill.
- **Fusing and confirming.** Views of one class join a finding when they overlap most of the smaller
  box, so partial views of a large patch merge while neighbours stay apart. A finding's support is
  the share of the frames that saw its spot (by the vehicle's own poses) in which it was detected;
  findings with support of at least 0.5 are *confirmed*, the rest *possible*. None is dropped.
- **Anode stations** ([`stations.py`](../src/auvinspect/stations.py)). The hull plan (the vessel's
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
- **Report** ([`report.py`](../src/auvinspect/report.py)). A hull photomosaic built from the vehicle's
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
[benchmark](production.md#benchmark) scores held-out hulls. With the classical baseline on the seed-7 survey
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

## Change detection

- **The hull months later** ([`evolution.py`](../src/auvinspect/evolution.py)). `evolve` turns a
  scenario into the same hull some months on, at the rates in `evolution`: patch outlines advance
  until the area reaches (1 + rate)^months (by an amount that varies along the outline, stopping
  short of other items), cracks extend at both tips, new defects appear at a Poisson rate per square
  metre and month, intact anodes deplete and depleted ones fall off, leaving a footprint. Old pixels
  are untouched and every item keeps its id, so each change has exact ground truth. The evolved
  scenario records its base digest and the rates in its lineage, and has its own digest.
- **Comparing two surveys** ([`change.py`](../src/auvinspect/change.py)). The later survey's findings,
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
