# Output files

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
