"""Run a planned survey through the renderer: frames, per-frame ground truth, coverage verdict."""
from __future__ import annotations

import json
import logging
import os
import time

import numpy as np
from PIL import Image

from .coverage import Coverage
from .planner import lane_plan
from .render import InspectionRenderer
from .rng import stream
from .scenario import Scenario

log = logging.getLogger(__name__)


def visible_items(f) -> dict[int, int]:
    ids, counts = np.unique(f.label[f.label >= 0], return_counts=True)
    return {int(i): int(c) for i, c in zip(ids, counts)}


def save_frame(out_dir: str, name: str, f):
    """Image under frames/ and per-pixel ground truth under gt/: label, depth, pixel footprint on the
    hull and the contrast-to-noise ratio of the reference defect over one coverage cell."""
    Image.fromarray(f.rgb).save(os.path.join(out_dir, 'frames', name + '.png'), compress_level=1)
    np.savez_compressed(os.path.join(out_dir, 'gt', name + '.npz'), label=f.label, depth=f.depth,
                        gsd_mm=np.where(f.on_hull, np.minimum(f.gsd_mm, 6e4), np.nan).astype(np.float16),
                        cnr=np.minimum(f.cnr, 6e4).astype(np.float16))


def run(scn: Scenario, out_dir: str, save_frames: bool = True, cameras: list[str] | None = None) -> dict:
    cfg = scn.cfg
    inspect_cam = cfg['survey']['camera']
    cams = list(dict.fromkeys([inspect_cam] + list(cameras or cfg['survey']['record_cameras'])))
    plan = lane_plan(scn.hull, cfg)
    os.makedirs(out_dir, exist_ok=True)
    if save_frames:
        os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'gt'), exist_ok=True)
    rend = InspectionRenderer(scn)
    cov = Coverage(scn)
    rng = stream(scn.seed, 'survey_noise')
    t0 = time.time()
    try:
        with open(os.path.join(out_dir, 'frames.jsonl'), 'w') as fj:
            for pose in plan:
                rend.set_pose(pose['rig_position'], pose['yaw'], pose['tilt_deg'])
                for cam in cams:
                    f = rend.capture(cam, rng)
                    new = cov.add(f) if cam == inspect_cam else 0
                    name = f'{pose["index"]:05d}_{cam}'
                    if save_frames:
                        save_frame(out_dir, name, f)
                    fj.write(json.dumps({
                        'frame': name, 'pose': pose, 'camera': cam,
                        'cam_pos': [round(float(v), 6) for v in f.cam_pos],
                        'cam_axes': [[round(float(v), 6) for v in row] for row in f.cam_mat],
                        'gain': round(f.gain, 4), 'new_cells': new,
                        'visible_items': visible_items(f),
                    }) + '\n')
    finally:
        rend.close()
    summary = cov.summary()
    summary.update({'frames': len(plan) * len(cams), 'poses': len(plan), 'cameras': cams,
                    'scenario_digest': scn.digest(), 'elapsed_s': round(time.time() - t0, 2)})
    with open(os.path.join(out_dir, 'coverage.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    Image.fromarray(cov.image()).save(os.path.join(out_dir, 'coverage.png'))
    log.info('survey: %d poses, coverage %.3f', len(plan), summary['coverage_fraction'])
    return summary
