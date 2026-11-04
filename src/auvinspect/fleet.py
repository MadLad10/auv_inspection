"""A fleet: several vehicles inspecting one hull together, on one clock.

The planned lanes are split into contiguous groups, one per vehicle (at most as many vehicles as
lanes). Each vehicle flies its group, every lane in the direction the whole plan flies it: the
vehicles start together at the same hull end and move in step, a group of lanes apart. Every lane
still ends past the hull ends, where the vehicle fixes its along-ship position from the sounder.
Each vehicle answers for the girth band between the midpoints to its neighbours' lanes: its
revisits target holes there only.

All vehicles advance one control tick at a time (mission.py's step generator). A vehicle's frames,
claims, events and metrics are its own, in v<k>/ under the fleet directory, laid out as a mission's
are. The renderer is shared, and one referee takes every vehicle's frames for the fleet's coverage.
Vehicle 0 keeps the mission seed, so a fleet of one is the single-vehicle mission; the others draw
theirs from it. The fleet referee reports the fleet's coverage and the union of the vehicles'
claims against the truth, the makespan (the last vehicle's finish), and the smallest distance
between two vehicles over the mission, with the ticks below fleet.min_separation_m.

There is no communication between the vehicles yet: each flies its share as assigned before the
dive, and a vehicle that aborts leaves its share undone.
"""
from __future__ import annotations

import json
import math
import os
import time

import numpy as np
from PIL import Image

from .coverage import Coverage
from .mission import _Mission, claim_check
from .observe import METRICS_FILE, write_metrics
from .planner import lane_girths
from .render import InspectionRenderer
from .rng import stream
from .scenario import Scenario


def lane_groups(n_lanes: int, vehicles: int) -> list[list[int]]:
    """Contiguous groups of lane indices, as equal as possible, one per vehicle."""
    if not 1 <= vehicles <= n_lanes:
        raise ValueError(f'a fleet of {vehicles} cannot share {n_lanes} lanes (1 to {n_lanes} vehicles)')
    return [[int(k) for k in g] for g in np.array_split(np.arange(n_lanes), vehicles)]


def band_edges(girths, groups: list[list[int]]) -> list[float]:
    """Girth boundaries between the vehicles' shares: midpoints between neighbouring groups' lanes,
    open at the ends."""
    inner = [(girths[a[-1]] + girths[b[0]]) / 2 for a, b in zip(groups, groups[1:])]
    return [-math.inf, *[float(x) for x in inner], math.inf]


def vehicle_seed(seed: int, k: int) -> int:
    return seed if k == 0 else int(stream(seed, f'fleet_vehicle_{k}').integers(2 ** 31 - 1))


def run(scn: Scenario, out_dir: str, vehicles: int | None = None, save_frames: bool = True, seed: int | None = None) -> dict:
    """Fly the fleet over the scenario and score it (see the module notes)."""
    cfg = scn.cfg
    n = int(vehicles or cfg['fleet']['vehicles'])
    seed = scn.seed if seed is None else int(seed)
    girths = lane_girths(scn.hull, cfg)
    groups = lane_groups(len(girths), n)
    edges = band_edges(girths, groups)
    os.makedirs(out_dir, exist_ok=True)
    t0 = time.time()
    rend = InspectionRenderer(scn)
    fleet_ref = Coverage(scn)
    agents: list[_Mission] = []
    try:
        for k in range(n):
            d = os.path.join(out_dir, f'v{k}')
            os.makedirs(d, exist_ok=True)
            if save_frames:
                for sub in ('frames', 'gt'):
                    os.makedirs(os.path.join(d, sub), exist_ok=True)
            agents.append(_Mission(scn, d, save_frames, vehicle_seed(seed, k), None, lanes=groups[k],
                                   band=(edges[k], edges[k + 1]), renderer=rend, fleet_referee=fleet_ref))
        gens = [a.run() for a in agents]
        summaries: list[dict | None] = [None] * n
        lim = cfg['fleet']['min_separation_m']
        sep = {'min_m': math.inf, 't': None, 'pair': None, 'ticks_below': 0, 'limit_m': lim}
        while any(s is None for s in summaries):
            for k, g in enumerate(gens):
                if summaries[k] is None:
                    try:
                        next(g)
                    except StopIteration as e:
                        summaries[k] = e.value
            if n > 1:
                pos = [a.vehicle.truth_pose()[0] for a in agents]
                below = False
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(pos[i] - pos[j]))
                        below |= dist < lim
                        if dist < sep['min_m']:
                            sep.update(min_m=round(dist, 4), t=round(max(a.vehicle.t for a in agents), 2), pair=[i, j])
                sep['ticks_below'] += int(below)
        for a, s in zip(agents, summaries):
            a.finish(s)
            a.write(s)
        if not math.isfinite(sep['min_m']):
            sep['min_m'] = None                          # a fleet of one
        summary = _summary(scn, agents, summaries, groups, edges, fleet_ref, sep, seed, time.time() - t0)
    finally:
        for a in agents:
            a.close()
        rend.close()
    _write(out_dir, summary, fleet_ref, agents)
    return summary


def _summary(scn, agents, summaries, groups, edges, fleet_ref, sep, seed, elapsed) -> dict:
    true = fleet_ref.covered()
    claimed = np.zeros_like(true)
    for a in agents:
        claimed |= a.onboard.claimed.covered()
    ref = fleet_ref.summary()
    return {**ref, 'scenario_digest': scn.digest(), 'seed': seed, 'vehicles': len(agents), 'elapsed_s': round(elapsed, 2),
            'makespan_s': max(s['sim_time_s'] for s in summaries),
            'claimed': {'fraction': round(float(claimed.mean()), 5),
                        'false_claim_fraction': round(float((claimed & ~true).sum() / max(claimed.sum(), 1)), 5)},
            'aborted': [k for k, s in enumerate(summaries) if s['aborted']],
            'separation': sep,
            'per_vehicle': [{'vehicle': k, 'seed': a.seed, 'lanes': groups[k],
                             'band_m': [None if not math.isfinite(e) else round(e, 4) for e in edges[k:k + 2]],
                             'sim_time_s': s['sim_time_s'], 'poses': s['poses'], 'aborted': s['aborted'],
                             'coverage_fraction': s['coverage_fraction'], 'claimed_fraction': s['claimed']['fraction'],
                             'holes_remaining': s['claimed']['holes_remaining'], 'battery_percent_final': s['battery_percent_final']}
                            for k, (a, s) in enumerate(zip(agents, summaries))]}


def _write(out_dir: str, summary: dict, fleet_ref: Coverage, agents):
    with open(os.path.join(out_dir, 'fleet.json'), 'w') as f:
        json.dump(summary, f, indent=1)
    Image.fromarray(fleet_ref.image()).save(os.path.join(out_dir, 'coverage.png'))
    claimed = np.zeros_like(fleet_ref.covered())
    for a in agents:
        claimed |= a.onboard.claimed.covered()
    Image.fromarray(claim_check(fleet_ref.covered(), claimed)).save(os.path.join(out_dir, 'claim_check.png'))
    lb = {'seed': summary['seed'], 'scenario': summary['scenario_digest'][:12]}
    sep = summary['separation']
    write_metrics(os.path.join(out_dir, METRICS_FILE), [
        ('auvinspect_fleet_vehicles', 'Vehicles in the fleet.', 'gauge', lb, summary['vehicles']),
        ('auvinspect_fleet_makespan_seconds', 'Simulated time until the last vehicle finished.', 'gauge', lb, summary['makespan_s']),
        ('auvinspect_fleet_aborted_vehicles', 'Vehicles that abandoned their share.', 'gauge', lb, len(summary['aborted'])),
        ('auvinspect_fleet_claimed_fraction', 'Share of the band the vehicles together claim.', 'gauge', lb, summary['claimed']['fraction']),
        ('auvinspect_referee_fleet_coverage_fraction', "Simulation only: the referee's coverage by all vehicles' frames.", 'gauge', lb,
         summary['coverage_fraction']),
        ('auvinspect_referee_fleet_false_claim_fraction', "Simulation only: share of the fleet's claimed cells not covered.", 'gauge', lb,
         summary['claimed']['false_claim_fraction']),
        ('auvinspect_referee_fleet_min_separation_m', 'Simulation only: smallest distance between two vehicles.', 'gauge', lb,
         sep['min_m']),
        ('auvinspect_referee_fleet_ticks_below_separation_total', 'Simulation only: ticks with two vehicles closer than the limit.',
         'counter', lb, sep['ticks_below'])])
