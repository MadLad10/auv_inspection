"""A fleet: several vehicles inspecting one hull together, on one clock, coordinated over an acoustic link.

**Sharing.** The planned lanes are split into contiguous groups, one per vehicle (at most as many
vehicles as lanes). Each vehicle flies its group, every lane in the direction the whole plan flies
it: the vehicles start together at the same hull end and move in step, a group of lanes apart.
Every lane still ends past the hull ends, where the vehicle fixes its along-ship position from the
sounder. Each vehicle answers for the girth band between the midpoints to its neighbours' lanes:
its revisits target holes there only.

**One clock.** All vehicles advance one control tick at a time (mission.py's step generator). A
vehicle's frames, claims, events and metrics are its own, in v<k>/ under the fleet directory, laid
out as a mission's are. The renderer is shared, and one referee takes every vehicle's frames for
the fleet's coverage. Vehicle 0 keeps the mission seed, so a fleet of one is the single-vehicle
mission; the others draw theirs from it. fleet.vehicle_faults adds faults to one vehicle only.

**Coordination** (fleet.coordination). The vehicles and a topside station on the ship's side share
the acoustic link of acoustics.py. A vehicle reports each lane it finishes, that it has finished
its work (it then holds its position), or that it is aborting (with the lanes it has done). The
topside hands an aborted vehicle's unfinished lanes, with their girth band, to a vehicle that has
finished its own (the nearest in girth), and releases the vehicles once no one is still working.
These rules keep vehicles apart without knowing where the others are:
- lanes are handed over only after an explicit abort, never because a vehicle is silent: the
  topside cannot know where a silent vehicle is. It flags a vehicle unheard for
  fleet.silent_after_s, keeps waiting for it until fleet.silent_margin_s past its planned finish,
  and its data is read when it is recovered;
- a vehicle takes over only once its own work is done, and only when no vehicle whose share lies
  between it and the work is still working, so its transit crosses no one. While it waits it holds
  in the middle of its stretch (sections), which no neighbour's work reaches, or where it is, in
  its own girth band (lanes). Revisits stay within the vehicle's own stretch;
- an aborting vehicle moves fleet.clearance_m farther off the hull before it climbs to the surface
  (the climb crosses the others' lanes), less off the hull's bottom if the seabed is near. The
  topside's transducer hangs farther out still.

**The fleet referee** reports the fleet's coverage, the union of the vehicles' claims against the
truth, the makespan (when the last vehicle finished its work), the lanes flown twice, the
hand-overs (orphaned, assigned, flown), the link's traffic, and the smallest distance between two
vehicles over the mission with the ticks below fleet.min_separation_m.

**One run for the rest of the pipeline.** The fleet directory holds the scenario's spec, so each
v<k>/ is a run as a mission's is, and merged/ is the whole fleet as one run: every vehicle's frame
records (names prefixed v<k>_, image and ground-truth files linked), the fleet referee's verdict and
the union of the claims. Detection, the map (whose multi-view confirmation now draws on every
vehicle's frames), the station check and the report run on it unchanged. The merged run counts as
aborted only when planned lanes were left unflown.
"""
from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
import shutil
import time

import numpy as np
from PIL import Image

from .acoustics import TOPSIDE, Link, Node
from .coverage import Coverage
from .guidance import LegFollower, Path
from .mission import _Mission, claim_check
from .observe import METRICS_FILE, Events, write_metrics
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


def lanes_band(girths, lanes: list[int]) -> list[float | None]:
    """The girth band of some lanes: midpoints to the lanes beside them (None: open)."""
    lo, hi = min(lanes), max(lanes)
    return [None if lo == 0 else round((girths[lo - 1] + girths[lo]) / 2, 4),
            None if hi == len(girths) - 1 else round((girths[hi] + girths[hi + 1]) / 2, 4)]


def shares(scn: Scenario, n: int) -> list[dict]:
    """Each vehicle's share of the plan: its lanes, the stretch of hull it flies them over (None: the
    whole length), the cells it answers for, and its centre (work is handed to the nearest)."""
    cfg, hull = scn.cfg, scn.hull
    girths = lane_girths(hull, cfg)
    if cfg['fleet']['allocation'] == 'lanes':
        groups = lane_groups(len(girths), n)
        edges = band_edges(girths, groups)
        return [{'lanes': g, 'x_range': None, 'centre': float(np.mean([girths[k] for k in g])),
                 'area': {'s': [None if not math.isfinite(e) else e for e in edges[k:k + 2]], 'x': None}}
                for k, g in enumerate(groups)]
    e = np.linspace(0.0, hull.length, n + 1)
    ov = cfg['fleet']['section_overlap_m']
    return [{'lanes': list(range(len(girths))), 'x_range': [float(e[k] - ov), float(e[k + 1] + ov)], 'centre': float((e[k] + e[k + 1]) / 2),
             'area': {'s': None, 'x': [None if k == 0 else float(e[k]), None if k == n - 1 else float(e[k + 1])]}}
            for k in range(n)]


def vehicle_seed(seed: int, k: int) -> int:
    return seed if k == 0 else int(stream(seed, f'fleet_vehicle_{k}').integers(2 ** 31 - 1))


def topside_position(scn: Scenario) -> np.ndarray:
    t = scn.cfg['acoustic']['topside']
    return scn.hull.point(t['along'] * scn.hull.length, t['girth_m']) + t['standoff_m'] * scn.hull.normal(t['girth_m'])


class Topside:
    """The coordinator on the ship's side (see the module notes). Lanes are (share, lane) pairs: the
    same lane of the plan is a different piece of work in each stretch of the hull."""

    def __init__(self, node: Node, plan: list[dict], girths, planned_end: list[float], lane_s: float, cfg: dict, events: Events,
                 position=None, rng=None):
        self.node, self.cfg, self.events, self.girths, self.lane_s, self.plan = node, cfg, events, girths, lane_s, plan
        self.lanes = {k: [(k, j) for j in sh['lanes']] for k, sh in enumerate(plan)}
        self.done: set[tuple[int, int]] = set()
        self.state = {k: 'working' for k in self.lanes}
        self.heard = {k: 0.0 for k in self.lanes}
        self.planned_end = list(planned_end)
        self.pending: list[dict] = []
        self.takeovers: list[dict] = []
        self.silent: set[int] = set()
        self.position, self.rng, self.fixes = position, rng, 0

    def on_packet(self, sender: str, pos, t_sent: float, t_heard: float):
        """The ship's acoustic positioning: where the sender was when it transmitted, sent back to it."""
        u = self.cfg['acoustic']['usbl']
        if not u['enabled'] or sender == TOPSIDE or self.position is None:
            return
        std = u['std_base_m'] + u['std_frac'] * float(np.linalg.norm(np.asarray(pos) - self.position))
        x, y = np.asarray(pos)[:2] + self.rng.normal(0.0, std, 2)
        self.node.send('fix', to=sender, reliable=False, t=round(t_sent, 3), x=float(x), y=float(y), std=round(std, 4))
        self.fixes += 1

    def step(self, t: float):
        f = self.cfg['fleet']
        for m in self.node.take(('status', 'lane_done', 'finished', 'abort')):
            k = int(m.sender[1:])
            self.heard[k] = t
            if k in self.silent:
                self.silent.discard(k)
                self.events.emit(t, 'vehicle_heard_again', 'info', vehicle=k)
            if m.kind == 'lane_done':
                self.done.update((m.body['share'], j) for j in m.body['lanes'])
            elif m.kind == 'abort':
                self.done.update(tuple(d) for d in m.body['lanes'])
            if m.kind == 'finished' and self.state[k] == 'working':
                self.state[k] = 'finished'
            elif m.kind == 'abort' and self.state[k] != 'aborted':
                self.state[k] = 'aborted'
                orphan = [d for d in self.lanes[k] if d not in self.done]
                self.events.emit(t, 'vehicle_aborted', 'critical', vehicle=k, lanes_left=[list(d) for d in orphan])
                for share in sorted({d[0] for d in orphan}):
                    self.pending.append({'share': share, 'lanes': [j for sh, j in orphan if sh == share], 'from': k, 't': round(t, 2)})
        for k, st in self.state.items():
            if st in ('working', 'finished') and k not in self.silent and t - self.heard[k] > f['silent_after_s']:
                self.silent.add(k)
                self.events.emit(t, 'vehicle_silent', 'warning', vehicle=k, unheard_s=round(t - self.heard[k], 1))
        free = [k for k, st in self.state.items() if st == 'finished' and k not in self.silent]
        for job in list(self.pending):
            share = self.plan[job['share']]
            clear = [v for v in free if not any(self.state[u] == 'working' for u in range(min(v, job['share']) + 1, max(v, job['share'])))]
            if not clear:
                continue                                 # wait: a vehicle still working lies between
            self.pending.remove(job)
            k = min(clear, key=lambda v: abs(self.plan[v]['centre'] - share['centre']))
            free.remove(k)
            area = {'s': lanes_band(self.girths, job['lanes']), 'x': share['area']['x']}
            self.node.send('assign', to=f'v{k}', lanes=job['lanes'], share=job['share'], x_range=share['x_range'], area=area)
            self.lanes[k] += [(job['share'], j) for j in job['lanes']]
            self.state[k] = 'working'
            self.planned_end[k] = t + self.lane_s * len(job['lanes'])
            self.takeovers.append({'share': job['share'], 'lanes': job['lanes'], 'from': job['from'], 'to': k,
                                   'orphaned_t': job['t'], 'assigned_t': round(t, 2)})
            self.events.emit(t, 'lanes_reassigned', 'warning', share=job['share'], lanes=job['lanes'], from_vehicle=job['from'], to_vehicle=k)
        waiting = [k for k, st in self.state.items()
                   if st == 'working' and (k not in self.silent or t <= self.planned_end[k] + f['silent_margin_s'])]
        if not waiting:
            for k, st in self.state.items():
                if st == 'finished':
                    self.node.send('release', to=f'v{k}')
                    self.state[k] = 'released'
                    self.events.emit(t, 'vehicle_released', 'info', vehicle=k)


def run(scn: Scenario, out_dir: str, vehicles: int | None = None, save_frames: bool = True, seed: int | None = None) -> dict:
    """Fly the fleet over the scenario and score it (see the module notes)."""
    cfg = scn.cfg
    f = cfg['fleet']
    n = int(vehicles or f['vehicles'])
    seed = scn.seed if seed is None else int(seed)
    girths = lane_girths(scn.hull, cfg)
    if f['allocation'] == 'lanes':
        lane_groups(len(girths), n)                      # refuses more vehicles than lanes
    plan = shares(scn, n)
    unknown = set(map(int, f['vehicle_faults'])) - set(range(n))
    if unknown:
        raise ValueError(f'fleet.vehicle_faults names vehicles {sorted(unknown)}; the fleet has 0 to {n - 1}')
    os.makedirs(out_dir, exist_ok=True)
    with open(os.path.join(out_dir, 'scenario.json'), 'w') as fh:     # the runs below sit in a scenario directory
        json.dump({**scn.spec(), 'digest': scn.digest()}, fh, indent=1, sort_keys=True)
    t0 = time.time()
    rend = InspectionRenderer(scn)
    fleet_ref = Coverage(scn)
    agents: list[_Mission] = []
    topside = link = None
    try:
        for k in range(n):
            d = os.path.join(out_dir, f'v{k}')
            os.makedirs(d, exist_ok=True)
            if save_frames:
                for sub in ('frames', 'gt'):
                    os.makedirs(os.path.join(d, sub), exist_ok=True)
            own = f['vehicle_faults'].get(k, f['vehicle_faults'].get(str(k), []))
            vcfg = copy.deepcopy(cfg)
            vcfg['faults'] = list(cfg['faults']) + list(own)
            sh = plan[k]
            agents.append(_Mission(dataclasses.replace(scn, cfg=vcfg), d, save_frames, vehicle_seed(seed, k), None, lanes=sh['lanes'],
                                   area=sh['area'], renderer=rend, fleet_referee=fleet_ref, x_range=sh['x_range'], share=k))
        if f['coordination'] and n > 1:                # a fleet of one has no one to coordinate with
            top = Node(TOPSIDE)
            pos = topside_position(scn)
            top.position = lambda: pos
            top.status = lambda: {}                      # a heartbeat: the vehicles know the topside is there
            nodes = [top]
            for k, a in enumerate(agents):
                node = Node(f'v{k}')
                node.position = (lambda a=a: a.vehicle.truth_pose()[0])
                node.dead = (lambda t, a=a: a.vehicle.faults.get('modem_dead', t) is not None)
                node.status = a.link_status
                a.link = node
                nodes.append(node)
            link = Link(nodes, cfg, stream(seed, 'acoustic_link'))
            planned = [sum(LegFollower(Path(leg, scn.hull, cfg), cfg).nominal_duration() for leg in a.legs) for a in agents]
            lane_s = max(p / len(sh['lanes']) for p, sh in zip(planned, plan))
            os.makedirs(os.path.join(out_dir, TOPSIDE), exist_ok=True)
            topside = Topside(top, plan, girths, planned, lane_s, cfg, Events(os.path.join(out_dir, TOPSIDE), cfg),
                              position=pos, rng=stream(seed, 'usbl'))
            top.on_packet = topside.on_packet
        gens = [a.run() for a in agents]
        summaries: list[dict | None] = [None] * n
        lim = f['min_separation_m']
        sep = {'min_m': math.inf, 't': None, 'pair': None, 'ticks_below': 0, 'limit_m': lim}
        while any(s is None for s in summaries):
            for k, g in enumerate(gens):
                if summaries[k] is None:
                    try:
                        next(g)
                    except StopIteration as e:
                        summaries[k] = e.value
            t = max(a.vehicle.t for a in agents)
            if link is not None:
                link.step(t)
                topside.step(t)
            if n > 1:
                pos = [a.vehicle.truth_pose()[0] for a in agents]
                below = False
                for i in range(n):
                    for j in range(i + 1, n):
                        dist = float(np.linalg.norm(pos[i] - pos[j]))
                        below |= dist < lim
                        if dist < sep['min_m']:
                            sep.update(min_m=round(dist, 4), t=round(t, 2), pair=[i, j])
                sep['ticks_below'] += int(below)
        for a, s in zip(agents, summaries):
            a.finish(s)
            a.write(s)
        if not math.isfinite(sep['min_m']):
            sep['min_m'] = None                          # a fleet of one
        summary = _summary(scn, agents, summaries, plan, fleet_ref, sep, seed, time.time() - t0, topside, link)
    finally:
        for a in agents:
            a.close()
        if topside is not None:
            topside.events.close()
        rend.close()
    _write(out_dir, summary, fleet_ref, agents)
    _merge(out_dir, summary, summaries, agents, fleet_ref)
    return summary


def _merge(out_dir: str, summary: dict, summaries: list[dict], agents, fleet_ref: Coverage):
    """The fleet as one run, merged/ (see the module notes)."""
    m = os.path.join(out_dir, 'merged')
    if os.path.isdir(m):
        shutil.rmtree(m)
    for sub in ('frames', 'gt'):
        os.makedirs(os.path.join(m, sub))
    with open(os.path.join(m, 'frames.jsonl'), 'w') as fo:
        for k, a in enumerate(agents):
            for line in open(os.path.join(a.out_dir, 'frames.jsonl')):
                rec = json.loads(line)
                name = rec['frame']
                rec.update(frame=f'v{k}_{name}', vehicle=k)
                fo.write(json.dumps(rec) + '\n')
                for sub, ext in (('frames', '.png'), ('gt', '.npz')):
                    src = os.path.join(a.out_dir, sub, name + ext)
                    if os.path.exists(src):
                        dst = os.path.join(m, sub, rec['frame'] + ext)
                        os.symlink(os.path.relpath(src, os.path.dirname(dst)), dst)
    claimed = np.zeros_like(fleet_ref.covered())
    for a in agents:
        claimed |= a.onboard.claimed.covered()
    frames: dict[str, int] = {}
    for s in summaries:
        for k, v in s['claimed']['frames'].items():
            frames[k] = frames.get(k, 0) + v
    left = summary['lanes']['not_flown']
    cov = {k: summary[k] for k in summary if k in ('coverage_fraction', 'by_class', 'items', 'inspect_girth_m', 'cell_m', 'cells')}
    cov.update({'scenario_digest': summary['scenario_digest'], 'seed': summary['seed'], 'fleet': True, 'vehicles': summary['vehicles'],
                'aborted': f'planned lanes not flown: {left}' if left else None,
                'sim_time_s': summary['makespan_s'], 'poses': sum(s['poses'] for s in summaries),
                'claimed': {**summary['claimed'], 'holes_remaining': sum(s['claimed']['holes_remaining'] for s in summaries), 'frames': frames},
                'contacts': {'ticks': sum(s['contacts']['ticks'] for s in summaries)},
                'navigation': {'health': [{**h, 'vehicle': k} for k, s in enumerate(summaries) for h in s['navigation']['health']]}})
    with open(os.path.join(m, 'coverage.json'), 'w') as fh:
        json.dump(cov, fh, indent=1)
    Image.fromarray(fleet_ref.image()).save(os.path.join(m, 'coverage.png'))
    Image.fromarray(claimed.astype(np.uint8) * 255).save(os.path.join(m, 'claimed.png'))


def _summary(scn, agents, summaries, plan, fleet_ref, sep, seed, elapsed, topside, link) -> dict:
    true = fleet_ref.covered()
    claimed = np.zeros_like(true)
    for a in agents:
        claimed |= a.onboard.claimed.covered()
    by_lane = scn.cfg['fleet']['allocation'] == 'lanes'
    key = (lambda share, lane: lane) if by_lane else (lambda share, lane: f'{share}:{lane}')    # a lane, or a lane in a stretch
    flown: dict = {key(k, j): [] for k, sh in enumerate(plan) for j in sh['lanes']}
    for k, a in enumerate(agents):
        for share, lane in a.lanes_done:
            flown[key(share, lane)].append(k)
    ref = fleet_ref.summary()
    out = {**ref, 'scenario_digest': scn.digest(), 'seed': seed, 'vehicles': len(agents), 'elapsed_s': round(elapsed, 2),
           'coordination': topside is not None,
           'makespan_s': round(max(a.work_end_t for a in agents), 2),
           'end_s': max(s['sim_time_s'] for s in summaries),
           'claimed': {'fraction': round(float(claimed.mean()), 5),
                       'false_claim_fraction': round(float((claimed & ~true).sum() / max(claimed.sum(), 1)), 5)},
           'aborted': [k for k, s in enumerate(summaries) if s['aborted']],
           'lanes': {'flown_by': {str(k): v for k, v in flown.items()}, 'not_flown': [k for k, v in flown.items() if not v],
                     'flown_twice': [k for k, v in flown.items() if len(v) > 1]},
           'separation': sep,
           'allocation': scn.cfg['fleet']['allocation'],
           'per_vehicle': [{'vehicle': k, 'seed': a.seed, 'lanes': plan[k]['lanes'], 'x_range_m': plan[k]['x_range'],
                            'lanes_done': [key(sh, j) for sh, j in a.lanes_done],
                            'band_m': [None if e is None else round(e, 4) for e in (plan[k]['area']['s'] or [None, None])],
                            'ship_fixes': a.onboard.ship_fix_stats,
                            'sim_time_s': s['sim_time_s'], 'work_end_s': round(a.work_end_t, 2), 'poses': s['poses'],
                            'aborted': s['aborted'], 'coverage_fraction': s['coverage_fraction'],
                            'claimed_fraction': s['claimed']['fraction'], 'holes_remaining': s['claimed']['holes_remaining'],
                            'battery_percent_final': s['battery_percent_final']}
                           for k, (a, s) in enumerate(zip(agents, summaries))]}
    if topside is not None:
        done_t = {}
        for k, a in enumerate(agents):
            for leg in a.leg_log:
                if leg['phase'] == 'takeover' and leg['kind'] == 'lane' and leg['outcome'] == 'done':
                    done_t[(leg['share'], leg['lane'])] = leg['t'][1]
        for h in topside.takeovers:
            h['flown_t'] = max((done_t[(h['share'], lane)] for lane in h['lanes'] if (h['share'], lane) in done_t), default=None)
        out['handovers'] = topside.takeovers
        out['unassigned_lanes'] = sorted({key(job['share'], lane) for job in topside.pending for lane in job['lanes']}, key=str)
        out['ship_fixes_sent'] = topside.fixes
        out['silent_vehicles'] = sorted(topside.silent)
        out['link'] = link.summary()
    return out


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
    samples = [
        ('auvinspect_fleet_vehicles', 'Vehicles in the fleet.', 'gauge', lb, summary['vehicles']),
        ('auvinspect_fleet_makespan_seconds', 'Simulated time until the last vehicle finished its work.', 'gauge', lb, summary['makespan_s']),
        ('auvinspect_fleet_aborted_vehicles', 'Vehicles that abandoned their share.', 'gauge', lb, len(summary['aborted'])),
        ('auvinspect_fleet_claimed_fraction', 'Share of the band the vehicles together claim.', 'gauge', lb, summary['claimed']['fraction']),
        ('auvinspect_fleet_lanes_not_flown', 'Planned lanes no vehicle finished.', 'gauge', lb, len(summary['lanes']['not_flown'])),
        ('auvinspect_fleet_lanes_flown_twice', 'Planned lanes finished by more than one vehicle.', 'gauge', lb,
         len(summary['lanes']['flown_twice'])),
        ('auvinspect_referee_fleet_coverage_fraction', "Simulation only: the referee's coverage by all vehicles' frames.", 'gauge', lb,
         summary['coverage_fraction']),
        ('auvinspect_referee_fleet_false_claim_fraction', "Simulation only: share of the fleet's claimed cells not covered.", 'gauge', lb,
         summary['claimed']['false_claim_fraction']),
        ('auvinspect_referee_fleet_min_separation_m', 'Simulation only: smallest distance between two vehicles.', 'gauge', lb, sep['min_m']),
        ('auvinspect_referee_fleet_ticks_below_separation_total', 'Simulation only: ticks with two vehicles closer than the limit.',
         'counter', lb, sep['ticks_below'])]
    if summary['coordination']:
        samples.append(('auvinspect_fleet_handovers', 'Groups of lanes handed from an aborted vehicle to another.', 'gauge', lb,
                        len(summary['handovers'])))
        for name, st in summary['link']['nodes'].items():
            nl = {**lb, 'node': name}
            samples += [('auvinspect_link_packets_total', 'Acoustic packets sent.', 'counter', nl, st['packets']),
                        ('auvinspect_link_bytes_total', 'Acoustic payload bytes sent.', 'counter', nl, st['bytes']),
                        ('auvinspect_link_resent_total', 'Reliable messages sent again.', 'counter', nl, st['resent']),
                        ('auvinspect_link_given_up_total', 'Reliable messages abandoned after the last retry.', 'counter', nl, st['given_up']),
                        ('auvinspect_link_packets_lost_total', 'Packets this node failed to hear.', 'counter', nl, st['lost'])]
    write_metrics(os.path.join(out_dir, METRICS_FILE), samples)
