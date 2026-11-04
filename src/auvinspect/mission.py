"""Fly a survey with the vehicle in the loop, and score it.

Each control tick (vehicle.control_dt_s):
  1. the onboard software reads the sensors and updates its estimate (onboard.py);
  2. frames whose position the vehicle believes it has reached are captured. The renderer uses the
     true pose from the dynamics, roll and pitch included, and the water and camera faults of the
     moment. The referee scores each frame; the onboard checks it and claims its footprint;
  3. guidance and control (guidance.py) turn the estimate into thruster requests;
  4. the dynamics advance one tick.
After the planned survey the vehicle plans revisits of the holes in its claimed coverage, for up to
onboard.replan_rounds rounds. When its battery monitor reads below onboard.battery_reserve_pct, the
vehicle abandons the mission and climbs along the hull at the standoff to the waterline (a path that
keeps clear of the hull, even from under the bilge). The referee then reports the true coverage, where the claim departs
from it, the navigation error, the clearance to the hull and the fault log. As it flies, the mission
writes the operator's event stream (events.jsonl: only what the vehicle knows), and at the end its
metrics (metrics.prom), as observe.py describes.
"""
from __future__ import annotations

import json
import logging
import math
import os
import time

import numpy as np
from PIL import Image

from .camera import axes
from .coverage import Coverage
from .acoustics import TOPSIDE
from .guidance import Controller, Leg, LegFollower, Path, lane_ends, survey_legs, takeover_legs
from .observe import METRICS_FILE, Events, mission_samples, write_metrics
from .onboard import Onboard
from .render import InspectionRenderer
from .rng import stream
from .scenario import Scenario
from .survey import save_frame, visible_items
from .vehicle import Vehicle

log = logging.getLogger(__name__)

CLAIM_COLOURS = {'both': (60, 180, 75), 'false_claim': (230, 25, 75), 'unclaimed': (0, 130, 200), 'neither': (0, 0, 0)}


def _r(v, nd: int = 4) -> list:
    return [round(float(u), nd) for u in v]


class _Mission:
    """One vehicle's mission. `run()` is a generator that advances one control tick per step, so a
    fleet (fleet.py) can fly several vehicles on one clock; a fleet member flies `lanes` of the plan
    over `x_range` of the hull (its share, numbered `share`), answers for the cells of `area`
    ({'s': girth band, 'x': stretch}), and shares the fleet's renderer and coverage referee."""

    def __init__(self, scn: Scenario, out_dir: str, save_frames: bool, seed: int, cameras, lanes: list[int] | None = None,
                 area: dict | None = None, renderer=None, fleet_referee=None, x_range: tuple[float, float] | None = None,
                 share: int | None = None):
        self.scn, self.cfg, self.hull = scn, scn.cfg, scn.hull
        self.out_dir, self.save_frames, self.seed = out_dir, save_frames, seed
        self.inspect_cam = self.cfg['survey']['camera']
        self.cams = list(dict.fromkeys([self.inspect_cam] + list(cameras or self.cfg['survey']['record_cameras'])))
        self.legs = survey_legs(self.hull, self.cfg, lanes, x_range, share)
        first = Path(self.legs[0], self.hull, self.cfg)
        self.vehicle = Vehicle(self.hull, self.cfg, first.p[0], seed)
        self.vehicle.tilt_deg = self.vehicle.tilt_cmd = float(first.tilt[0])
        rng = stream(seed, 'nav_initial_fix')
        fix = self.cfg['navigation']['initial_fix_std']
        self.onboard = Onboard(self.hull, self.cfg, first.p[0] + rng.normal(0, fix['pos_m'], 3),
                               rng.normal(0, math.radians(fix['yaw_deg'])))
        if area is not None:
            self.onboard.assign_area(**area)
        if x_range is not None:
            self.onboard.lane_limits = lane_ends(self.hull, self.cfg, x_range)
        self.ctrl = Controller(self.cfg, self.vehicle.model)
        self.referee = Coverage(scn)
        self.fleet_referee = fleet_referee
        self.own_renderer = renderer is None
        self.rend = renderer or InspectionRenderer(scn)
        self.noise = stream(seed, 'mission_noise')
        self.dt = self.vehicle.dt
        # roll pitches the inspection view by the projection of the camera's tilt axis on the vehicle x axis
        self.roll_sign = float(axes(self.vehicle.look)[0, 0]) if self.cfg['vehicle']['control']['tilt_compensates_roll'] else 0.0
        self.n_frames = 0
        self.ticks: list[list[float]] = []
        self.frame_errors: list[list[float]] = []
        self.claims = {'claimed': 0, 'uncertain': 0, 'black': 0, 'below_limits': 0}
        self.leg_log: list[dict] = []
        self.aborted = None
        self.surfacing = False
        self.events = Events(out_dir, self.cfg)
        self.link = None                  # a fleet member's modem (acoustics.Node), set by fleet.py
        self.lanes_done: list[tuple[int | None, int]] = []      # (share, lane) of each lane finished
        self._k = 0                       # revisit rounds so far
        self.work_end_t = 0.0

    # ------------------------------------------------------------------ one leg
    def fly(self, leg, leg_index: int, fj) -> str:
        path = Path(leg, self.hull, self.cfg)
        fol = LegFollower(path, self.cfg)
        v = self.cfg['vehicle']
        t_end = self.vehicle.t + v['leg_timeout_factor'] * max(fol.nominal_duration(), v['lead_m'] / leg.speed)
        while True:
            readings = self.vehicle.readings()
            self.onboard.update(readings, self.vehicle.t)
            self.take_fixes()
            self.events.tick(self.vehicle.t, readings, self.onboard)
            if not self.surfacing and self.onboard.battery_low():
                self.aborted = f'battery reserve ({self.onboard.battery:.0f}%) at t={self.vehicle.t:.2f} s'
                self.events.emit(self.vehicle.t, 'abort', 'critical', reason='battery reserve', battery_pct=round(self.onboard.battery, 1),
                                 action='climbing along the hull to the surface')
                if self.link is not None:
                    self.link.send('abort', to=TOPSIDE, lanes=[list(d) for d in self.lanes_done])
                return 'aborted'
            if self.events.status_due(self.vehicle.t):
                self.status(leg, leg_index)
            p_est = self.onboard.position()
            if fol.searching:
                if not self.onboard.echo:
                    return 'done'                 # the beam passed the hull end: the along-ship fix is in
                if fol.at_limit(p_est):
                    return 'edge_not_found'
            elif fol.at_limit(p_est):
                if not (leg.edge_search and self.onboard.echo):
                    return 'done'
                fol.search()
            if self.vehicle.t >= t_end:
                return 'timeout'
            p_ref, v_ref, tilt_ref = fol.step(p_est, self.dt)
            est = self.onboard.state()
            self.vehicle.tilt_cmd = tilt_ref - self.roll_sign * math.degrees(est['roll'])
            for _ in fol.due_captures():
                self.capture(leg, leg_index, fj)
            thrust = self.ctrl.command(est, p_ref, v_ref, self.dt)
            self.vehicle.step(thrust)
            self.log_tick(p_est, est, p_ref, thrust)
            yield
            if self.vehicle.t >= self.cfg['mission']['max_seconds']:
                self.aborted = 'time limit'
                self.events.emit(self.vehicle.t, 'abort', 'critical', reason='time limit')
                return 'aborted'

    def take_fixes(self):
        """Fuse the position fixes the topside sent from the ship's acoustic positioning."""
        if self.link is None:
            return
        for m in self.link.take(('fix',)):
            if not self.ticks:
                continue
            tk = np.asarray(self.ticks)
            est_then = (float(np.interp(m.body['t'], tk[:, 0], tk[:, 7])), float(np.interp(m.body['t'], tk[:, 0], tk[:, 8])))
            self.onboard.ship_fix(m.body, est_then)

    def status(self, leg, leg_index: int):
        """The heartbeat: where the vehicle believes it is, how sure, and how far along."""
        x, s = self.onboard.camera_hull_xs()
        sa, sg = self.onboard.hull_std(s)
        self.events.emit(self.vehicle.t, 'status', 'debug', phase=leg.phase, leg=leg_index, lane=leg.lane, x_m=round(x, 2),
                         s_m=round(s, 2), std_along_m=round(sa, 3), std_girth_m=round(sg, 3),
                         battery_pct=round(self.onboard.battery, 1), frames=dict(self.claims),
                         claimed_fraction=round(self.onboard.claimed.fraction(), 4))

    def log_tick(self, p_est, est, p_ref, thrust):
        pos, q = self.vehicle.truth_pose()
        rpy = self.vehicle.truth_rpy()
        sa, sg = self.onboard.hull_std(float(self.hull.inverse(p_est)[1]))
        self.ticks.append([self.vehicle.t, *pos, *rpy, *p_est, est['yaw'], *p_ref, *thrust, sa, sg, self.vehicle.tilt_deg,
                           self.vehicle.clearance()])

    # ------------------------------------------------------------------ frames
    def capture(self, leg, leg_index: int, fj):
        v = self.vehicle
        pos, q = v.truth_pose()
        self.rend.set_pose(pos, attitude=q, tilt_deg={self.inspect_cam: v.tilt_deg})
        water, black, motion = v.water(), v.camera_blackout(), v.camera_motion()
        idx = self.n_frames
        self.n_frames += 1
        for cam in self.cams:
            f = self.rend.capture(cam, self.noise, water=water, blackout=black and cam == self.inspect_cam, motion=motion)
            name = f'{idx:05d}_{cam}'
            rec = {'frame': name, 'camera': cam, 't': round(v.t, 3), 'phase': leg.phase, 'leg': leg_index, 'lane': leg.lane,
                   'cam_pos': _r(f.cam_pos, 6), 'cam_axes': [_r(row, 6) for row in f.cam_mat],
                   'tilt_deg': round(v.tilt_deg, 3), 'exposure_ms': round(f.exposure_s * 1000, 4), 'gain': round(f.gain, 4),
                   'blur_px': round(f.blur_px, 3), 'signal': f.signal,
                   'active_faults': v.faults.active_types(v.t), 'visible_items': visible_items(f)}
            if cam == self.inspect_cam:
                rec['new_cells'] = self.referee.add(f)
                if self.fleet_referee is not None:
                    self.fleet_referee.add(f)
                ob = self.onboard.process_frame(f.rgb, f.exposure_s, f.gain)
                self.claims[ob['claim']] += 1
                self.events.frame(v.t, ob['claim'])
                rec['onboard'] = ob
                rec['cam_error_hull_m'] = self._hull_error(np.asarray(ob['cam_pos_est']), f.cam_pos, f.cam_mat)
                self.frame_errors.append(rec['cam_error_hull_m'])
            if self.save_frames:
                save_frame(self.out_dir, name, f, self.referee.quality(f))
            fj.write(json.dumps(rec) + '\n')

    def _hull_error(self, c_est, c_true, A) -> list[float]:
        """Estimated minus true camera position, along the ship / along the girth / along the normal."""
        h = self.hull
        r = float(h.raycast(c_true, -A[:, 2]))
        s = float(h.inverse(c_true + (r if math.isfinite(r) else 0.0) * -A[:, 2])[1])
        phi = float(h.turn_angle(s))
        e = c_est - c_true
        return _r([e[0], e @ np.array([0.0, math.sin(phi), -math.cos(phi)]), e @ h.normal(s)])

    # ------------------------------------------------------------------ the whole mission
    def run(self):
        """Generator: one step per control tick; its return value is the summary. A fleet member with a
        link (acoustics.Node) reports its lanes, then holds for lanes handed over until released."""
        t0 = time.time()
        self.rounds = []
        self.events.emit(0.0, 'mission_started', 'info', seed=self.seed, scenario=self.scn.digest()[:12], legs=len(self.legs),
                         lanes=sum(1 for leg in self.legs if leg.kind == 'lane'), cameras=self.cams)
        with open(os.path.join(self.out_dir, 'frames.jsonl'), 'w') as fj:
            self.fj = fj
            yield from self._work(self.legs)
            self.work_end_t = self.vehicle.t
            while self.link is not None and not self.aborted:
                self.link.send('finished', to=TOPSIDE)
                yield from self._go_wait()
                legs = yield from self.hold()
                if not legs:
                    break
                yield from self._work(legs)
                self.work_end_t = self.vehicle.t
        remaining = self.onboard.holes()
        return self.summary(self.rounds, remaining, time.time() - t0)

    def _work(self, legs):
        """Fly the legs, then revisit the holes they leave in the vehicle's band, up to
        onboard.replan_rounds rounds."""
        holes, r = [], 0
        while True:
            n0 = self.n_frames
            outcome = {'done': 0, 'timeout': 0, 'edge_not_found': 0}
            for i, leg in enumerate(legs):
                t_start = self.vehicle.t
                res = yield from self.fly(leg, i, self.fj)
                self.leg_log.append({'phase': leg.phase, 'kind': leg.kind, 'lane': leg.lane, 'share': leg.share, 'outcome': res,
                                     'from_xs': _r([leg.x[0], leg.s[0]], 3), 'to_xs': _r([leg.x[-1], leg.s[-1]], 3),
                                     't': [round(t_start, 2), round(self.vehicle.t, 2)]})
                self.events.emit(self.vehicle.t, 'leg_finished', 'info' if res == 'done' else 'warning', phase=leg.phase,
                                 kind=leg.kind, lane=leg.lane, outcome=res, seconds=round(self.vehicle.t - t_start, 2))
                if res == 'aborted':
                    if self.aborted.startswith('battery'):
                        yield from self.surface(self.fj)
                    break
                if res == 'done' and leg.kind == 'lane' and leg.lane is not None:
                    self.lanes_done.append((leg.share, leg.lane))
                    if self.link is not None:
                        self.link.send('lane_done', to=TOPSIDE, lanes=[leg.lane], share=leg.share)
                outcome[res] += 1
            self.rounds.append({'phase': legs[0].phase if legs else f'revisit_{self._k}', 'legs': len(legs),
                                'legs_timed_out': outcome['timeout'], 'edges_not_found': outcome['edge_not_found'],
                                'frames': self.n_frames - n0, 'holes_targeted': len(holes),
                                'coverage_fraction_after': round(self.referee.fraction(), 5),
                                'claimed_fraction_after': round(self.onboard.claimed.fraction(), 5)})
            if self.aborted or r >= self.cfg['onboard']['replan_rounds']:
                return
            r += 1
            self._k += 1
            k = self._k
            legs, holes = self.onboard.revisit_legs(k)
            if legs:
                self.events.emit(self.vehicle.t, 'revisit_planned', 'info', round=k, holes=len(holes), legs=len(legs))
            elif holes:
                self.events.emit(self.vehicle.t, 'revisit_skipped', 'warning', round=k, holes=len(holes),
                                 reasons=sorted({h.get('skipped', '') for h in holes}))
            if not legs:
                if holes:
                    self.rounds.append({'phase': f'revisit_{k}', 'legs': 0, 'holes_targeted': 0,
                                        'skipped': sorted({h.get('skipped', '') for h in holes})})
                return

    def _go_wait(self):
        """Where a fleet member waits: the middle of its stretch of the hull, which no neighbour's work
        reaches (its last lane ended where a neighbour may still revisit); with a share of whole lanes,
        where it is, in its own girth band."""
        if self.onboard.lane_limits is None or self.aborted:
            return
        x, s = self.onboard.camera_hull_xs()
        xc = sum(self.onboard.lane_limits) / 2
        leg = Leg('transit', 'hold', [x, xc], [s, s], self.cfg['vehicle']['transit_speed_mps'])
        t_start = self.vehicle.t
        res = yield from self.fly(leg, 0, self.fj)
        self.leg_log.append({'phase': 'hold', 'kind': 'transit', 'lane': None, 'share': None, 'outcome': res,
                             'from_xs': _r([x, s], 3), 'to_xs': _r([xc, s], 3), 't': [round(t_start, 2), round(self.vehicle.t, 2)]})

    def hold(self):
        """Hold the current position until the topside hands over lanes (their legs are returned) or
        releases the vehicle (None), at most fleet.hold_max_s; or until the topside, which broadcasts
        its status every turn of the channel, has been unheard for fleet.silent_after_s (link lost)."""
        p_hold = self.onboard.position()
        t_end = self.vehicle.t + self.cfg['fleet']['hold_max_s']
        self.events.emit(self.vehicle.t, 'holding', 'info', lanes_done=list(self.lanes_done))
        zero = np.zeros(3)
        while True:
            got = self.link.take(('assign', 'release'))
            jobs = [m.body for m in got if m.kind == 'assign']
            if jobs:
                legs = []
                x, s = self.onboard.camera_hull_xs()
                for job in jobs:
                    self.onboard.extend_area(**job['area'])
                    self.events.emit(self.vehicle.t, 'lanes_assigned', 'info', share=job['share'], lanes=job['lanes'])
                    more = takeover_legs(self.hull, self.cfg, job['lanes'], x, s, x_range=job['x_range'], share=job['share'])
                    legs += more
                    x, s = more[-1].x[-1], more[-1].s[-1]
                return legs
            if got:
                self.events.emit(self.vehicle.t, 'released', 'info')
                return None
            if self.vehicle.t >= t_end:
                self.events.emit(self.vehicle.t, 'hold_timeout', 'warning', seconds=self.cfg['fleet']['hold_max_s'])
                return None
            unheard = self.vehicle.t - self.link.last_heard.get(TOPSIDE, 0.0)
            if unheard > self.cfg['fleet']['silent_after_s']:
                self.events.emit(self.vehicle.t, 'link_lost', 'warning', unheard_s=round(unheard, 1),
                                 action='ending the mission where it is; the data is read at recovery')
                return None
            readings = self.vehicle.readings()
            self.onboard.update(readings, self.vehicle.t)
            self.take_fixes()
            self.events.tick(self.vehicle.t, readings, self.onboard)
            if self.onboard.battery_low():
                self.aborted = f'battery reserve ({self.onboard.battery:.0f}%) at t={self.vehicle.t:.2f} s'
                self.events.emit(self.vehicle.t, 'abort', 'critical', reason='battery reserve', battery_pct=round(self.onboard.battery, 1))
                self.link.send('abort', to=TOPSIDE, lanes=[list(d) for d in self.lanes_done])
                return None
            est = self.onboard.state()
            thrust = self.ctrl.command(est, p_hold, zero, self.dt)
            self.vehicle.step(thrust)
            self.log_tick(self.onboard.position(), est, p_hold, thrust)
            yield

    def link_status(self) -> dict:
        x, s = self.onboard.camera_hull_xs()
        return {'x': round(x, 1), 's': round(s, 2), 'battery': round(self.onboard.battery), 'lanes_done': len(self.lanes_done)}

    def surface(self, fj):
        """Abort: climb along the hull to the waterline, taking no frames. A fleet member first moves
        fleet.clearance_m farther off the hull and climbs there: the others fly at the standoff, and the
        climb crosses their lanes. Off the hull's bottom that is towards the seabed, so no farther than
        its DVL altitude less fleet.seabed_margin_m allows."""
        self.surfacing = True
        x, s = self.onboard.camera_hull_xs()
        speed = self.cfg['vehicle']['transit_speed_mps']
        clear = self.cfg['fleet']['clearance_m'] if self.link is not None else 0.0
        n_z = float(self.hull.normal(s)[2])
        alt = (self.vehicle.readings().get('dvl') or {}).get('altitude')
        if clear > 0 and n_z < 0 and alt is not None:          # moving off the hull's bottom moves towards the seabed
            clear = min(clear, max(0.0, (alt - self.cfg['fleet']['seabed_margin_m']) / -n_z))
        if clear > 0:
            d = self.cfg['survey']['standoff_m']
            legs = [Leg('transit', 'abort', [x, x], [s, s], speed, standoff=[d, d + clear]),
                    Leg('transit', 'abort', [x, x], [s, 0.0], speed, standoff=[d + clear] * 2)]
        else:
            legs = [Leg('transit', 'abort', [x, x], [s, 0.0], speed)]
        for lg in legs:
            t_start = self.vehicle.t
            res = yield from self.fly(lg, 0, fj)
            self.leg_log.append({'phase': 'abort', 'kind': 'transit', 'lane': None, 'outcome': res,
                                 'from_xs': _r([lg.x[0], lg.s[0]], 3), 'to_xs': _r([lg.x[-1], lg.s[-1]], 3),
                                 't': [round(t_start, 2), round(self.vehicle.t, 2)]})

    def summary(self, rounds, remaining, elapsed) -> dict:
        ref = self.referee.summary()
        true, claimed = self.referee.covered(), self.onboard.claimed.covered()
        tk = np.asarray(self.ticks) if self.ticks else np.zeros((0, 24))
        herr = np.hypot(tk[:, 7] - tk[:, 1], tk[:, 8] - tk[:, 2]) if len(tk) else np.zeros(1)
        fe = np.asarray(self.frame_errors) if self.frame_errors else np.zeros((1, 3))
        ref.update({
            'scenario_digest': self.scn.digest(), 'seed': self.seed, 'aborted': self.aborted,
            'sim_time_s': round(self.vehicle.t, 2), 'elapsed_s': round(elapsed, 2),
            'frames': self.n_frames * len(self.cams), 'poses': self.n_frames, 'cameras': self.cams,
            'rounds': rounds, 'legs': self.leg_log,
            'claimed': {
                'fraction': round(float(claimed.mean()), 5),
                'unverified_fraction': round(float((self.onboard.unverified.covered() & ~claimed).mean()), 5),
                'false_claim_fraction': round(float((claimed & ~true).sum() / max(claimed.sum(), 1)), 5),
                'unclaimed_seen_fraction': round(float((true & ~claimed).sum() / max(true.sum(), 1)), 5),
                'holes_remaining': len(remaining), 'hole_cells_remaining': int(sum(h['cells'] for h in remaining)),
                'frames': dict(self.claims),
            },
            'navigation': {
                'horizontal_error_rms_m': round(float(np.sqrt((herr ** 2).mean())), 4),
                'horizontal_error_max_m': round(float(herr.max()), 4),
                'horizontal_error_final_m': round(float(herr[-1]), 4),
                'camera_error_rms_m': dict(zip(('along', 'girth', 'normal'), _r(np.sqrt((fe ** 2).mean(0))))),
                'camera_error_max_m': dict(zip(('along', 'girth', 'normal'), _r(np.abs(fe).max(0)))),
                'standoff_updates': dict(self.onboard.standoff_stats),
                'hull_end_updates': dict(self.onboard.hull_end_stats),
                'ship_fix_updates': dict(self.onboard.ship_fix_stats),
                'ekf': self.onboard.nav.summary(),
                'health': list(self.onboard.health_log),
            },
            'clearance_min_m': round(float(tk[:, -1].min()), 4) if len(tk) else None,
            'contacts': {'ticks': len(self.vehicle.contacts),
                         'max_penetration_m': max((c['penetration_m'] for c in self.vehicle.contacts), default=0.0),
                         'first': self.vehicle.contacts[:5]},
            'battery_percent_final': round(self.vehicle.battery_percent(), 2),
            'faults': self.vehicle.faults.log,
        })
        return ref

    def finish(self, summary: dict):
        cl = summary['claimed']
        self.events.emit(self.vehicle.t, 'mission_finished', 'critical' if self.aborted else 'info',
                         outcome=f'aborted: {self.aborted}' if self.aborted else 'completed', claimed_fraction=cl['fraction'],
                         holes_remaining=cl['holes_remaining'], frames=cl['frames'], battery_pct=round(self.onboard.battery, 1))

    def write(self, summary: dict):
        with open(os.path.join(self.out_dir, 'coverage.json'), 'w') as f:
            json.dump(summary, f, indent=1)
        write_metrics(os.path.join(self.out_dir, METRICS_FILE), mission_samples(summary, self.events.counts))
        Image.fromarray(self.referee.image()).save(os.path.join(self.out_dir, 'coverage.png'))
        Image.fromarray(self.onboard.claimed.image()).save(os.path.join(self.out_dir, 'claimed.png'))
        Image.fromarray(claim_check(self.referee.covered(), self.onboard.claimed.covered())).save(
            os.path.join(self.out_dir, 'claim_check.png'))
        cols = ['t', 'x', 'y', 'z', 'roll', 'pitch', 'yaw', 'x_est', 'y_est', 'z_est', 'yaw_est', 'x_ref', 'y_ref', 'z_ref',
                *[f'thrust_{n}' for n in self.ctrl.names], 'std_along', 'std_girth', 'tilt_deg', 'clearance']
        np.savez_compressed(os.path.join(self.out_dir, 'trajectory.npz'), columns=np.array(cols),
                            data=np.asarray(self.ticks, np.float64))

    def close(self):
        self.events.close()
        if self.own_renderer:
            self.rend.close()
        self.vehicle.close()


def drain(gen):
    """Run a generator to its end and return its value."""
    while True:
        try:
            next(gen)
        except StopIteration as e:
            return e.value


def claim_check(true: np.ndarray, claimed: np.ndarray) -> np.ndarray:
    """Colour map of claim against truth: green both, red claimed but not seen, blue seen but not
    claimed, black neither."""
    out = np.zeros((*true.shape, 3), np.uint8)
    for key, m in (('both', true & claimed), ('false_claim', claimed & ~true), ('unclaimed', true & ~claimed)):
        out[m] = CLAIM_COLOURS[key]
    return out


def run(scn: Scenario, out_dir: str, save_frames: bool = True, seed: int | None = None,
        cameras: list[str] | None = None) -> dict:
    os.makedirs(out_dir, exist_ok=True)
    if save_frames:
        os.makedirs(os.path.join(out_dir, 'frames'), exist_ok=True)
        os.makedirs(os.path.join(out_dir, 'gt'), exist_ok=True)
    m = _Mission(scn, out_dir, save_frames, scn.seed if seed is None else int(seed), cameras)
    try:
        summary = drain(m.run())
        m.finish(summary)
        m.write(summary)
    finally:
        m.close()
    log.info('mission: %d poses, coverage %.4f, claimed %.4f, aborted=%s', summary['poses'],
             summary['coverage_fraction'], summary['claimed']['fraction'], summary['aborted'])
    return summary
