"""Observability: what an operator sees of a mission, and the alert rules applied to it.

- **Event stream** (events.jsonl in the run directory), written and flushed as the mission flies. It
  holds only what the vehicle itself knows, never ground truth. Each event has the simulated time t,
  a name and a level (debug, info, warning, critical):
    mission_started, mission_finished   the plan; the outcome, claimed share, holes left, battery
    status                              every observability.status_every_s: phase, leg, estimated hull
                                        position and its 1-sigma, battery, frames by claim, claimed share
    leg_finished                        its outcome (a timeout or an unfound hull end is a warning)
    revisit_planned, revisit_skipped    the holes in the claimed coverage targeted, or why they are not
    dvl_lost, dvl_restored              the DVL's validity flag
    camera_black, camera_restored       dead inspection frames
    sensor_excluded, sensor_restored    the onboard health checks
    turbid_water, water_clearer         the visibility from the water-quality sensor falling below
                                        observability.turbid_visibility_m, and back above it times
                                        turbid_clear_factor
    battery_low, abort                  the battery below observability.battery_warn_pct; the reserve reached
  The sounder's echo is lost at every hull end on purpose (the along-ship fix), so it is not an event.
- **Metrics** (metrics.prom), in Prometheus' text format for a textfile collector, written when the
  mission ends: the vehicle's own numbers as auvinspect_*, and the referee's verdicts, which exist
  only in simulation, as auvinspect_referee_*. `benchmark score` and `gate` write a suite's values
  the same way.
- **Stage traces** (trace.jsonl beside a benchmark hull's result.json): generate, fly, detect and
  report, each with its wall-clock start and duration.
- **Alert rules** (observability.alerts) over a run's metrics: `auvinspect health` prints the alerts
  that fire and exits 1 when one is critical. `auvinspect watch` follows a mission's events as it flies.
"""
from __future__ import annotations

import json
import math
import os
import re
import time
from contextlib import contextmanager
from datetime import datetime, timezone

LEVELS = ('debug', 'info', 'warning', 'critical')
EVENTS_FILE, METRICS_FILE, TRACE_FILE = 'events.jsonl', 'metrics.prom', 'trace.jsonl'


# ---------------------------------------------------------------------------------------------- events
class Events:
    """The mission's event stream (see the module notes)."""

    def __init__(self, out_dir: str, cfg: dict):
        self.f = open(os.path.join(out_dir, EVENTS_FILE), 'w')
        self.o, self.vis_constant = cfg['observability'], cfg['water']['visibility_constant']
        self.counts = {lv: 0 for lv in LEVELS}
        self._ok = {'dvl': True, 'camera': True, 'water': True}
        self._health_seen = 0
        self._battery_warned = False
        self._next_status = 0.0

    def emit(self, t: float, name: str, level: str = 'info', **fields):
        self.counts[level] += 1
        self.f.write(json.dumps({'t': round(float(t), 3), 'event': name, 'level': level, **fields}) + '\n')
        self.f.flush()

    def _flip(self, t: float, key: str, ok: bool, lost: tuple[str, str], back: tuple[str, str], **fields):
        if ok != self._ok[key]:
            self._ok[key] = ok
            name, level = back if ok else lost
            self.emit(t, name, level, **fields)

    def tick(self, t: float, readings: dict, onboard):
        """Sensor validity, health checks, water and battery, once per control tick."""
        dvl = readings.get('dvl')
        if dvl is not None:
            self._flip(t, 'dvl', bool(dvl['valid']), ('dvl_lost', 'warning'), ('dvl_restored', 'info'))
        for h in onboard.health_log[self._health_seen:]:
            excluded = 'excluded' in h['event']
            self.emit(t, 'sensor_excluded' if excluded else 'sensor_restored', 'warning' if excluded else 'info',
                      sensor=h['sensor'], detail=h['event'])
        self._health_seen = len(onboard.health_log)
        wq = readings.get('water_quality')
        if wq is not None:
            vis = self.vis_constant / max(float(wq['attenuation_per_m'][1]), 1e-9)
            limit = self.o['turbid_visibility_m'] * (1.0 if self._ok['water'] else self.o['turbid_clear_factor'])
            self._flip(t, 'water', vis >= limit, ('turbid_water', 'warning'), ('water_clearer', 'info'), visibility_m=round(vis, 2))
        if not self._battery_warned and onboard.battery < self.o['battery_warn_pct']:
            self._battery_warned = True
            self.emit(t, 'battery_low', 'warning', battery_pct=round(onboard.battery, 1))

    def frame(self, t: float, claim: str):
        self._flip(t, 'camera', claim != 'black', ('camera_black', 'warning'), ('camera_restored', 'info'))

    def status_due(self, t: float) -> bool:
        if t + 1e-9 < self._next_status:
            return False
        self._next_status = t + self.o['status_every_s']
        return True

    def close(self):
        self.f.close()


def read_events(run_dir: str) -> list[dict]:
    with open(os.path.join(run_dir, EVENTS_FILE)) as f:
        return [json.loads(line) for line in f if line.strip()]


def format_event(e: dict) -> str:
    fields = ' '.join(f'{k}={v}' for k, v in e.items() if k not in ('t', 'event', 'level'))
    return f'{e["t"]:9.2f} s  {e["level"].upper():8s} {e["event"]:17s} {fields}'.rstrip()


def watch(run_dir: str, follow: bool = True, poll_s: float = 0.5, out=print):
    """Print a mission's events, following the file as it grows until mission_finished."""
    path = os.path.join(run_dir, EVENTS_FILE)
    while not os.path.exists(path):
        if not follow:
            raise FileNotFoundError(path)
        time.sleep(poll_s)
    with open(path) as f:
        buf = ''
        while True:
            chunk = f.readline()
            if not chunk:
                if not follow:
                    return
                time.sleep(poll_s)
                continue
            buf += chunk
            if not buf.endswith('\n'):
                continue
            e = json.loads(buf)
            buf = ''
            out(format_event(e))
            if e['event'] == 'mission_finished':
                return


# ---------------------------------------------------------------------------------------------- metrics
def _labels(labels: dict) -> str:
    if not labels:
        return ''
    esc = {k: str(v).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n') for k, v in labels.items()}
    return '{' + ','.join(f'{k}="{v}"' for k, v in sorted(esc.items())) + '}'


def prom_text(samples: list[tuple[str, str, str, dict, float]]) -> str:
    """Prometheus text exposition of (name, help, type, labels, value) samples, grouped by name."""
    out, seen = [], {}
    for name, help_, kind, labels, value in samples:
        seen.setdefault(name, (help_, kind, []))[2].append((labels, value))
    for name, (help_, kind, rows) in seen.items():
        out += [f'# HELP {name} {help_}', f'# TYPE {name} {kind}']
        out += [f'{name}{_labels(lb)} {float(v)!r}' if v is not None and math.isfinite(float(v)) else f'{name}{_labels(lb)} NaN'
                for lb, v in rows]
    return '\n'.join(out) + '\n'


_SAMPLE = re.compile(r'^([a-zA-Z_:][a-zA-Z0-9_:]*)(\{(.*)\})?\s+(\S+)$')
_LABEL = re.compile(r'([a-zA-Z_][a-zA-Z0-9_]*)="((?:[^"\\]|\\.)*)"')


def parse_prom(text: str) -> list[tuple[str, dict, float]]:
    out = []
    for line in text.splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        m = _SAMPLE.match(line.strip())
        if not m:
            raise ValueError(f'not a Prometheus sample: {line!r}')
        labels = {k: v.replace('\\n', '\n').replace('\\"', '"').replace('\\\\', '\\') for k, v in _LABEL.findall(m.group(3) or '')}
        out.append((m.group(1), labels, float(m.group(4))))
    return out


def write_metrics(path: str, samples: list) -> None:
    with open(path + '.tmp', 'w') as f:
        f.write(prom_text(samples))
    os.replace(path + '.tmp', path)


def mission_samples(summary: dict, counts: dict) -> list[tuple]:
    """A mission's metrics from its summary (coverage.json) and its event counts."""
    lb = {'seed': summary['seed'], 'scenario': summary['scenario_digest'][:12]}
    cl = summary['claimed']
    legs: dict[str, int] = {}
    for leg in summary['legs']:
        legs[leg['outcome']] = legs.get(leg['outcome'], 0) + 1
    s = [('auvinspect_mission_sim_seconds', 'Simulated mission time.', 'gauge', lb, summary['sim_time_s']),
         ('auvinspect_mission_wall_seconds', 'Wall-clock time of the mission.', 'gauge', lb, summary['elapsed_s']),
         ('auvinspect_mission_aborted', 'Whether the mission was abandoned (1) or completed (0).', 'gauge', lb, int(bool(summary['aborted'])))]
    s += [('auvinspect_frames_total', 'Inspection frames by the onboard claim check.', 'counter', {**lb, 'claim': k}, v)
          for k, v in sorted(cl['frames'].items())]
    s += [('auvinspect_legs_total', 'Legs flown, by outcome.', 'counter', {**lb, 'outcome': k}, v) for k, v in sorted(legs.items())]
    s += [('auvinspect_events_total', 'Events in the stream, by level.', 'counter', {**lb, 'level': k}, v) for k, v in counts.items()]
    s += [('auvinspect_claimed_fraction', 'Share of the inspection band the vehicle claims to have covered.', 'gauge', lb, cl['fraction']),
          ('auvinspect_holes_remaining', 'Holes left in the claimed coverage.', 'gauge', lb, cl['holes_remaining']),
          ('auvinspect_battery_percent', 'State of charge at the end.', 'gauge', lb, summary['battery_percent_final'])]
    s += [('auvinspect_standoff_updates_total', 'Standoff sounder updates to the navigation, by result.', 'counter',
           {**lb, 'result': k}, v) for k, v in sorted(summary['navigation']['standoff_updates'].items())]
    nav = summary['navigation']
    s += [('auvinspect_referee_coverage_fraction', 'Simulation only: the referee\'s coverage of the band.', 'gauge', lb, summary['coverage_fraction']),
          ('auvinspect_referee_false_claim_fraction', 'Simulation only: share of claimed cells the referee does not count covered.',
           'gauge', lb, cl['false_claim_fraction']),
          ('auvinspect_referee_nav_error_rms_m', 'Simulation only: horizontal navigation error, RMS.', 'gauge', lb,
           nav['horizontal_error_rms_m']),
          ('auvinspect_referee_clearance_min_m', 'Simulation only: smallest clearance to the hull or seabed.', 'gauge', lb,
           summary['clearance_min_m']),
          ('auvinspect_referee_contact_ticks_total', 'Simulation only: control ticks in contact.', 'counter', lb,
           summary['contacts']['ticks'])]
    faults: dict[str, int] = {}
    for f in summary['faults']:
        if f['event'] == 'onset':
            faults[f['type']] = faults.get(f['type'], 0) + 1
    s += [('auvinspect_referee_faults_injected_total', 'Simulation only: faults injected, by type.', 'counter', {**lb, 'type': k}, v)
          for k, v in sorted(faults.items())]
    return s


def suite_samples(summary: dict, passed: bool | None = None) -> list[tuple]:
    """A benchmark suite's values (benchmark.json) and, after a gate, whether it passed."""
    lb = {'suite': summary['suite']}
    s = [('auvinspect_benchmark_hulls', 'Hulls scored in the suite.', 'gauge', lb, summary['hulls'])]
    for k, v in sorted(summary['values'].items()):
        s.append(('auvinspect_benchmark_value', 'A benchmark value, pooled over the suite\'s hulls.', 'gauge', {**lb, 'value': k}, v['value']))
        for side in ('low', 'high'):
            if v.get(side) is not None:
                s.append((f'auvinspect_benchmark_interval_{side}', f'The {side} end of the value\'s interval.', 'gauge',
                          {**lb, 'value': k}, v[side]))
    if passed is not None:
        s.append(('auvinspect_benchmark_gate_passed', 'Whether the suite passed its gate.', 'gauge', lb, int(passed)))
    return s


# ---------------------------------------------------------------------------------------------- alerts
_OPS = {'>': lambda a, b: a > b, '>=': lambda a, b: a >= b, '<': lambda a, b: a < b, '<=': lambda a, b: a <= b,
        '==': lambda a, b: a == b, '!=': lambda a, b: a != b}


def alerts(samples: list[tuple[str, dict, float]], rules: list[dict]) -> list[dict]:
    """The rules that fire on parsed metric samples: one alert per matching sample."""
    out = []
    for r in rules:
        for name, labels, value in samples:
            if name == r['metric'] and all(labels.get(k) == str(v) for k, v in (r.get('labels') or {}).items()) \
                    and _OPS[r['op']](value, r['value']):
                out.append({'alert': r['name'], 'severity': r['severity'], 'metric': name, 'labels': labels, 'value': value,
                            'rule': f'{r["op"]} {r["value"]}'})
    return out


def health(run_dir: str, cfg: dict) -> tuple[list[dict], bool]:
    """The alerts a run's metrics raise, and whether any is critical."""
    with open(os.path.join(run_dir, METRICS_FILE)) as f:
        fired = alerts(parse_prom(f.read()), cfg['observability']['alerts'])
    return fired, any(a['severity'] == 'critical' for a in fired)


# ---------------------------------------------------------------------------------------------- traces
class Trace:
    """Stage spans of one benchmark hull, appended to trace.jsonl."""

    def __init__(self, out_dir: str, **attrs):
        self.path, self.attrs = os.path.join(out_dir, TRACE_FILE), attrs
        self.spans: list[dict] = []

    @contextmanager
    def span(self, name: str, **attrs):
        start, t0 = datetime.now(timezone.utc).isoformat(timespec='milliseconds'), time.perf_counter()
        status = 'ok'
        try:
            yield
        except BaseException:
            status = 'error'
            raise
        finally:
            rec = {'span': name, 'start': start, 'seconds': round(time.perf_counter() - t0, 3), 'status': status,
                   **self.attrs, **attrs}
            self.spans.append(rec)
            with open(self.path, 'a') as f:
                f.write(json.dumps(rec) + '\n')

    def samples(self) -> list[tuple]:
        lb = {k: v for k, v in self.attrs.items() if k in ('suite', 'seed')}
        return [('auvinspect_stage_seconds', 'Wall-clock seconds of a pipeline stage.', 'gauge', {**lb, 'stage': s['span']}, s['seconds'])
                for s in self.spans]
