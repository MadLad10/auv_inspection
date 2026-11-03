"""Observability: the metrics format, the alert rules, stage traces and the event stream's tools."""
import json

import pytest

from auvinspect import config, observe


def test_metrics_round_trip_in_prometheus_text_format():
    samples = [('auvinspect_frames_total', 'Frames.', 'counter', {'seed': 3, 'claim': 'black'}, 4),
               ('auvinspect_frames_total', 'Frames.', 'counter', {'seed': 3, 'claim': 'claimed'}, 90),
               ('auvinspect_battery_percent', 'Battery.', 'gauge', {'note': 'a "quoted"\\ value'}, 61.5),
               ('auvinspect_referee_clearance_min_m', 'Clearance.', 'gauge', {}, None)]
    text = observe.prom_text(samples)
    assert text.count('# TYPE auvinspect_frames_total counter') == 1 and text.count('# HELP') == 3
    back = observe.parse_prom(text)
    assert back[0] == ('auvinspect_frames_total', {'seed': '3', 'claim': 'black'}, 4.0)
    assert back[2] == ('auvinspect_battery_percent', {'note': 'a "quoted"\\ value'}, 61.5)
    assert back[3][0] == 'auvinspect_referee_clearance_min_m' and back[3][2] != back[3][2]     # NaN
    with pytest.raises(ValueError):
        observe.parse_prom('not a sample line here')


def test_alert_rules_fire_on_matching_samples_only():
    samples = observe.parse_prom(observe.prom_text([
        ('auvinspect_frames_total', 'Frames.', 'counter', {'claim': 'black'}, 3),
        ('auvinspect_frames_total', 'Frames.', 'counter', {'claim': 'claimed'}, 500),
        ('auvinspect_mission_aborted', 'Aborted.', 'gauge', {}, 0)]))
    rules = config.load()['observability']['alerts']
    fired = observe.alerts(samples, rules)
    assert [(a['alert'], a['severity'], a['labels']) for a in fired] == [('dead frames', 'warning', {'claim': 'black'})]
    assert observe.alerts(samples, [{'name': 'many frames', 'metric': 'auvinspect_frames_total', 'op': '>=', 'value': 500,
                                     'severity': 'critical'}])[0]['labels'] == {'claim': 'claimed'}


def test_stage_spans_are_timed_and_kept_even_when_a_stage_fails(tmp_path):
    t = observe.Trace(str(tmp_path), suite='ci', seed=401)
    with t.span('detect', detector='classical'):
        pass
    with pytest.raises(RuntimeError):
        with t.span('report'):
            raise RuntimeError('broken')
    spans = [json.loads(line) for line in (tmp_path / observe.TRACE_FILE).read_text().splitlines()]
    assert [(s['span'], s['status'], s['seed']) for s in spans] == [('detect', 'ok', 401), ('report', 'error', 401)]
    assert spans[0]['detector'] == 'classical' and spans[0]['seconds'] >= 0
    assert {s[3]['stage'] for s in t.samples()} == {'detect', 'report'}


def test_watch_prints_events_until_the_mission_finishes(tmp_path):
    lines = [{'t': 0.0, 'event': 'mission_started', 'level': 'info', 'legs': 9},
             {'t': 61.25, 'event': 'dvl_lost', 'level': 'warning'},
             {'t': 300.0, 'event': 'mission_finished', 'level': 'info', 'outcome': 'completed'},
             {'t': 301.0, 'event': 'after the end', 'level': 'debug'}]
    (tmp_path / observe.EVENTS_FILE).write_text(''.join(json.dumps(e) + '\n' for e in lines))
    out = []
    observe.watch(str(tmp_path), follow=True, out=out.append)                  # stops at mission_finished
    assert len(out) == 3 and 'WARNING' in out[1] and 'dvl_lost' in out[1] and out[0].endswith('legs=9')
