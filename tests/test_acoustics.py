"""The acoustic link and the topside coordinator, without vehicles: timing, capacity, loss and
retries, and the hand-over and release rules."""
import copy
import math

import numpy as np
import pytest

from auvinspect import config
from auvinspect.acoustics import TOPSIDE, Link, Node
from auvinspect.fleet import Topside
from auvinspect.observe import Events


def _link(cfg, positions, dead=None, seed=0):
    nodes = []
    for name, pos in positions.items():
        n = Node(name)
        n.position = (lambda p=np.asarray(pos, float): p)
        n.dead = (lambda t, d=(dead or {}).get(name): d is not None and t >= d)
        nodes.append(n)
    return Link(nodes, cfg, np.random.default_rng(seed)), {n.name: n for n in nodes}


def _run(link, until, dt=0.05):
    for k in range(int(until / dt) + 1):
        link.step(k * dt)


@pytest.fixture
def cfg():
    c = copy.deepcopy(config.load())
    c['acoustic'].update(loss_base=0.0, loss_per_100m=0.0)
    return c


def test_a_reliable_message_waits_for_its_slot_arrives_once_and_is_acknowledged(cfg):
    link, n = _link(cfg, {TOPSIDE: [0, 0, 0], 'v0': [150, 0, 0]})
    n['v0'].send('lane_done', to=TOPSIDE, lanes=[3])
    _run(link, 2.05)                                   # v0's slot starts at 2 s; the packet is still under way
    assert not n[TOPSIDE].inbox
    _run(link, 4.0)
    a = cfg['acoustic']
    got = n[TOPSIDE].take(('lane_done',))
    assert len(got) == 1 and got[0].body == {'lanes': [3]}
    assert n[TOPSIDE].last_heard['v0'] >= 2.0 + 150 / a['sound_speed_mps']
    _run(link, 20.0)
    assert not n['v0'].outbox and n['v0'].stats['resent'] == 0      # the ack came back: no resend
    assert not n[TOPSIDE].take(('lane_done',))


def test_a_slot_carries_what_fits_and_the_rest_waits(cfg):
    link, n = _link(cfg, {TOPSIDE: [0, 0, 0], 'v0': [10, 0, 0]})
    n['v0'].status = lambda: {}
    for k in range(30):
        n['v0'].send('lane_done', to=TOPSIDE, lanes=[k])
    _run(link, 4.0)                                    # one packet: sent at 2 s, 1.5 s on the air at most
    first = len(n[TOPSIDE].take(('lane_done',)))
    assert 0 < first < 30 and n['v0'].stats['bytes'] <= link.capacity
    _run(link, 60.0)
    assert first + len(n[TOPSIDE].take(('lane_done',))) == 30


def test_lost_packets_are_resent_until_acknowledged_and_abandoned_when_no_one_hears(cfg):
    lossy = copy.deepcopy(cfg)
    lossy['acoustic']['loss_base'] = 0.4                # each way: a round trip succeeds a third of the time
    link, n = _link(lossy, {TOPSIDE: [0, 0, 0], 'v0': [10, 0, 0]}, seed=3)
    n['v0'].send('abort', to=TOPSIDE, lanes=[0, 1])
    _run(link, 200.0)
    assert len(n[TOPSIDE].take(('abort',))) == 1 and n['v0'].stats['resent'] > 0 and n['v0'].stats['given_up'] == 0
    link, n = _link(cfg, {TOPSIDE: [0, 0, 0], 'v0': [10, 0, 0]}, dead={TOPSIDE: 0.0})
    n['v0'].send('abort', to=TOPSIDE, lanes=[0])
    _run(link, 400.0)
    assert n['v0'].stats['given_up'] == 1 and n['v0'].stats['sent']['abort'] == cfg['acoustic']['max_retries']
    link, n = _link(cfg, {TOPSIDE: [0, 0, 0], 'v0': [cfg['acoustic']['max_range_m'] + 1, 0, 0]})
    n['v0'].send('finished', to=TOPSIDE)
    _run(link, 30.0)
    assert not n[TOPSIDE].inbox and n[TOPSIDE].stats['lost'] > 0


def _topside(cfg, tmp_path, planned=(100.0, 100.0)):
    node = Node(TOPSIDE)
    girths = [1.0, 2.0, 3.0, 4.0]
    plan = [{'lanes': [0, 1], 'x_range': None, 'centre': 1.5, 'area': {'s': [None, 2.5], 'x': None}},
            {'lanes': [2, 3], 'x_range': None, 'centre': 3.5, 'area': {'s': [2.5, None], 'x': None}}]
    return Topside(node, plan, girths, list(planned), 50.0, cfg, Events(str(tmp_path), cfg)), node


def _hear(node, sender, kind, **body):
    from auvinspect.acoustics import Message
    node.inbox.append(Message(kind, sender, TOPSIDE, body))


def test_an_abort_hands_the_lanes_left_to_a_vehicle_done_with_its_own(cfg, tmp_path):
    top, node = _topside(cfg, tmp_path)
    _hear(node, 'v1', 'abort', lanes=[[1, 2]])
    top.step(10.0)
    assert top.pending == [{'share': 1, 'lanes': [3], 'from': 1, 't': 10.0}] and not node.outbox      # no one free yet
    _hear(node, 'v0', 'lane_done', lanes=[0, 1], share=0)
    _hear(node, 'v0', 'finished')
    top.step(20.0)
    (m,) = [m for m in node.outbox if m.kind == 'assign']
    assert m.to == 'v0' and m.body == {'lanes': [3], 'share': 1, 'x_range': None, 'area': {'s': [3.5, None], 'x': None}}
    assert top.takeovers[0]['to'] == 0 and top.state[0] == 'working' and not top.pending
    _hear(node, 'v0', 'lane_done', lanes=[3], share=1)
    _hear(node, 'v0', 'finished')
    top.step(80.0)
    assert [m.to for m in node.outbox if m.kind == 'release'] == ['v0']


def test_a_silent_vehicle_keeps_its_lanes_and_is_waited_for_until_past_its_planned_finish(cfg, tmp_path):
    top, node = _topside(cfg, tmp_path, planned=(100.0, 100.0))
    _hear(node, 'v0', 'finished')
    top.step(5.0)
    _hear(node, 'v0', 'status')                        # v0 keeps reporting; v1 is never heard
    top.step(cfg['fleet']['silent_after_s'] + 10.0)
    assert top.silent == {1} and not top.pending and not [m for m in node.outbox if m.kind in ('assign', 'release')]
    for t in np.arange(50.0, 100.0 + cfg['fleet']['silent_margin_s'], 10.0):
        _hear(node, 'v0', 'status')
        top.step(float(t))
    assert not [m for m in node.outbox if m.kind == 'release']
    _hear(node, 'v0', 'status')
    top.step(100.0 + cfg['fleet']['silent_margin_s'] + 1.0)
    assert [m.to for m in node.outbox if m.kind == 'release'] == ['v0'] and not top.takeovers
    assert math.isfinite(top.heard[0])


def test_the_topside_sends_each_vehicle_where_it_was_when_it_transmitted(cfg, tmp_path):
    cfg['acoustic']['usbl']['enabled'] = True
    top, node = _topside(cfg, tmp_path)
    top.position, top.rng = np.zeros(3), np.random.default_rng(0)
    top.on_packet('v1', np.array([20.0, 0.0, -2.0]), 12.0, 12.1)
    (m,) = node.outbox
    u = cfg['acoustic']['usbl']
    assert m.kind == 'fix' and m.to == 'v1' and not m.reliable and m.body['t'] == 12.0
    assert m.body['std'] == pytest.approx(u['std_base_m'] + u['std_frac'] * math.hypot(20.0, 2.0), abs=1e-4)
    assert abs(m.body['x'] - 20.0) < 5 * m.body['std'] and abs(m.body['y']) < 5 * m.body['std']
    top.on_packet(TOPSIDE, np.zeros(3), 1.0, 1.0)
    assert len(node.outbox) == 1
