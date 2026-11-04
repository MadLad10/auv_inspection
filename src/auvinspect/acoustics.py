"""The acoustic link between the fleet's vehicles and the topside station.

One shared channel, time-division: the nodes (topside, then each vehicle) take turns, each holding
the channel for acoustic.slot_s. In its slot a node sends one packet of at most
bitrate_bps x (slot_s - guard_s) / 8 bytes: first the acknowledgements it owes, then its reliable
messages (new ones, and those unacknowledged for ack_timeout_cycles turns of the channel), then a
status report if room is
left (acknowledgements that do not fit wait for the next slot). A packet reaches every other node after its transmission time plus range / sound_speed_mps,
and each receiver loses it independently with probability loss_base + loss_per_100m x range / 100
(always beyond max_range_m), drawn from the seeded stream. A node whose modem has failed
(the modem_dead fault) neither sends nor hears.

Reliable messages carry an id. The receiver acknowledges each one, and acts on it once however
often it arrives. The sender resends it until it is acknowledged, or gives up after max_retries
sends. Message sizes (bytes) come from acoustic.message_bytes; a message's size grows with what it
lists (lanes).

Ranges are taken between the true positions: the channel is part of the simulated world, not of
what the vehicles know. A node may watch the packets it hears (on_packet): the topside's acoustic
positioning measures where the sender was when it transmitted.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

TOPSIDE = 'topside'


@dataclass
class Message:
    kind: str
    sender: str
    to: str | None                     # None: everyone
    body: dict = field(default_factory=dict)
    reliable: bool = True
    id: int = 0
    sends: int = 0
    last_sent: float = -math.inf

    def size(self, sizes: dict) -> int:
        return int(sizes[self.kind] + sizes['per_item'] * len(self.body.get('lanes', [])))


class Node:
    """One modem: what it has to send, and what it has heard."""

    def __init__(self, name: str):
        self.name = name
        self.outbox: list[Message] = []
        self.inbox: list[Message] = []
        self.acks_owed: list[tuple[str, int]] = []
        self.seen: set[tuple[str, int]] = set()
        self.status = None                 # callable: the status body to send when there is room
        self.dead = None                   # callable(t): the modem has failed
        self.position = None               # callable: true position (world, m)
        self.on_packet = None              # callable(sender, true position at transmission, t sent, t heard)
        self._next_id = 1
        self.last_heard: dict[str, float] = {}       # when each other node was last heard
        self.stats = {'packets': 0, 'bytes': 0, 'sent': {}, 'resent': 0, 'given_up': 0, 'heard': 0, 'lost': 0}

    def send(self, kind: str, to: str | None = None, reliable: bool = True, **body):
        m = Message(kind, self.name, to, body, reliable, self._next_id if reliable else 0)
        if reliable:
            self._next_id += 1
        self.outbox.append(m)
        return m

    def take(self, kinds: tuple[str, ...]) -> list[Message]:
        """Remove and return the heard messages of these kinds."""
        got = [m for m in self.inbox if m.kind in kinds]
        self.inbox = [m for m in self.inbox if m.kind not in kinds]
        return got


class Link:
    """The shared channel (see the module notes)."""

    def __init__(self, nodes: list[Node], cfg: dict, rng: np.random.Generator):
        self.nodes = {n.name: n for n in nodes}
        self.order = [n.name for n in nodes]
        self.c = cfg['acoustic']
        self.rng = rng
        self.capacity = int(self.c['bitrate_bps'] * (self.c['slot_s'] - self.c['guard_s']) / 8)
        self.ack_timeout = self.c['ack_timeout_cycles'] * self.c['slot_s'] * len(self.order)
        self.in_flight: list[tuple] = []          # (arrival, receiver, messages, sender, sender's position, t sent)
        self._slot = -1
        self.log: list[dict] = []

    def _range(self, a: str, b: str) -> float:
        return float(np.linalg.norm(np.asarray(self.nodes[a].position()) - np.asarray(self.nodes[b].position())))

    def _dead(self, name: str, t: float) -> bool:
        n = self.nodes[name]
        return bool(n.dead and n.dead(t))

    def step(self, t: float):
        """Deliver what has arrived by t, then let the owner of a slot that has begun transmit."""
        due = [p for p in self.in_flight if p[0] <= t]
        self.in_flight = [p for p in self.in_flight if p[0] > t]
        for arrival, rx, msgs, tx, pos, t_tx in sorted(due, key=lambda p: p[0]):
            node = self.nodes[rx]
            if node.on_packet is not None:
                node.on_packet(tx, pos, t_tx, arrival)
            self._receive(node, msgs, t)
        slot = int(math.floor(t / self.c['slot_s'] + 1e-9))
        if slot != self._slot:
            self._slot = slot
            owner = self.order[slot % len(self.order)]
            if not self._dead(owner, t):
                self._transmit(self.nodes[owner], t)

    def _receive(self, node: Node, msgs: list[Message], t: float):
        for m in msgs:
            node.last_heard[m.sender] = t
            if m.kind == 'status' and node.name != TOPSIDE:
                continue                                  # vehicles only note that the topside is there
            if m.kind == 'ack':
                if m.to == node.name:
                    acked = set(m.body['ids'])
                    node.outbox = [o for o in node.outbox if not (o.reliable and o.id in acked)]
                continue
            if m.to not in (None, node.name):
                continue
            if m.reliable:
                node.acks_owed.append((m.sender, m.id))
                if (m.sender, m.id) in node.seen:
                    continue
                node.seen.add((m.sender, m.id))
            node.inbox.append(m)
            node.stats['heard'] += 1

    def _transmit(self, node: Node, t: float):
        sizes, cap = self.c['message_bytes'], self.capacity
        packet, used = [], 0
        acks: dict[str, list[int]] = {}
        for to, mid in node.acks_owed:
            acks.setdefault(to, []).append(mid)
        node.acks_owed = []
        for to, ids in acks.items():
            m = Message('ack', node.name, to, {'ids': sorted(set(ids))}, reliable=False)
            size = sizes['ack'] + sizes['per_item'] * len(m.body['ids'])
            if used + size <= cap:
                packet.append(m)
                used += size
            else:
                node.acks_owed += [(to, i) for i in m.body['ids']]
        for m in list(node.outbox):
            if m.reliable and t - m.last_sent < self.ack_timeout:
                continue
            size = m.size(sizes)
            if used + size > cap:
                continue
            if m.reliable and m.sends >= self.c['max_retries']:
                node.outbox.remove(m)
                node.stats['given_up'] += 1
                self.log.append({'t': round(t, 2), 'node': node.name, 'gave_up': m.kind, 'to': m.to})
                continue
            node.stats['resent'] += int(m.sends > 0)
            m.sends += 1
            m.last_sent = t
            node.stats['sent'][m.kind] = node.stats['sent'].get(m.kind, 0) + 1
            packet.append(m)
            used += size
            if not m.reliable:
                node.outbox.remove(m)
        if node.status is not None and used + sizes['status'] <= cap:
            m = Message('status', node.name, None if node.name == TOPSIDE else TOPSIDE, node.status(), reliable=False)
            packet.append(m)
            used += sizes['status']
            node.stats['sent']['status'] = node.stats['sent'].get('status', 0) + 1
        if not packet:
            return
        node.stats['packets'] += 1
        node.stats['bytes'] += used
        duration = used * 8 / self.c['bitrate_bps']
        pos = np.asarray(node.position(), float).copy()
        for rx in self.order:
            if rx == node.name or self._dead(rx, t):
                continue
            r = self._range(node.name, rx)
            p_loss = 1.0 if r > self.c['max_range_m'] else min(1.0, self.c['loss_base'] + self.c['loss_per_100m'] * r / 100.0)
            if self.rng.uniform() < p_loss:
                self.nodes[rx].stats['lost'] += 1
                continue
            self.in_flight.append((t + duration + r / self.c['sound_speed_mps'], rx, list(packet), node.name, pos, t))

    def summary(self) -> dict:
        return {'capacity_bytes_per_slot': self.capacity, 'cycle_s': round(self.c['slot_s'] * len(self.order), 3),
                'nodes': {n: dict(node.stats) for n, node in self.nodes.items()}, 'log': self.log}
