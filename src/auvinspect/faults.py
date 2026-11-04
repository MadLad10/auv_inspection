"""Scheduled faults with a ground-truth onset/clear log.

A fault is {type, t_start, duration, params}; duration null means permanent. Sensors, actuators and
the camera query the injector every tick; the log records exactly when each fault was active, so
onboard fault handling can be scored against it.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

FAULT_TYPES = {
    'camera_blackout': 'the inspection camera returns black frames',
    'turbidity': 'the water clouds (params: visibility_m)',
    'dvl_dropout': 'the DVL reports no valid velocity',
    'imu_bias': 'gyro z bias step (params: gyro_bias_z, rad/s)',
    'compass_disturbance': 'heading offset (params: offset_deg)',
    'pressure_stuck': 'the depth sensor holds its last value',
    'thruster_dead': 'one thruster produces no thrust (params: name)',
    'standoff_dropout': 'the standoff sounder returns no echo',
    'battery_drop': 'part of the stored energy is lost (params: drop_pct)',
    'modem_dead': 'the acoustic modem neither sends nor hears (a fleet member)',
}


@dataclass
class Fault:
    type: str
    t_start: float = 0.0
    duration: float | None = None
    params: dict = field(default_factory=dict)

    def __post_init__(self):
        if self.type not in FAULT_TYPES:
            raise ValueError(f'unknown fault type {self.type!r}; known: {sorted(FAULT_TYPES)}')

    def active(self, t: float) -> bool:
        return t >= self.t_start and (self.duration is None or t < self.t_start + self.duration)


class FaultInjector:
    def __init__(self, faults: list | None = None):
        self.faults = [f if isinstance(f, Fault) else Fault(**f) for f in (faults or [])]
        self.log: list[dict] = []
        self._state: dict[int, bool] = {}

    def update(self, t: float):
        """Record onsets and clears; call once per tick."""
        for i, f in enumerate(self.faults):
            a = f.active(t)
            if a != self._state.get(i, False):
                self._state[i] = a
                self.log.append({'t': round(t, 3), 'event': 'onset' if a else 'clear', **asdict(f)})

    def get(self, ftype: str, t: float) -> Fault | None:
        return next((f for f in self.faults if f.type == ftype and f.active(t)), None)

    def all(self, ftype: str, t: float) -> list[Fault]:
        return [f for f in self.faults if f.type == ftype and f.active(t)]

    def active_types(self, t: float) -> list[str]:
        return sorted({f.type for f in self.faults if f.active(t)})
