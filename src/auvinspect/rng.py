"""Named, independent random streams so that adding draws in one component never shifts another."""
from __future__ import annotations

import zlib

import numpy as np


def stream(seed: int, name: str) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([int(seed), zlib.crc32(name.encode())]))
