import os

import pytest

os.environ.setdefault('MUJOCO_GL', 'egl')

from auvinspect import config, scenario  # noqa: E402

SMALL = {'hull': {'length_m': 6.0}}


@pytest.fixture(scope='session')
def small_cfg():
    return config.load(overrides=SMALL)


@pytest.fixture(scope='session')
def small_scn(small_cfg):
    return scenario.generate(3, small_cfg)
