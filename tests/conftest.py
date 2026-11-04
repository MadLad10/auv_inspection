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


@pytest.fixture(scope='session')
def small_mission(tmp_path_factory, small_scn):
    """The small scenario flown once with the vehicle in the loop (no frames saved): summary and directory."""
    from auvinspect import mission
    out = tmp_path_factory.mktemp('nominal')
    return mission.run(small_scn, str(out), save_frames=False), out


@pytest.fixture(scope='session')
def small_survey(tmp_path_factory, small_scn):
    """The small scenario saved, with a kinematic survey that keeps its frames and ground truth."""
    from auvinspect import survey
    root = tmp_path_factory.mktemp('small_survey')
    small_scn.save(str(root / 'scn'))
    survey.run(small_scn, str(root / 'scn' / 'survey'), save_frames=True)
    return root
