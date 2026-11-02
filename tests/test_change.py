"""The hull some months later (scenario evolution) and change detection between two surveys."""
import numpy as np
import pytest

from auvinspect import change, evolution
from auvinspect.hull import anode_stations


@pytest.fixture(scope='module')
def later(small_scn, small_cfg):
    return evolution.evolve(small_scn, 12, small_cfg)


def test_evolution_grows_defects_and_leaves_old_paint_alone(small_scn, small_cfg, later):
    e = small_cfg['evolution']
    assert later.lineage['base_digest'] == small_scn.digest() and later.digest() != small_scn.digest()
    assert evolution.evolve(small_scn, 12, small_cfg).digest() == later.digest()          # deterministic
    before = {it['id']: it for it in small_scn.items}
    after = {it['id']: it for it in later.items}
    assert set(before) <= set(after)                                                      # ids are kept
    for iid, it in before.items():
        if it['class'] in evolution.GROWING:
            target = (1 + e[it['class']]['growth_per_month']) ** 12
            assert 1.0 < after[iid]['area_m2'] / it['area_m2'] <= target * 1.01          # grows, at most to the rate
        if it['class'] == 'crack':
            assert after[iid]['length_m'] == pytest.approx(it['length_m'] + e['crack']['growth_mm_per_month'] * 12 / 1000)
    old = small_scn.labels >= 0
    assert np.array_equal(later.labels[old], small_scn.labels[old])                      # old defects keep their pixels
    same = later.labels == small_scn.labels
    assert np.array_equal(later.texture[same & old], small_scn.texture[same & old])
    for iid in set(after) - set(before):
        assert after[iid]['new_after_months'] == 12
    truth = evolution.changes(small_scn, later, small_cfg['change']['growth_ratio'])
    assert all(truth[i] == 'new' for i in set(after) - set(before))


def _f(fid, cls, box, area, tier='confirmed', item=None):
    return {'id': fid, 'class': cls, 'score': 0.9, 'hull_box': box, 'centre': [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2],
            'area_m2': area, 'tier': tier, 'match': {'item': item} if item is not None else {'false': True}}


def _item(iid, cls, area, **kw):
    return {'id': iid, 'class': cls, 'area_m2': area, **kw}


def test_change_labels_and_their_score(small_cfg):
    before = [_f('A1', 'corrosion', [1.0, 1.0, 1.2, 1.2], 0.03, item=1),
              _f('A2', 'coating_damage', [3.0, 1.0, 3.3, 1.3], 0.05, item=2),
              _f('A4', 'corrosion', [6.0, 1.0, 6.1, 1.1], 0.01, 'possible')]                     # false, gone later
    after = [_f('B1', 'corrosion', [0.98, 0.98, 1.25, 1.25], 0.05, item=1),                    # grown x1.67
             _f('B2', 'coating_damage', [3.0, 1.0, 3.3, 1.3], 0.052, item=2),                  # unchanged
             _f('B3', 'corrosion', [3.0, 1.0, 3.3, 1.3], 0.05, 'possible'),                    # other class: no partner
             _f('B4', 'crack', [8.0, 2.0, 8.2, 2.01], 0.002, item=4),                          # new
             _f('B5', 'biofouling', [9.0, 1.0, 9.1, 1.1], 0.01, 'possible')]                    # false new
    changes = change.compare(before, after, small_cfg)
    got = {(c['before'], c['after']): (c['change'], c['tier']) for c in changes}
    assert got == {('A1', 'B1'): ('grown', 'confirmed'), ('A2', 'B2'): ('unchanged', 'confirmed'),
                   (None, 'B3'): ('new', 'possible'), (None, 'B4'): ('new', 'confirmed'),
                   (None, 'B5'): ('new', 'possible'), ('A4', None): ('gone', 'possible')}
    spec_a = {'items': [_item(1, 'corrosion', 0.03), _item(2, 'coating_damage', 0.05)]}
    spec_b = {'items': [_item(1, 'corrosion', 0.05), _item(2, 'coating_damage', 0.05), _item(4, 'crack', 0.002)]}
    m = change.evaluate(changes, before, after, spec_a, spec_b, small_cfg)
    assert m['new'] == {'found': 1, 'flagged': 1, 'recall': 1.0, 'false': 2}
    assert m['grown'] == {'found': 1, 'flagged': 1, 'recall': 1.0, 'false': 0}
    assert m['area_ratio_log_error_median'] == pytest.approx(np.log(1.04) / 2, abs=1e-3)   # median of 0 and log 1.04


def test_station_changes_and_their_score(small_cfg):
    assert [change.station_change(a, b) for a, b in (('intact', 'depleted'), ('depleted', 'missing'), ('intact', 'missing'),
                                                       ('missing', 'intact'), ('depleted', 'depleted'), ('intact', 'not_inspected'))] \
        == ['anode_depleted', 'anode_lost', 'anode_lost', 'anode_replaced', 'unchanged', 'unknown']
    plan = anode_stations(small_cfg)
    assert len(plan) == 2                                                   # the 6 m hull: stations at 1.5 and 4.5 m
    st = lambda sid, status: {'id': sid, 'x': 0.0, 'status': status}
    changes = change.compare_stations([st('S01', 'intact'), st('S02', 'depleted')], [st('S01', 'depleted'), st('S02', 'depleted')])
    assert [c['change'] for c in changes] == ['anode_depleted', 'unchanged']

    def spec(statuses):
        items = []
        for k, (p, status) in enumerate(zip(plan, statuses)):
            if status == 'missing':
                items.append({'id': k, 'class': 'missing_anode', 'bbox_xs': p['box']})
            else:
                items.append({'id': k, 'class': 'anode', 'status': status, 'centroid_xs': [p['x'], p['s']]})
        return {'config': small_cfg, 'items': items}
    m = change.evaluate_stations(changes, spec(['intact', 'intact']), spec(['depleted', 'missing']))
    assert m['correct'] == 1 and m['anode_depleted'] == {'true': 1, 'flagged': 1, 'false': 0}
    assert m['anode_lost'] == {'true': 1, 'flagged': 0, 'false': 0}         # S02 lost its anode; the check missed it
