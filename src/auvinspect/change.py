"""Change detection: the findings of two surveys of one hull compared, and scored against the truth.

Pairing: findings of the later survey (B) are taken in order of score and paired with the unpaired
finding of the earlier survey (A) of the same class whose hull box overlaps theirs by at least
change.match_overlap of the smaller box (both grown by mapping.merge_gap_m). An anode in A may pair
with a missing-anode footprint in B. Each pair or lone finding is labelled:
  new          a B finding with no partner in A
  grown        a pair whose area grew by at least change.growth_ratio
  unchanged    a pair that did not
  anode_lost   an anode in A where B finds its footprint
  gone         an A finding with no partner in B (repaired, cleaned, or not found this time)
Every A finding is a candidate partner whatever its tier, so a possible finding confirmed later is not
new. A change carries the tier of its B finding (of its A finding when gone).

Scoring (simulation only): the scenarios' items keep their ids between the surveys, so each item's
true change is known (evolution.changes). A label is checked on the B finding's matched item. Reported
for the changes of confirmed findings and for all of them: for new defects, growth and lost anodes,
how many such items the later survey found and how many it labelled so, with the false alarms; the
error of the estimated area ratio on pairs whose findings both match the same item; the label
confusion.
"""
from __future__ import annotations

import json
import os

import numpy as np

from . import evolution
from .mapping import _overlap


def compare(before: list[dict], after: list[dict], cfg: dict) -> list[dict]:
    gap, thr = cfg['mapping']['merge_gap_m'], cfg['change']['match_overlap']

    def compatible(a, b):
        return a['class'] == b['class'] or (a['class'] == 'anode' and b['class'] == 'missing_anode')
    free = {f['id']: f for f in before}
    out = []
    for b in sorted(after, key=lambda f: -f['score']):
        cands = [(_overlap(a['hull_box'], b['hull_box'], gap), a) for a in free.values() if compatible(a, b)]
        ov, a = max(cands, key=lambda c: c[0], default=(0.0, None))
        if a is None or ov < thr:
            out.append({'change': 'new', 'class': b['class'], 'before': None, 'after': b['id'], 'tier': b.get('tier'),
                        'hull_box': b['hull_box'], 'centre': b['centre'], 'area_ratio': None})
            continue
        del free[a['id']]
        ratio = b['area_m2'] / max(a['area_m2'], 1e-9)
        label = ('anode_lost' if a['class'] != b['class'] else
                 'grown' if ratio >= cfg['change']['growth_ratio'] else 'unchanged')
        out.append({'change': label, 'class': b['class'], 'before': a['id'], 'after': b['id'], 'tier': b.get('tier'),
                    'hull_box': b['hull_box'], 'centre': b['centre'], 'area_ratio': round(ratio, 3)})
    for a in free.values():
        out.append({'change': 'gone', 'class': a['class'], 'before': a['id'], 'after': None, 'tier': a.get('tier'),
                    'hull_box': a['hull_box'], 'centre': a['centre'], 'area_ratio': None})
    return out


def evaluate(changes: list[dict], before: list[dict], after: list[dict], spec_a: dict, spec_b: dict, cfg: dict) -> dict:
    truth = evolution.changes(spec_a, spec_b, cfg['change']['growth_ratio'])
    fa, fb = {f['id']: f for f in before}, {f['id']: f for f in after}
    areas_a = {it['id']: it.get('area_m2') for it in spec_a['items']}
    areas_b = {it['id']: it.get('area_m2') for it in spec_b['items']}
    confusion: dict[str, dict[str, int]] = {}
    labelled = {}
    ratio_err = []
    for c in changes:
        if c['after'] is None:
            continue
        item = fb[c['after']].get('match', {}).get('item')
        t = truth.get(item, 'false') if item is not None else 'false'
        confusion.setdefault(t, {}).setdefault(c['change'], 0)
        confusion[t][c['change']] += 1
        if item is not None:
            labelled[item] = c['change']
        if c['before'] is not None and item is not None and fa[c['before']].get('match', {}).get('item') == item \
                and areas_a.get(item) and areas_b.get(item):
            ratio_err.append(abs(np.log(c['area_ratio'] / (areas_b[item] / areas_a[item]))))

    def recall(kind, label):
        ids = [i for i, t in truth.items() if t == kind and i in labelled]   # such items the later survey found
        n = sum(1 for i in ids if labelled[i] == label)
        return {'found': len(ids), 'flagged': n, 'recall': round(n / len(ids), 4) if ids else None,
                'false': sum(row.get(label, 0) for t, row in confusion.items() if t != kind)}
    return {'new': recall('new', 'new'), 'grown': recall('grown', 'grown'), 'anode_lost': recall('anode_lost', 'anode_lost'),
            'area_ratio_log_error_median': round(float(np.median(ratio_err)), 4) if ratio_err else None,
            'true_changes': {k: sum(1 for t in truth.values() if t == k) for k in sorted(set(truth.values()))},
            'confusion': confusion}


def run(before_report: str, after_report: str, cfg: dict) -> tuple[list[dict], dict]:
    """Changes between two reports' findings, scored for confirmed findings and for all of them."""
    def load(rep):
        findings = json.load(open(os.path.join(rep, 'findings.json')))
        manifest = json.load(open(os.path.join(rep, 'manifest.json')))
        spec = json.load(open(os.path.join(manifest['scenario'], 'scenario.json')))
        return findings, spec
    fa, sa = load(before_report)
    fb, sb = load(after_report)
    if sb.get('lineage', {}).get('base_digest') not in (None, sa.get('digest')):
        raise ValueError('the later survey is not of the earlier hull (lineage digest differs)')
    changes = compare(fa, fb, cfg)
    return changes, {'confirmed': evaluate([c for c in changes if c['tier'] == 'confirmed'], fa, fb, sa, sb, cfg),
                     'all': evaluate(changes, fa, fb, sa, sb, cfg)}
