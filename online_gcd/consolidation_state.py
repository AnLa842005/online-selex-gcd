"""Strict versioned JSON lineage state, separate from unchanged baseline schema."""
from copy import deepcopy
from dataclasses import asdict
import re

from online_gcd.consolidation import ConsolidatingBank, EXPERIMENT
from online_gcd.prototype_bank import BASELINE_VERSION, integer, real


def consolidation_state_dict(bank):
    return {'schema_version': 2, 'baseline_version': EXPERIMENT, 'feature_dim': bank.feature_dim,
            'assignment_threshold': bank.threshold, 'consolidation_threshold': bank.consolidation_threshold,
            'momentum': bank.momentum, 'next_novel_id': bank._next_novel_id, 'last_session': bank.last_session,
            'prototypes': [asdict(p) for p in bank.prototypes], 'aliases': dict(bank.aliases),
            'merge_history': deepcopy(bank.merge_history), 'consolidation_block_size': bank.block_size}


def _novel_id(cid):
    if not isinstance(cid, str) or not re.fullmatch(r'novel-[0-9]{6,}', cid):
        raise ValueError('lineage IDs must be novel IDs')
    number = int(cid[6:])
    if number < 1 or cid != f'novel-{number:06d}':
        raise ValueError('invalid novel ID encoding')


def consolidation_bank_from_state(data, expected_feature_dim=None):
    from online_gcd.state import _baseline_bank_from_state
    fields = {'schema_version', 'baseline_version', 'feature_dim', 'assignment_threshold',
              'consolidation_threshold', 'momentum', 'next_novel_id', 'last_session', 'prototypes',
              'aliases', 'merge_history', 'consolidation_block_size'}
    if set(data) != fields or type(data['schema_version']) is not int or data['schema_version'] != 2 \
            or data['baseline_version'] != EXPERIMENT:
        raise ValueError('invalid consolidation state fields/version')
    aliases, history = data['aliases'], data['merge_history']
    if not isinstance(aliases, dict) or not isinstance(history, list):
        raise ValueError('invalid aliases or merge history')
    for retired, target in aliases.items():
        _novel_id(retired)
        _novel_id(target)
    legacy = {k: deepcopy(data[k]) for k in ('feature_dim', 'momentum', 'next_novel_id', 'last_session', 'prototypes')}
    legacy.update(schema_version=1, baseline_version=BASELINE_VERSION, threshold=data['assignment_threshold'])
    base = _baseline_bank_from_state(legacy, expected_feature_dim, retired_ids=aliases)
    bank = ConsolidatingBank(base.feature_dim, base.threshold, base.momentum,
                             data['consolidation_threshold'], data['consolidation_block_size'])
    bank._prototypes = base._prototypes
    bank._next_novel_id, bank._last_session = base._next_novel_id, base.last_session
    if set(aliases) & set(bank._prototypes):
        raise ValueError('retired IDs cannot still be active')
    bank.aliases = dict(aliases)
    for retired in aliases:
        bank.resolve_alias(retired)  # Explicit missing-target/cycle validation; no mutation.
    edges, last_session = {}, 0
    for row in history:
        if not isinstance(row, dict) or set(row) != {'retired_id', 'survivor_id', 'session_id', 'similarity'}:
            raise ValueError('invalid merge history fields')
        retired, survivor = row['retired_id'], row['survivor_id']
        _novel_id(retired)
        _novel_id(survivor)
        session = integer(row['session_id'], 'merge session', 1)
        similarity = real(row['similarity'], 'merge similarity', -1, 1)
        if (retired in edges or survivor in edges or retired == survivor
                or not last_session <= session <= bank.last_session
                or similarity < bank.consolidation_threshold):
            raise ValueError('inconsistent merge history ordering/threshold')
        edges[retired] = survivor
        last_session = session
    if set(edges) != set(aliases):
        raise ValueError('retired IDs must correspond exactly to explicit merges')
    for retired in edges:
        current, seen = retired, set()
        while current in edges:
            if current in seen:
                raise ValueError('cycle in merge history')
            seen.add(current)
            current = edges[current]
        if current != bank.resolve_alias(retired):
            raise ValueError('alias and merge history disagree')
    bank.merge_history = deepcopy(history)
    bank._rebuild_cache()
    return bank
