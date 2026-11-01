"""Strict JSON learner state: no pickle, evaluator mappings or observation history."""
from dataclasses import asdict
import json
from pathlib import Path
import re

import numpy as np

from online_gcd.prototype_bank import BASELINE_VERSION, Prototype, PrototypeBank, integer, vector


def state_dict(bank):
    return {
        'schema_version': 1, 'baseline_version': BASELINE_VERSION,
        'feature_dim': bank.feature_dim, 'threshold': bank.threshold, 'momentum': bank.momentum,
        'next_novel_id': bank._next_novel_id, 'last_session': bank.last_session,
        'prototypes': [asdict(p) for p in bank.prototypes],
    }


def state_json(bank):
    return json.dumps(state_dict(bank), sort_keys=True, indent=2, allow_nan=False) + '\n'


def save_state(bank, path):
    content = state_json(bank)
    with Path(path).open('x', encoding='utf-8', newline='\n') as handle:
        handle.write(content)


def _unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f'duplicate JSON key: {key}')
        result[key] = value
    return result


def _reject_constant(value):
    raise ValueError(f'invalid JSON number: {value}')


def bank_from_state(data, expected_feature_dim=None):
    fields = {'schema_version', 'baseline_version', 'feature_dim', 'threshold', 'momentum',
              'next_novel_id', 'last_session', 'prototypes'}
    if not isinstance(data, dict) or set(data) != fields:
        raise ValueError('invalid learner state fields')
    if type(data['schema_version']) is not int or data['schema_version'] != 1 or data['baseline_version'] != BASELINE_VERSION:
        raise ValueError('unsupported learner state version')
    bank = PrototypeBank(data['feature_dim'], data['threshold'], data['momentum'])
    if expected_feature_dim is not None:
        integer(expected_feature_dim, 'expected_feature_dim', 1)
        if bank.feature_dim != expected_feature_dim:
            raise ValueError('state feature dimension is incompatible')
    next_id = integer(data['next_novel_id'], 'next_novel_id', 1)
    last = integer(data['last_session'], 'last_session')
    if not isinstance(data['prototypes'], list) or not data['prototypes']:
        raise ValueError('state must contain initialized known prototypes')
    prototype_fields = {'cluster_id', 'vector', 'observation_count', 'originated_known',
                        'creation_session', 'last_updated_session'}
    novel_numbers = set()
    for item in data['prototypes']:
        if not isinstance(item, dict) or set(item) != prototype_fields:
            raise ValueError('invalid prototype fields')
        cid = item['cluster_id']
        if not isinstance(cid, str) or cid in bank._prototypes:
            raise ValueError('invalid or duplicate cluster ID')
        known = item['originated_known']
        if type(known) is not bool:
            raise ValueError('originated_known must be boolean')
        creation = integer(item['creation_session'], 'creation_session')
        updated = integer(item['last_updated_session'], 'last_updated_session')
        count = integer(item['observation_count'], 'observation_count', 1)
        if not creation <= updated <= last:
            raise ValueError('invalid prototype session ordering')
        if updated > creation and count < 2:
            raise ValueError('updated prototype must have more than one observation')
        if known:
            if not re.fullmatch(r'known-(0|[1-9][0-9]*)', cid) or creation != 0:
                raise ValueError('invalid known prototype ID or creation session')
        else:
            if not re.fullmatch(r'novel-[0-9]{6,}', cid) or creation < 1:
                raise ValueError('invalid novel prototype ID or creation session')
            number = int(cid[6:])
            if cid != f'novel-{number:06d}' or number < 1:
                raise ValueError('invalid novel counter encoding')
            novel_numbers.add(number)
        values = vector(item['vector'], bank.feature_dim)
        if not np.isclose(np.linalg.norm(values), 1.0, atol=1e-12, rtol=0):
            raise ValueError('saved prototype vector must already be normalized')
        bank._prototypes[cid] = Prototype(cid, tuple(values), count, known, creation, updated)
    if not any(p.originated_known for p in bank.prototypes):
        raise ValueError('state requires labelled known prototypes')
    if novel_numbers != set(range(1, len(novel_numbers) + 1)) or next_id != len(novel_numbers) + 1:
        raise ValueError('next_novel_id disagrees with the undeleted cluster history')
    if max(p.last_updated_session for p in bank.prototypes) != last:
        raise ValueError('last_session disagrees with prototype metadata')
    bank._next_novel_id, bank._last_session = next_id, last
    return bank


def load_state(path, expected_feature_dim=None):
    try:
        data = json.loads(Path(path).read_text(encoding='utf-8'), object_pairs_hook=_unique_object,
                          parse_constant=_reject_constant)
    except (TypeError, json.JSONDecodeError) as exc:
        raise ValueError(f'invalid learner state JSON: {exc}') from exc
    return bank_from_state(data, expected_feature_dim)
