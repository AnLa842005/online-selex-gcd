"""Strict schema 3 for active Euclidean hierarchy/evidence; no retired registry."""
from dataclasses import asdict
import re


from online_gcd.prototype_bank import BASELINE_VERSION, integer, real
from online_gcd.selex_euclidean import SelexEuclideanBank, SelexOptions, experiment_identity


def selex_state_dict(bank):
    if bank._collect_only:
        raise ValueError('transient pseudo-calibration learner is not a runtime checkpoint')
    from online_gcd.artifacts import digest
    settings = bank.options.snapshot()
    return {'schema_version': 3, 'method_version': experiment_identity(bank.options), 'feature_dim': bank.feature_dim,
            'assignment_threshold': bank.threshold, 'structure_threshold': bank.structure_threshold,
            'geometry_threshold': bank.geometry_threshold, 'options': settings, 'config_checksum': digest(settings),
            'calibration_identities': dict(bank.calibration_identities),
            'next_novel_id': bank._next_novel_id, 'last_session': bank.last_session,
            'prototypes': [asdict(p) for p in bank.prototypes],
            'edges': [{'members': list(pair), 'evidence_sum': total, 'opportunities': count}
                      for pair, (total, count) in sorted(bank.edges.items())],
            'nodes': [{'node_id': nid, 'members': list(members)} for nid, members in sorted(bank.nodes.items())],
            'next_node_id': bank.next_node_id, 'base_observation_count': bank.base_observation_count,
            'observation_steps': bank.observation_steps,
            'pruned_edge_count': bank.pruned_edge_count, 'last_rebuild_session': bank.last_rebuild_session}


def selex_bank_from_state(data, expected_feature_dim=None):
    from online_gcd.artifacts import digest
    from online_gcd.state import _baseline_bank_from_state
    fields = {'schema_version', 'method_version', 'feature_dim', 'assignment_threshold', 'structure_threshold',
              'geometry_threshold', 'options', 'config_checksum', 'calibration_identities', 'next_novel_id',
              'last_session', 'prototypes', 'edges', 'nodes', 'next_node_id', 'observation_steps',
              'pruned_edge_count', 'last_rebuild_session', 'base_observation_count'}
    if set(data) != fields or type(data['schema_version']) is not int or data['schema_version'] != 3:
        raise ValueError('invalid Euclidean state fields/version')
    settings = data['options']
    if not isinstance(settings, dict) or set(settings) != set(SelexOptions().snapshot()):
        raise ValueError('invalid Euclidean options')
    options = SelexOptions(**{key: settings[key] for key in ('mode', 'mixing', 'hierarchy_enabled', 'block_size')})
    if data['method_version'] != experiment_identity(options):
        raise ValueError('method identity/options mismatch')
    if (settings != options.snapshot() or any(type(settings[k]) is not type(v) for k,v in options.snapshot().items())
            or data['config_checksum'] != digest(settings)):
        raise ValueError('frozen options/config checksum mismatch')
    identities = data['calibration_identities']
    if not isinstance(identities, dict) or not set(identities) <= {'assignment', 'structure', 'geometry'} \
            or any(not isinstance(v, str) or re.fullmatch('[0-9a-f]{64}', v) is None for v in identities.values()):
        raise ValueError('invalid calibration identities')
    legacy = {k: data[k] for k in ('feature_dim', 'next_novel_id', 'last_session', 'prototypes')}
    legacy.update(schema_version=1, baseline_version=BASELINE_VERSION, threshold=data['assignment_threshold'], momentum=.9)
    base = _baseline_bank_from_state(legacy, expected_feature_dim)
    bank = SelexEuclideanBank(base.feature_dim, base.threshold, structure_threshold=data['structure_threshold'],
                              geometry_threshold=data['geometry_threshold'], options=options, calibration_identities=identities)
    bank._prototypes = base._prototypes
    bank._next_novel_id, bank._last_session = base._next_novel_id, base.last_session
    bank.next_node_id = integer(data['next_node_id'], 'next_node_id', 1)
    bank.base_observation_count = integer(data['base_observation_count'], 'base_observation_count', 1)
    bank.observation_steps = integer(data['observation_steps'], 'observation_steps')
    bank.pruned_edge_count = integer(data['pruned_edge_count'], 'pruned_edge_count')
    bank.last_rebuild_session = integer(data['last_rebuild_session'], 'last_rebuild_session')
    if bank.last_rebuild_session > bank.last_session:
        raise ValueError('hierarchy session is in the future')
    base_count = sum(p.observation_count for p in bank.prototypes) - bank.observation_steps
    if (base_count != bank.base_observation_count
            or base_count < sum(p.originated_known for p in bank.prototypes)
            or bool(bank.observation_steps) != bool(bank.last_session)):
        raise ValueError('observation count/session metadata inconsistent')

    def members(value, minimum, maximum):
        if not isinstance(value, list) or not minimum <= len(value) <= maximum \
                or any(not isinstance(cid, str) for cid in value) or value != sorted(set(value)):
            raise ValueError('invalid sorted unique membership')
        if any(cid not in bank._prototypes for cid in value):
            raise ValueError('missing leaf or hierarchy cycle: node references must be leaves')
        if any(bank._prototypes[cid].originated_known for cid in value):
            raise ValueError('known leaf cannot participate in structure')
        return tuple(value)

    if not isinstance(data['edges'], list) or not isinstance(data['nodes'], list):
        raise ValueError('edges/nodes must be lists')
    degree = {}
    for edge in data['edges']:
        if not isinstance(edge, dict) or set(edge) != {'members', 'evidence_sum', 'opportunities'}:
            raise ValueError('invalid evidence fields')
        pair = members(edge['members'], 2, 2)
        if pair in bank.edges:
            raise ValueError('duplicate evidence edge')
        count = integer(edge['opportunities'], 'opportunities', 1)
        total = real(edge['evidence_sum'], 'evidence_sum', 0, count)
        if count > bank.observation_steps or total <= 0:
            raise ValueError('invalid evidence/opportunity total')
        bank.edges[pair] = (total, count)
        for cid in pair:
            degree[cid] = degree.get(cid, 0) + 1
            if degree[cid] > 8:
                raise ValueError('evidence degree exceeds eight')
    if bank.options.mode == 'geometry' and bank.edges:
        raise ValueError('geometry ablation must not contain temporal evidence')
    occupied = set()
    for node in data['nodes']:
        if not isinstance(node, dict) or set(node) != {'node_id', 'members'}:
            raise ValueError('invalid node fields')
        nid = node['node_id']
        if not isinstance(nid, str) or re.fullmatch(r'node-[0-9]{6,}', nid) is None:
            raise ValueError('invalid node ID')
        number = int(nid[5:])
        if number < 1 or nid != f'node-{number:06d}' or number >= bank.next_node_id or nid in bank.nodes:
            raise ValueError('duplicate/nonmonotone node ID')
        group = members(node['members'], 2, 8)
        if occupied & set(group):
            raise ValueError('hierarchy memberships overlap')
        if any(bank._prototypes[cid].creation_session > bank.last_rebuild_session for cid in group):
            raise ValueError('node contains a leaf born after rebuild')
        occupied.update(group)
        bank.nodes[nid] = group
    if bank.nodes and (not bank.options.hierarchy_enabled or bank.last_rebuild_session == 0):
        raise ValueError('nodes inconsistent with hierarchy lifecycle')
    bank._rebuild_cache()
    bank._rebuild_membership_cache()
    return bank
