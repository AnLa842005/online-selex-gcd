"""Trusted fixture-only orchestration for frozen Euclidean evidence learners."""
from collections import defaultdict
import json
from pathlib import Path
import platform
import time

import numpy as np
from threadpoolctl import threadpool_limits, threadpool_info

from online_gcd.artifacts import digest, read_json, verify, file_sha256
from online_gcd.calibration import calibrate_known_base
from online_gcd.evaluator import OnlineEvaluator
from online_gcd.metrics import fragmentation
from online_gcd.selex_calibration import calibrate_selex_base
from online_gcd.selex_euclidean import EXPERIMENT, GEOMETRY_EXPERIMENT, SelexEuclideanBank, SelexOptions, experiment_identity
from online_gcd.state import state_dict, state_json
from online_gcd.synthetic import LabelledEmbedding
from online_gcd.selex_synthetic import generate_selex_synthetic as generate_synthetic


def stable_result(value):
    """Semantic artifacts are byte-stable; wall clocks live in another envelope."""
    if isinstance(value, dict):
        return {k: stable_result(v) for k, v in value.items()
                if k != 'performance' and not k.endswith('_seconds')}
    if isinstance(value, (list, tuple)):
        return [stable_result(v) for v in value]
    return value


def collapse_report(labels, predictions):
    classes = defaultdict(set)
    for label, prediction in zip(labels, predictions):
        if prediction is not None:
            classes[prediction].add(label)
    return {'impure_prediction_cluster_count': sum(len(v) > 1 for v in classes.values()),
            'class_merge_excess': sum(max(0, len(v) - 1) for v in classes.values())}


class SelexEvaluator(OnlineEvaluator):
    def __init__(self, *args, reference=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.reference = reference

    def _before_session_probe(self, bank, session_id):
        if self.progress:
            self.progress(f'Hierarchy {session_id} start: leaves={len(bank.prototypes)}')
        self.last_structure = bank.rebuild(reference=self.reference)
        if self.progress:
            self.progress(f'Hierarchy {session_id} end: nodes={len(bank.nodes)} edges={self.last_structure["structural_edge_count"]}')

    def _probe_extras(self, labels, predictions):
        return {'active_probe_fragmentation': fragmentation(labels, predictions),
                'incorrect_collapse': collapse_report(labels, predictions)}

    def evaluate_session(self, session_id, records, predictions_by_id, bank):
        records = tuple(records)
        report = super().evaluate_session(session_id, records, predictions_by_id, bank)
        report.update(self.last_structure)
        report.update(raw_stream_accuracy=report['session_local']['hungarian_accuracy'],
                      raw_stream_fragmentation=report['session_local']['fragmentation'],
                      raw_historical_fragmentation=report['cumulative']['fragmentation'],
                      active_probe_fragmentation=report['probe']['active_probe_fragmentation'],
                      incorrect_collapse=collapse_report([r.class_id for r in records],
                                                         [predictions_by_id[r.sample_id] for r in records]),
                      leaf_retirement_count=0)
        self.last_timing['metric_computation_seconds'] = max(
            0., self.last_timing['metric_computation_seconds'] - self.last_structure['hierarchy_update_seconds'])
        return report


def _calibrate(base, feature_dim, options):
    groups = {c: [feature for _, feature in rows] for c, rows in base.items()}
    # Identities deliberately cover known-base projections only. Full cache or
    # partition hashes include future observations and stay in runner provenance.
    base_identity = digest({str(c): [sid for sid, _ in rows] for c, rows in base.items()})
    feature_identity = digest({str(c): [np.asarray(x).tolist() for _, x in rows] for c, rows in base.items()})
    assignment = calibrate_known_base(groups, base_identity=base_identity, cache_identity=feature_identity)
    threshold = assignment['threshold']
    structural = calibrate_selex_base(base, threshold, base_identity=base_identity, cache_identity=feature_identity,
                                     mode=options.mode, block_size=options.block_size)
    identities = {'assignment': assignment['checksum'],
                  'structure' if options.mode == 'temporal' else 'geometry': structural['checksum']}
    bank = SelexEuclideanBank(feature_dim, threshold,
                             structure_threshold=structural['threshold'] if options.mode == 'temporal' else None,
                             geometry_threshold=structural['threshold'] if options.mode == 'geometry' else None,
                             options=options, calibration_identities=identities)
    bank.initialize_known(groups)
    return bank, assignment, structural


def _run(base, feature_dim, session_source, known, introduction, probes, *, options,
         progress=None, reference=False, source=None, snapshot_callback=None):
    def log(message):
        if progress:
            progress(message)
    started = time.perf_counter()
    log('Known-base assignment/structural calibration start (no evaluator inputs)')
    bank, assignment, structural = _calibrate(base, feature_dim, options)
    calibrated = time.perf_counter()
    log(f'Calibration end: assignment={bank.threshold:.17g} '
        f'{options.mode}={structural["threshold"]:.17g}')
    evaluator = SelexEvaluator(known, introduction, probes, reference=reference)
    evaluator.progress = log
    base_probe = evaluator.evaluate_base(bank)
    base_probe_seconds = evaluator.last_timing['probe_evaluation_seconds']
    sessions, performance = [], []
    birth_origins = {f'known-{c}': c for c in known}  # Privileged evaluator-only diagnostic.
    for session_id, observations, records in session_source:
        tick = time.perf_counter()
        log(f'Session {session_id} start')
        predictions = {}
        assignment_seconds, evidence_seconds = 0., 0.
        for observation in observations:
            predictions[observation.sample_id] = bank.observe(observation.sample_id, session_id, observation.feature)
            assignment_seconds += bank.last_timing['assignment_seconds']
            evidence_seconds += bank.last_timing['structural_evidence_seconds']
        # No labels or future schedule enter observe, calibration or rebuild.
        for record in records:
            birth_origins.setdefault(predictions[record.sample_id], record.class_id)
        report = evaluator.evaluate_session(session_id, records, predictions, bank)
        report['structural_mixed_origin_group_count'] = sum(
            len({birth_origins[cid] for cid in members}) > 1 for members in bank.nodes.values())
        report['learner_state_json_bytes'] = len(state_json(bank).encode('utf-8'))
        exported = state_dict(bank)
        report['hierarchy_evidence_json_bytes'] = len(json.dumps(
            {'nodes':exported['nodes'],'edges':exported['edges']},sort_keys=True).encode('utf-8'))
        report['prototype_vector_payload_bytes'] = len(bank.prototypes) * feature_dim * 8
        times = {'session_id': session_id, 'assignment_seconds': assignment_seconds,
                 'structural_evidence_seconds': evidence_seconds,
                 'hierarchy_update_seconds': report['hierarchy_update_seconds'],
                 'structural_similarity_seconds': report['structural_similarity_seconds'],
                 **evaluator.last_timing, 'total_session_seconds': time.perf_counter() - tick}
        performance.append(times)
        sessions.append(report)
        if snapshot_callback:
            snapshot_callback(session_id, state_dict(bank))
        log(f'Session {session_id} end: leaves={len(bank.prototypes)} nodes={len(bank.nodes)}')
    if not sessions:
        raise ValueError('at least one session required')
    final = sessions[-1]
    code_root=Path(__file__).absolute().parent
    code_identity=digest({p.name:file_sha256(p) for p in sorted(code_root.glob('selex_*.py'))})
    result = {'schema_version': 1, 'experiment': experiment_identity(options),
              'audience': 'evaluator_only', 'validation_only': True, 'source': source,
              'implementation_sources_sha256':code_identity,
              'versions':{'python':platform.python_version(),'numpy':np.__version__,
                          'blas':[{k:lib.get(k) for k in ('internal_api','version','num_threads','architecture')}
                                  for lib in threadpool_info() if lib.get('user_api')=='blas']},
              'settings': options.snapshot(), 'assignment_threshold': bank.threshold,
              'structure_threshold': bank.structure_threshold, 'geometry_threshold': bank.geometry_threshold,
              'assignment_calibration': assignment, 'structural_calibration': structural,
              'base_probe': base_probe, 'sessions': sessions,
              'summary': {'final_session_accuracy': final['session_local']['hungarian_accuracy'],
                          'average_online_accuracy': float(np.mean([r['raw_stream_accuracy']['all']['value'] for r in sessions])),
                          'cumulative_accuracy': final['cumulative']['hungarian_accuracy'],
                          'probe_accuracy': final['probe']['fixed_mapping_accuracy'],
                          'average_forgetting': final['probe']['average_forgetting'],
                          'active_prototype_count': len(bank.prototypes), 'active_structural_node_count': len(bank.nodes),
                          'structural_edge_count': final['structural_edge_count'],
                          'discovered_novel_cluster_count': sum(not p.originated_known for p in bank.prototypes),
                          'raw_historical_fragmentation': final['raw_historical_fragmentation'],
                          'active_probe_fragmentation': final['active_probe_fragmentation'],
                          'incorrect_collapse': final['probe']['incorrect_collapse']},
              'performance': {'calibration_seconds': calibrated - started, 'base_probe_seconds': base_probe_seconds,
                              'sessions': performance, 'total_seconds': time.perf_counter() - started,
                              'blas_threads': 1, 'memory_note': 'state/vector payload bytes, not peak RSS'}}
    return result, bank


def run_selex_synthetic(config, *, options=None, progress=None, reference=False, snapshot_callback=None):
    with threadpool_limits(limits=1, user_api='blas'):
        data = generate_synthetic(config)
        base = {c: [(p.sample_id, p.feature) for p in data.base if p.class_id == c] for c in data.known_classes}
        source = ((sid, data.learner_session(sid), data.stream.evaluator_sessions[sid - 1])
                  for sid in range(1, config.num_online_sessions + 1))
        return _run(base, config.feature_dim, source, data.known_classes, data.stream.class_introduction_schedule,
                    data.probes, options=options or SelexOptions(),
                    progress=progress, reference=reference, snapshot_callback=snapshot_callback,
                    source={'kind': 'synthetic_embeddings', 'seed': config.seed, 'checksum': data.checksum})


def run_selex_fake_cache(manifests, cache, config, *, options=None, progress=None, reference=False, snapshot_callback=None):
    from online_gcd.cub_manifest import learner_session, load_bundle
    from online_gcd.feature_cache import load_cache
    from online_gcd.stream import EvaluationRecord
    from online_gcd.synthetic import FeatureObservation
    with threadpool_limits(limits=1, user_api='blas'):
        metadata = verify(read_json(Path(cache) / 'cache.json'))
        if not isinstance(metadata.get('extraction'),dict) or metadata['extraction'].get('kind') != 'fixture':
            raise ValueError('Phase 4B-1 runner accepts artificial fixture caches only; real CUB is deferred')
        bundle = load_bundle(manifests, config, official=False)
        features, provenance = load_cache(cache, bundle, allow_fixture=True)
        if provenance['extraction']['kind'] != 'fixture':
            raise ValueError('Phase 4B-1 runner accepts artificial fixture caches only; real CUB is deferred')
        known = bundle['partition']['known_classes']
        base = {c: [(r['sample_id'], features[r['feature_row']]) for r in bundle['base']['records'] if r['known_class_id'] == c]
                for c in known}
        probes = tuple(LabelledEmbedding(r['sample_id'], r['class_id'], features[r['feature_row']])
                       for r in bundle['probe_evaluator']['records'])
        def session_source():
            for sid in range(1, 6):
                observations = tuple(FeatureObservation(r['sample_id'], sid, features[r['feature_row']])
                                     for r in learner_session(bundle, sid))
                records = tuple(EvaluationRecord(r['sample_id'], sid, r['class_id'], None, False)
                                for r in bundle['stream_evaluator']['records'] if r['session_id'] == sid)
                yield sid, observations, records
        return _run(base, 768, session_source(), known, bundle['partition']['class_introduction_schedule'], probes,
                    options=options or SelexOptions(), progress=progress, reference=reference,
                    snapshot_callback=snapshot_callback,
                    source={'kind': 'fake_cache', 'cache_checksum': provenance['checksum'], 'configuration': config.snapshot()})


def comparison_snapshot(result, bank, probes):
    """Evaluator-only reporting shared by all fixture arms, no label feedback."""
    before = state_json(bank)
    predictions = bank.predict_batch([p.feature for p in probes])
    if state_json(bank) != before:
        raise RuntimeError('comparison probes mutated learner')
    labels = [p.class_id for p in probes]
    summary = result['summary']
    return {'final_stream_accuracy': summary['final_session_accuracy'],
            'average_online_accuracy': summary['average_online_accuracy'],
            'cumulative_accuracy': summary['cumulative_accuracy'],
            'probe_accuracy': summary['probe_accuracy'], 'average_forgetting':summary['average_forgetting'],
            'active_prototype_count':len(bank.prototypes),
            'active_structural_node_count':len(getattr(bank,'nodes',{})),
            'structural_edge_count':summary.get('structural_edge_count',0),
            'raw_historical_fragmentation':result['sessions'][-1]['cumulative']['fragmentation'],
            'active_probe_fragmentation':fragmentation(labels,predictions),
            'incorrect_collapse':collapse_report(labels,predictions),
            'explicit_leaf_merge_count':len(getattr(bank,'aliases',{})),
            'learner_state_json_bytes':len(before.encode('utf-8')),
            'prototype_vector_payload_bytes':len(bank.prototypes)*bank.feature_dim*8}


def performance_envelope(result):
    performance = dict(result['performance'])
    if 'sessions' not in performance:
        performance['sessions'] = [dict(row['performance'],session_id=row['session_id']) for row in result['sessions']]
    return performance
