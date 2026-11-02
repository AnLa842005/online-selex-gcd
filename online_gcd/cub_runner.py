"""Trusted CPU runner: calibrated frozen features, Phase 2 bank and evaluator."""
import platform
import time
import tracemalloc

import numpy as np
import scipy
import sklearn
from threadpoolctl import threadpool_limits

from online_gcd.artifacts import code_commit, verify
from online_gcd.calibration import calibrate_known_base
from online_gcd.cub_manifest import learner_session, load_bundle
from online_gcd.evaluator import OnlineEvaluator
from online_gcd.feature_cache import load_cache
from online_gcd.prototype_bank import PrototypeBank, integer
from online_gcd.state import state_json
from online_gcd.stream import EvaluationRecord
from online_gcd.synthetic import LabelledEmbedding


def run_cub_baseline(manifests, cache, config, *, calibration_record=None, allow_fixture=False, blas_threads=1, progress=print, consolidation=None):
    integer(blas_threads, 'blas_threads', 1)
    # Configurable: tiny GEMV calls often lose time launching many BLAS workers.
    effective_threads = 1 if consolidation is not None and consolidation.enabled else blas_threads
    with threadpool_limits(limits=effective_threads, user_api='blas'):
        return _run_cub_baseline(manifests, cache, config, calibration_record=calibration_record,
                                 allow_fixture=allow_fixture, blas_threads=blas_threads, progress=progress, consolidation=consolidation)


def _run_cub_baseline(manifests, cache, config, *, calibration_record, allow_fixture, blas_threads, progress, consolidation):
    def log(message):
        if progress is not None:
            progress(message)

    started = time.perf_counter()
    log('Feature cache validation start')
    bundle = load_bundle(manifests, config, official=not allow_fixture)
    features, provenance = load_cache(cache, bundle, allow_fixture=allow_fixture)
    log(f'Feature cache validation end: samples={len(features)} elapsed={time.perf_counter() - started:.3f}s')
    p = bundle['partition']
    groups = {c: [] for c in p['known_classes']}
    # Calibration gets only the labelled base projection, no stream/probe data.
    for row in bundle['base']['records']:
        groups[row['known_class_id']].append(features[row['feature_row']])
    log('Threshold calibration start (labelled known base only)')
    calibration_started = time.perf_counter()
    calibration = calibrate_known_base(groups, base_identity=bundle['base']['checksum'],
                                        cache_identity=provenance['checksum'])
    if calibration_record is not None:
        verify(calibration_record)
        if calibration_record != calibration:
            raise ValueError('calibration does not match known base features, cache or deterministic objective')
    log(f'Threshold calibration end: threshold={calibration["threshold"]:.17g} elapsed={time.perf_counter() - calibration_started:.3f}s')
    if consolidation is not None and consolidation.enabled:
        from online_gcd.consolidation import ConsolidatingBank, EXPERIMENT
        from online_gcd.consolidation_calibration import calibrate_consolidation
        from online_gcd.consolidation_evaluator import ConsolidationEvaluator
        log('Consolidation calibration start (labelled known base only)')
        merge_started = time.perf_counter()
        merge_calibration = calibrate_consolidation(groups, base_identity=bundle['base']['checksum'],
                                                   cache_identity=provenance['checksum'])
        merge_calibration_seconds = time.perf_counter() - merge_started
        log(f'Consolidation calibration end: threshold={merge_calibration["threshold"]:.17g} elapsed={merge_calibration_seconds:.3f}s')
        bank = ConsolidatingBank(768, calibration['threshold'], 0.9, merge_calibration['threshold'], consolidation.block_size)
        evaluator_class = ConsolidationEvaluator
    else:
        bank = PrototypeBank(768, threshold=calibration['threshold'], momentum=0.9)
        evaluator_class = OnlineEvaluator
    log('Known prototype initialization start')
    bank.initialize_known(groups)
    log(f'Known prototype initialization end: prototypes={len(bank.prototypes)}')
    probes = tuple(LabelledEmbedding(r['sample_id'], r['class_id'], features[r['feature_row']])
                   for r in bundle['probe_evaluator']['records'])
    evaluator = evaluator_class(p['known_classes'], p['class_introduction_schedule'], probes)
    evaluator.progress = log
    base_probe = evaluator.evaluate_base(bank)
    sessions = []
    started_tracing = not tracemalloc.is_tracing()
    if started_tracing:
        tracemalloc.start()
    try:
        for session_id in range(1, 6):
            session_started = time.perf_counter()
            log(f'Session {session_id} start: prototypes={len(bank.prototypes)}')
            predictions = {}
            for row in learner_session(bundle, session_id):
                predictions[row['sample_id']] = bank.observe(row['sample_id'], row['session_id'],
                                                             features[row['feature_row']])
            processed = time.perf_counter()
            records = tuple(EvaluationRecord(r['sample_id'], session_id, r['class_id'], None, False)
                            for r in bundle['stream_evaluator']['records'] if r['session_id'] == session_id)
            report = evaluator.evaluate_session(session_id, records, predictions, bank)
            report['performance'] = {'processing_seconds': processed - session_started,
                                     'evaluation_seconds': time.perf_counter() - processed,
                                     'stream_assignment_seconds': processed - session_started,
                                     **evaluator.last_timing,
                                     'prototype_vector_payload_bytes': len(bank.prototypes) * 768 * 8,
                                     'learner_state_json_bytes': len(state_json(bank).encode('utf-8')),
                                     'peak_traced_python_bytes': tracemalloc.get_traced_memory()[1]}
            report['performance']['total_session_seconds'] = time.perf_counter() - session_started
            log(f'Session {session_id} end: prototypes={len(bank.prototypes)} elapsed={report["performance"]["total_session_seconds"]:.3f}s')
            sessions.append(report)
    finally:
        if started_tracing:
            tracemalloc.stop()
    final = sessions[-1]
    result = {'schema_version': 1,
              'experiment': 'fixture_only' if provenance['extraction']['kind'] == 'fixture' else 'frozen_DINO_clean_online',
              'audience': 'evaluator_only', 'configuration': config.snapshot(),
              'repository_commit': code_commit(), 'manifest_checksum': p['checksum'],
              'cache_checksum': provenance['checksum'], 'extraction_provenance': provenance,
              'calibration': calibration, 'base_probe': base_probe, 'sessions': sessions,
              'summary': {'final_session_accuracy': final['session_local']['hungarian_accuracy'],
                          'average_online_accuracy': float(np.mean([r['session_local']['hungarian_accuracy']['all']['value']
                                                                   for r in sessions])),
                          'cumulative_accuracy': final['cumulative']['hungarian_accuracy'],
                          'probe_accuracy': final['probe']['fixed_mapping_accuracy'],
                          'average_forgetting': final['probe']['average_forgetting'],
                          'active_prototype_count': final['active_prototype_count'],
                          'discovered_novel_cluster_count': final['discovered_novel_cluster_count']},
              'performance': {'total_seconds': time.perf_counter() - started,
                              'blas_threads': blas_threads,
                              'feature_payload_bytes': int(features.nbytes),
                              'memory_note': 'tracemalloc peak includes evaluator; excludes untraced native/RSS memory'},
              'versions': {'python': platform.python_version(), 'numpy': np.__version__,
                           'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__}}
    if consolidation is not None and consolidation.enabled:
        result.update(experiment=EXPERIMENT, control_ablation=True,
                      input_kind=provenance['extraction']['kind'],
                      assignment_threshold=bank.threshold, consolidation_threshold=bank.consolidation_threshold,
                      consolidation_calibration=merge_calibration,
                      consolidation_calibration_seconds=merge_calibration_seconds)
        result['performance'].update(blas_threads=1, requested_blas_threads=blas_threads)
        result['consolidation_policy'] = {'enabled': True, 'block_size': consolidation.block_size,
                                          'implementation': 'blockwise_row_best_v0',
                                          'scope': 'novel_to_novel_only',
                                          'pair_age': 'earliest_member_creation_session',
                                          'probe_mapping': 'survivor_mapping_then_lexical_retired_mapping'}
        result['summary'].update(retired_cluster_count=len(bank.aliases),
                                 ever_created_novel_cluster_count=bank._next_novel_id - 1,
                                 maximum_merge_chain_depth=bank.max_merge_chain_depth(),
                                 raw_historical_fragmentation=final['raw_historical_fragmentation'],
                                 active_probe_fragmentation=final['active_probe_fragmentation'])
    del features
    return result, bank
