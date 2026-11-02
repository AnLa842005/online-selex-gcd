"""Trusted synthetic orchestration; the bank receives only base labels or safe observations."""
from dataclasses import asdict
import platform

import numpy as np
import scipy
import sklearn
import yaml

from online_gcd.evaluator import OnlineEvaluator
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.synthetic import generate_synthetic


def run_experiment(config, *, consolidation=None, progress=None):
    if consolidation is not None and consolidation.enabled:
        from threadpoolctl import threadpool_limits
        # Match the clean baseline's one-thread assignment/calibration numerics.
        with threadpool_limits(limits=1, user_api='blas'):
            return _run_experiment(config, consolidation=consolidation, progress=progress)
    return _run_experiment(config, consolidation=consolidation, progress=progress)


def _run_experiment(config, *, consolidation, progress):
    data = generate_synthetic(config)
    # Deliberately pass only learner inference settings, never novel count/config.
    if consolidation is not None and consolidation.enabled:
        from online_gcd.artifacts import digest
        from online_gcd.consolidation import ConsolidatingBank, EXPERIMENT
        from online_gcd.consolidation_calibration import calibrate_consolidation
        from online_gcd.consolidation_evaluator import ConsolidationEvaluator
        if progress:
            progress('Consolidation calibration start: labelled known base only')
        import time
        calibration_started = time.perf_counter()
        merge_calibration = calibrate_consolidation(
            data.labelled_base_groups(),
            base_identity=digest([{'sample_id': p.sample_id, 'known_class_id': p.class_id} for p in data.base]),
            cache_identity='synthetic-base:' + digest([p.feature for p in data.base]))
        calibration_seconds = time.perf_counter() - calibration_started
        if progress:
            progress(f'Consolidation calibration end: elapsed={calibration_seconds:.3f}s')
        bank = ConsolidatingBank(config.feature_dim, config.threshold, config.momentum,
                                 merge_calibration['threshold'], consolidation.block_size)
        evaluator_class = ConsolidationEvaluator
    else:
        bank = PrototypeBank(config.feature_dim, config.threshold, config.momentum)
        evaluator_class = OnlineEvaluator
    bank.initialize_known(data.labelled_base_groups())
    evaluator = evaluator_class(data.known_classes, data.stream.class_introduction_schedule, data.probes)
    evaluator.progress = progress
    base = evaluator.evaluate_base(bank)
    sessions = []
    for session_id in range(1, config.num_online_sessions + 1):
        predictions = {}
        for observation in data.learner_session(session_id):
            predictions[observation.sample_id] = bank.observe(
                observation.sample_id, observation.session_id, observation.feature)
        sessions.append(evaluator.evaluate_session(
            session_id, data.stream.evaluator_sessions[session_id - 1], predictions, bank))
    final = sessions[-1]
    result = {
        'schema_version': 1, 'audience': 'evaluator_only',
        'baseline_version': config.baseline_version, 'dataset': 'synthetic',
        'configuration': asdict(config), 'synthetic_checksum': data.checksum,
        'stream_manifest_checksum': data.stream.checksum,
        'environment': {'python': platform.python_version(), 'numpy': np.__version__,
                        'scipy': scipy.__version__, 'scikit_learn': sklearn.__version__, 'pyyaml': yaml.__version__},
        'base_probe': base, 'sessions': sessions,
        'summary': {
            'final_session_accuracy': final['session_local']['hungarian_accuracy']['all']['value'],
            'average_online_accuracy': float(np.mean([
                row['session_local']['hungarian_accuracy']['all']['value'] for row in sessions])),
            'final_cumulative_accuracy': final['cumulative']['hungarian_accuracy'],
            'final_probe_accuracy': final['probe']['fixed_mapping_accuracy'],
            'final_average_forgetting': final['probe']['average_forgetting'],
            'final_identity_churn': final['probe']['identity_churn'],
            'cumulative_novel_nmi': final['cumulative']['novel_nmi'],
            'cumulative_novel_ari': final['cumulative']['novel_ari'],
            'cumulative_fragmentation': final['cumulative']['fragmentation'],
            'active_prototype_count': final['active_prototype_count'],
            'known_prototype_count': final['known_prototype_count'],
            'discovered_novel_cluster_count': final['discovered_novel_cluster_count'],
        },
    }
    if consolidation is not None and consolidation.enabled:
        result.update(experiment=EXPERIMENT, control_ablation=True,
                      assignment_threshold=bank.threshold, consolidation_threshold=bank.consolidation_threshold,
                      consolidation_calibration=merge_calibration,
                      consolidation_calibration_seconds=calibration_seconds, blas_threads=1)
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
    return result, bank
