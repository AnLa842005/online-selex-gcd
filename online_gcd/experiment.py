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


def run_experiment(config):
    data = generate_synthetic(config)
    # Deliberately pass only learner inference settings, never novel count/config.
    bank = PrototypeBank(config.feature_dim, config.threshold, config.momentum)
    bank.initialize_known(data.labelled_base_groups())
    evaluator = OnlineEvaluator(data.known_classes, data.stream.class_introduction_schedule, data.probes)
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
    return result, bank
