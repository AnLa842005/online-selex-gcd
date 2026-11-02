"""Opt-in CPU benchmark on seeded synthetic vectors; no wall-clock test assertion."""
import argparse
import json
from pathlib import Path
import platform
import statistics
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import numpy as np
from threadpoolctl import threadpool_limits
from online_gcd.prototype_bank import PrototypeBank
from online_gcd.state import state_json
from reference_online_v0 import PrototypeBank as ReferenceBank


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototypes', type=int, default=256)
    parser.add_argument('--dimension', type=int, default=768)
    parser.add_argument('--observations', type=int, default=192)
    parser.add_argument('--probes', type=int, default=256)
    parser.add_argument('--repeats', type=int, default=3)
    parser.add_argument('--blas-threads', type=int, default=1)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        parser.error('all sizes, repetitions and thread counts must be positive')
    rng = np.random.RandomState(123)
    groups = {c: [rng.normal(size=args.dimension)] for c in range(args.prototypes)}
    observations = rng.normal(size=(args.observations, args.dimension))
    probes = rng.normal(size=(args.probes, args.dimension))
    timings = {'reference': [], 'optimized': []}
    with threadpool_limits(limits=args.blas_threads, user_api='blas'):
        for _ in range(args.repeats):
            outputs = []
            for name, cls in [('reference', ReferenceBank), ('optimized', PrototypeBank)]:
                bank = cls(args.dimension, threshold=.9)
                bank.initialize_known(groups)
                start = time.perf_counter()
                ids = [bank.observe(str(i), 1, x) for i, x in enumerate(observations)]
                assigned = time.perf_counter()
                predictions = (bank.predict_batch(probes) if name == 'optimized'
                               else [bank.predict(x) for x in probes])
                done = time.perf_counter()
                timings[name].append({'stream_seconds': assigned-start, 'probe_seconds': done-assigned})
                outputs.append((ids, predictions, state_json(bank)))
            assert outputs[0] == outputs[1], 'reference equivalence failed'
    medians = {name: {key: statistics.median(row[key] for row in rows) for key in rows[0]}
               for name, rows in timings.items()}
    print(json.dumps({'fixture': 'seeded_gaussian_only', 'python': platform.python_version(),
                      'numpy': np.__version__, 'parameters': vars(args), 'median_seconds': medians,
                      'speedup': {key: medians['reference'][key]/medians['optimized'][key]
                                  for key in medians['reference']}, 'exact_ids_predictions_state': True}, indent=2))


if __name__ == '__main__':
    main()
