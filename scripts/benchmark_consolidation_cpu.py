"""Opt-in synthetic consolidation-kernel benchmark, reference versus optimized."""
import argparse
import json
from pathlib import Path
import statistics
import sys
import time

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
import numpy as np
from online_gcd.state import state_json
from test_online_consolidation import bank_fixture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prototypes', type=int, default=96)
    parser.add_argument('--dimension', type=int, default=64)
    parser.add_argument('--block-size', type=int, default=32)
    parser.add_argument('--repeats', type=int, default=3)
    args = parser.parse_args()
    if min(vars(args).values()) < 1:
        parser.error('sizes and repetitions must be positive')
    rng = np.random.RandomState(123)
    centers = rng.normal(size=(4, args.dimension))
    values = [centers[i % 4] + .1 * rng.normal(size=args.dimension) for i in range(args.prototypes)]
    times = {'reference': [], 'optimized': []}
    for _ in range(args.repeats):
        snapshots = []
        for implementation in times:
            bank = bank_fixture(values, threshold=.9, block_size=args.block_size)
            start = time.perf_counter()
            report = bank.consolidate(implementation=implementation)
            times[implementation].append(time.perf_counter() - start)
            snapshots.append(state_json(bank))
        assert snapshots[0] == snapshots[1], 'kernel state equivalence failed'
    medians = {key: statistics.median(values) for key, values in times.items()}
    print(json.dumps({'fixture': 'Gaussian centers only; not CUB evidence', 'parameters': vars(args),
                      'median_seconds': medians, 'speedup': medians['reference'] / medians['optimized'],
                      'merge_count': report['merge_count'], 'novel_prototypes_after': report['novel_prototypes_after'],
                      'exact_state_and_history': True}, indent=2))


if __name__ == '__main__':
    main()
