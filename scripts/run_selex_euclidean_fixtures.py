"""Opt-in synthetic/fake-cache Euclidean validation; never accepts real CUB inputs."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from online_gcd.artifacts import write_json
from online_gcd.clean_protocol import CleanConfig
from online_gcd.consolidation import ConsolidationOptions
from online_gcd.selex_euclidean import SelexOptions
from online_gcd.selex_runner import run_selex_fake_cache, run_selex_synthetic, stable_result, comparison_snapshot, performance_envelope
from online_gcd.state import save_state
from online_gcd.synthetic import SyntheticConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fixture', choices=('synthetic', 'hard-cache', 'noisy-cache'), default='synthetic')
    parser.add_argument('--variant', choices=('temporal', 'geometry', 'lambda-zero', 'no-hierarchy'), default='temporal')
    parser.add_argument('--block-size', type=int, default=256)
    parser.add_argument('--seed', type=int, default=1)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for generated fixture artifacts only')
    args = parser.parse_args()
    try:
        options = SelexOptions(mode='geometry' if args.variant == 'geometry' else 'temporal',
                               mixing=0 if args.variant == 'lambda-zero' else .25,
                               hierarchy_enabled=args.variant != 'no-hierarchy', block_size=args.block_size)
        config = SyntheticConfig(feature_dim=768, noise_std=.005, seed=args.seed)
        args.output_dir.mkdir(exist_ok=False)
        outputs = []
        comparison_probes = None
        if args.fixture == 'synthetic':
            result, bank = run_selex_synthetic(config, options=options, progress=lambda line: print(line, flush=True))
            outputs.append(('euclidean', result, bank))
        else:
            from online_gcd.cub_runner import run_cub_baseline
            from selex_fixtures import hard_fake_cache
            from online_cub_fixtures import fixture_cache
            # Fake CUB protocol has fixed seed 1; do not silently ignore --seed.
            if args.seed != 1:
                raise ValueError('fake-cache fixtures use fixed protocol seed 1')
            builder = hard_fake_cache if args.fixture == 'hard-cache' else fixture_cache
            bundle, manifests, cache, features, _ = builder(args.output_dir)
            from online_gcd.synthetic import LabelledEmbedding
            comparison_probes = tuple(LabelledEmbedding(r['sample_id'],r['class_id'],features[r['feature_row']])
                                      for r in bundle['probe_evaluator']['records'])
            for name, consolidation in (('baseline', ConsolidationOptions()),
                                         ('consolidation', ConsolidationOptions(True))):
                result, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                                consolidation=consolidation, progress=None)
                outputs.append((name, result, bank))
            result, bank = run_selex_fake_cache(manifests, cache, CleanConfig(), options=options,
                                                progress=lambda line: print(line, flush=True))
            outputs.append(('euclidean', result, bank))
        for name, result, bank in outputs:
            write_json(args.output_dir / (name + '.json'), stable_result(result))
            write_json(args.output_dir / (name + '_performance.json'), performance_envelope(result))
            save_state(bank, args.output_dir / (name + '_state.json'))
            summary = result['summary']
            print(f'{name}: fixture only; final_all={summary["final_session_accuracy"]["all"]["value"]:.6f} '
                  f'probe_all={summary["probe_accuracy"]["all"]["value"]:.6f} leaves={len(bank.prototypes)}')
        if comparison_probes is not None:
            write_json(args.output_dir / 'comparison.json',
                       {name:comparison_snapshot(result,bank,comparison_probes) for name,result,bank in outputs})
            write_json(args.output_dir / 'comparison_performance.json',
                       {name:performance_envelope(result) for name,result,_ in outputs})
        print(f'Generated fixture artifacts: {args.output_dir.absolute()} (not research results)')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Euclidean fixture validation failed: {exc}\n')


if __name__ == '__main__':
    main()
