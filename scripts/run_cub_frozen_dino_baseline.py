"""CPU-only clean frozen-DINO CUB baseline on verified feature caches."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_gcd.artifacts import read_json, write_json
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.state import save_state
from online_gcd.stream import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/online_selex/cub_frozen_dino_v0.yaml')
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output-json', type=Path, required=True)
    parser.add_argument('--calibration-json', type=Path, help='Optional previously saved matching calibration')
    parser.add_argument('--save-calibration', type=Path)
    parser.add_argument('--save-state', type=Path)
    parser.add_argument('--blas-threads', type=int, default=1, help='Baseline BLAS threads; consolidation fixes 1 for repeatable numerics')
    parser.add_argument('--consolidate-novel', action='store_true', help='Opt-in control ablation; not Online SelEx')
    parser.add_argument('--consolidation-block-size', type=int, default=256)
    parser.add_argument('--save-consolidation-calibration', type=Path)
    args = parser.parse_args()
    try:
        outputs = [p.resolve() for p in (args.output_json, args.save_calibration, args.save_state, args.save_consolidation_calibration) if p is not None]
        if len(set(outputs)) != len(outputs) or any(p.exists() or not p.parent.is_dir() for p in outputs):
            raise ValueError('outputs must be distinct new files in existing directories')
        from online_gcd.consolidation import ConsolidationOptions
        consolidation = ConsolidationOptions(args.consolidate_novel, args.consolidation_block_size)
        if args.save_consolidation_calibration and not consolidation.enabled:
            raise ValueError('--save-consolidation-calibration requires --consolidate-novel')
        config = load_config(args.config, CleanConfig)
        calibration = read_json(args.calibration_json) if args.calibration_json else None
        result, bank = run_cub_baseline(args.manifests, args.cache, config, calibration_record=calibration, blas_threads=args.blas_threads,
                                        progress=lambda message: print(message, flush=True), consolidation=consolidation)
        print('Novel-prototype consolidation control (not Online SelEx)' if consolidation.enabled else
              'Frozen-DINO clean online baseline (CPU; not synthetic or full-SelEx upper bound)')
        if consolidation.enabled:
            print('Stream scores use raw emitted IDs; probes use the consolidated bank; effective BLAS threads=1')
        print('Session  All     Old     New     Prototypes  Processing(s)')
        for row in result['sessions']:
            a = row['session_local']['hungarian_accuracy']
            values = ['N/A' if a[k]['value'] is None else f'{100*a[k]["value"]:.2f}' for k in ('all', 'old', 'new')]
            print(row['session_id'], *values, row['active_prototype_count'], row['performance']['processing_seconds'])
            if consolidation.enabled:
                print(f'  Novel {row["novel_prototypes_before"]} -> {row["novel_prototypes_after"]}; '
                      f'merges={row["merge_count"]}; consolidation={row["consolidation_time_seconds"]:.3f}s')
        print(json.dumps(result['summary'], indent=2))
        write_json(args.output_json, result)
        if args.save_calibration:
            write_json(args.save_calibration, result['calibration'])
        if args.save_consolidation_calibration:
            write_json(args.save_consolidation_calibration, result['consolidation_calibration'])
        if args.save_state:
            save_state(bank, args.save_state)
        for output in outputs:
            print(f'Saved: {output}', flush=True)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Clean CUB baseline failed: {exc}\n')


if __name__ == '__main__':
    main()
