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
    args = parser.parse_args()
    try:
        outputs = [p.resolve() for p in (args.output_json, args.save_calibration, args.save_state) if p is not None]
        if len(set(outputs)) != len(outputs) or any(p.exists() or not p.parent.is_dir() for p in outputs):
            raise ValueError('outputs must be distinct new files in existing directories')
        config = load_config(args.config, CleanConfig)
        calibration = read_json(args.calibration_json) if args.calibration_json else None
        result, bank = run_cub_baseline(args.manifests, args.cache, config, calibration_record=calibration)
        print('Frozen-DINO clean online baseline (CPU; not synthetic or full-SelEx upper bound)')
        print('Session  All     Old     New     Prototypes  Processing(s)')
        for row in result['sessions']:
            a = row['session_local']['hungarian_accuracy']
            values = ['N/A' if a[k]['value'] is None else f'{100*a[k]["value"]:.2f}' for k in ('all', 'old', 'new')]
            print(row['session_id'], *values, row['active_prototype_count'], row['performance']['processing_seconds'])
        print(json.dumps(result['summary'], indent=2))
        write_json(args.output_json, result)
        if args.save_calibration:
            write_json(args.save_calibration, result['calibration'])
        if args.save_state:
            save_state(bank, args.save_state)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Clean CUB baseline failed: {exc}\n')


if __name__ == '__main__':
    main()
