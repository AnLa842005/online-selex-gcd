"""Import persisted RunPod calibration into a new Phase 3B record, preserving provenance."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from online_gcd.artifacts import read_json, write_json
from online_gcd.result_record import import_calibration


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, default=ROOT / 'results/cub_frozen_dino_online_seed1.json')
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New JSON file; never overwrite original provenance')
    args = parser.parse_args()
    try:
        result = import_calibration(read_json(args.record), read_json(args.calibration))
        write_json(args.output, result)
        print(f'Validated calibration import: {args.output.absolute()}')
    except (ValueError, OSError, KeyError, TypeError) as exc:
        parser.exit(2, f'Calibration import failed: {exc}\n')


if __name__ == '__main__':
    main()
