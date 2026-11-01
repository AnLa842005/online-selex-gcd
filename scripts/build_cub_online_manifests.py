"""Build clean CUB partitions from metadata only. No images or weights are read."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_manifest import build_cub_bundle, save_bundle
from online_gcd.stream import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/online_selex/cub_frozen_dino_v0.yaml')
    parser.add_argument('--cub-root', type=Path, default=os.environ.get('CUB_ROOT'))
    parser.add_argument('--output', type=Path, required=True, help='New manifest directory; parent must exist')
    args = parser.parse_args()
    try:
        if args.cub_root is None:
            raise ValueError('set CUB_ROOT or --cub-root to CUB_200_2011 itself')
        config = load_config(args.config, CleanConfig)
        bundle = build_cub_bundle(args.cub_root, config)
        save_bundle(bundle, args.output)
        for name in ('base', 'stream_learner', 'probe_evaluator'):
            print(f'{name}: {len(bundle[name]["records"])} samples')
        print(f'Partition checksum: {bundle["partition"]["checksum"]}')
        print('Real CUB feature extraction/evaluation: NOT RUN by this command.')
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'Manifest preparation failed: {exc}\n')


if __name__ == '__main__':
    main()
