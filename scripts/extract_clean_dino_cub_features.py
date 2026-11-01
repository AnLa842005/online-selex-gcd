"""Extract frozen official DINO CLS features; no downloads or SelEx checkpoint path."""
import argparse
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_gcd.clean_dino import extract
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_manifest import load_bundle
from online_gcd.stream import load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/online_selex/cub_frozen_dino_v0.yaml')
    parser.add_argument('--cub-root', type=Path, default=os.environ.get('CUB_ROOT'))
    parser.add_argument('--checkpoint', type=Path, default=os.environ.get('DINO_PRETRAINED_PATH'))
    parser.add_argument('--manifests', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New cache directory; parent must exist')
    parser.add_argument('--device', default=os.environ.get('ONLINE_SELEX_GPU', 'cpu'))
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--reuse', action='store_true', help='Validate existing cache; never resumes partial output')
    args = parser.parse_args()
    try:
        if args.cub_root is None or args.checkpoint is None:
            raise ValueError('CUB root and official DINO pretrained checkpoint must be supplied')
        config = load_config(args.config, CleanConfig)
        bundle = load_bundle(args.manifests, config)
        metadata = extract(args.cub_root, args.checkpoint, bundle, config, args.output,
                           device=args.device, batch_size=args.batch_size, workers=args.workers, reuse=args.reuse)
        print(f'Complete cache checksum: {metadata["checksum"]}')
    except (ValueError, OSError, KeyError, RuntimeError) as exc:
        parser.exit(2, f'Clean DINO extraction failed: {exc}\n')


if __name__ == '__main__':
    main()
