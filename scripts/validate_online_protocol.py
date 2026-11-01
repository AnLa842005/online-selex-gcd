"""CPU-only CUB SSB protocol validation; no images, models or downloads."""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_gcd.ssb import CUB_SPLIT_SHA256, load_cub_ssb_classes
from online_gcd.stream import SampleRecord, build_stream, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/online_selex/cub_5sessions_seed1.yaml')
    parser.add_argument('--metadata', type=Path, help='Evaluator-only JSON array of SampleRecord objects; no image reads')
    parser.add_argument('--output', type=Path, help='New evaluator-only manifest path (parent directory must exist)')
    args = parser.parse_args()
    try:
        config = load_config(args.config)
        known, novel = load_cub_ssb_classes()
        records = None
        if args.metadata:
            values = json.loads(args.metadata.read_text(encoding='utf-8'))
            if not isinstance(values, list) or any(not isinstance(v, dict) for v in values):
                raise ValueError('metadata must be a JSON array of sample objects')
            try:
                records = [SampleRecord(**value) for value in values]
            except TypeError as exc:
                raise ValueError(f'invalid sample metadata fields: {exc}') from exc
        manifest = build_stream(records, known, novel, config, expected_cub=True)
        print(f'{config.protocol_name}: known={len(known)}, novel={len(novel)}, seed={config.seed}')
        print(f'mode={manifest.manifest_kind}; SSB SHA256={CUB_SPLIT_SHA256}')
        for session_id, records in enumerate(manifest.evaluator_sessions, 1):
            introduced = sum(s == session_id for _, s in manifest.class_introduction_schedule)
            seen = sum(s < session_id for _, s in manifest.class_introduction_schedule)
            print(f'Session {session_id}: new classes={introduced}, previously introduced={seen}, '
                  f'known eligible={len(known)}, observations={len(records)}')
        print(f'Manifest SHA256={manifest.checksum}')
        if args.output:
            manifest.export(args.output)
            print(f'Evaluator-only manifest written to {args.output}')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Protocol validation failed: {exc}\n')


if __name__ == '__main__':
    main()
