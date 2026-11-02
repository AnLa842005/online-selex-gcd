"""Internal fake-cache comparison: no real CUB, model, checkpoint or downloads."""
import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).absolute().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'tests')]
from online_cub_fixtures import fixture_cache, overfragmented_fixture_cache
from online_gcd.artifacts import write_json
from online_gcd.clean_protocol import CleanConfig
from online_gcd.consolidation import ConsolidationOptions
from online_gcd.cub_runner import run_cub_baseline
from online_gcd.state import save_state


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True, help='New directory for artificial fixtures/results only')
    parser.add_argument('--consolidation-block-size', type=int, default=256)
    parser.add_argument('--overfragmented', action='store_true', help='Use artificial two-mode novel classes')
    args = parser.parse_args()
    try:
        options = ConsolidationOptions(True, args.consolidation_block_size)
        args.output_dir.mkdir(exist_ok=False)
        build_fixture = overfragmented_fixture_cache if args.overfragmented else fixture_cache
        _, manifests, cache, _, _ = build_fixture(args.output_dir)
        for name, setting in (('baseline', ConsolidationOptions()), ('consolidation', options)):
            result, bank = run_cub_baseline(manifests, cache, CleanConfig(), allow_fixture=True,
                                            consolidation=setting, progress=lambda line: print(line, flush=True))
            write_json(args.output_dir / (name + '.json'), result)
            save_state(bank, args.output_dir / (name + '_state.json'))
            print(f'{name}: fixture only; prototypes={len(bank.prototypes)} '
                  f'merges={sum(row.get("merge_count", 0) for row in result["sessions"])}')
        print(f'Artificial fixture comparison saved: {args.output_dir.absolute()}')
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Fixture comparison failed: {exc}\n')


if __name__ == '__main__':
    main()
