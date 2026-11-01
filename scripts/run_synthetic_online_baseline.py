"""CPU-only synthetic prototype smoke experiment; never loads CUB or a checkpoint."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from online_gcd.experiment import run_experiment
from online_gcd.state import save_state
from online_gcd.stream import load_config
from online_gcd.synthetic import SyntheticConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=ROOT / 'configs/online_selex/synthetic_prototype_v0.yaml')
    parser.add_argument('--seed', type=int)
    parser.add_argument('--output-json', type=Path)
    parser.add_argument('--save-state', type=Path)
    args = parser.parse_args()
    try:
        config = load_config(args.config, SyntheticConfig)
        if args.seed is not None:
            config = replace(config, seed=args.seed)
        outputs = [path.resolve() for path in (args.output_json, args.save_state) if path is not None]
        if len(set(outputs)) != len(outputs):
            raise ValueError('result and learner state must use separate output paths')
        for path in outputs:
            if path.exists() or not path.parent.is_dir():
                raise ValueError(f'output must be a new file in an existing directory: {path}')
        result, bank = run_experiment(config)
        print(f'{config.baseline_version} | synthetic only | CPU | seed={config.seed}')
        print('Session  All     Old     New     CumAll  Probe   Clusters  Novel  Forget')
        for row in result['sessions']:
            local = row['session_local']['hungarian_accuracy']
            def percent(value):
                return 'N/A' if value is None else f'{100 * value:.2f}'
            values = [local[k]['value'] for k in ('all', 'old', 'new')]
            values += [row['cumulative']['hungarian_accuracy']['all']['value'],
                       row['probe']['fixed_mapping_accuracy']['all']['value']]
            print(f"{row['session_id']:7}  " + '  '.join(f'{percent(v):>6}' for v in values)
                  + f"  {row['active_prototype_count']:8}  {row['discovered_novel_cluster_count']:5}  "
                  + percent(row['probe']['average_forgetting']))
        print('Summary (accuracy fractions):')
        print(json.dumps(result['summary'], sort_keys=True, indent=2, allow_nan=False))
        print('Infrastructure validation only; not comparable to SelEx paper results.')
        if args.output_json:
            with args.output_json.open('x', encoding='utf-8', newline='\n') as handle:
                handle.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + '\n')
        if args.save_state:
            save_state(bank, args.save_state)
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Synthetic baseline failed: {exc}\n')


if __name__ == '__main__':
    main()
