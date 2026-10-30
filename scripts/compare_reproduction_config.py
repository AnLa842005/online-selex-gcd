"""Read-only static config comparison; never imports training or datasets."""
import argparse
import ast
import json
from pathlib import Path
import re
import subprocess

ROOT = Path(__file__).resolve().parents[1]
BASE = 'c3d3b42a0aa7d7c9ea748a8305a6890a34100f73'
SOURCE = 'methods/contrastive_training/contrastive_training.py'

def defaults(source):
    result = {}
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == 'add_argument':
            for kw in node.keywords:
                if kw.arg == 'default':
                    try:
                        result[ast.literal_eval(node.args[0]).lstrip('-')] = ast.literal_eval(kw.value)
                    except (ValueError, TypeError):
                        pass
    return result

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--actual-json', type=Path, help='Optional JSON object exported from actual run arguments')
    args = parser.parse_args()
    expected = defaults(subprocess.check_output(['git', 'show', BASE + ':' + SOURCE], cwd=ROOT, text=True))
    actual = defaults((ROOT / SOURCE).read_text())
    launcher = (ROOT / 'scripts/train_cub_server.sh').read_text().split('--dataset_name', 1)[1]
    launcher = '--dataset_name' + launcher.split('"${EXTRA[@]}"', 1)[0] + '\n'
    for key, raw in re.findall(r'--([\w-]+)\s+([^\\\n]+?)(?=\s+--|\s*\\|\n)', launcher):
        raw = raw.strip()
        try:
            value = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            value = raw.split() if key == 'eval_funcs' else raw
        actual[key] = value
    provenance = 'Repository defaults plus server launcher; NOT verified RunPod config'
    if args.actual_json:
        actual = json.loads(args.actual_json.read_text())
        provenance = str(args.actual_json)
    differences = {k: {'expected': v, 'actual': actual.get(k, '<missing>')} for k, v in expected.items() if actual.get(k) != v}
    print(json.dumps({'provenance': provenance, 'expected_source': BASE, 'actual': actual, 'differences': differences,
                      'fixed_behavior': {'scheduler': 'CosineAnnealingLR', 'eta_min': 0.0001, 'warmup_epochs': 0,
                      'split_seed': 0, 'selection': 'disjoint test Old accuracy, FIRST eval_func (v1 by default)'}}, indent=2))

if __name__ == '__main__':
    main()

