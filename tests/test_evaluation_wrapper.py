import ast
import os
from pathlib import Path
import shlex
import subprocess
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
WRAPPER = (ROOT / 'scripts/evaluate_cub_checkpoints.sh').read_text()


class EvaluationWrapperTests(unittest.TestCase):
    def test_wrapper_cli_contracts(self):
        for module in ('extract_features', 'k_means'):
            with self.subTest(module=module):
                source = (ROOT / 'methods/clustering' / (module + '.py')).read_text()
                supported = {
                    arg.value for node in ast.walk(ast.parse(source))
                    if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                    and node.func.attr == 'add_argument'
                    for arg in node.args if isinstance(arg, ast.Constant)
                }
                command, = [shlex.split(line) for line in WRAPPER.splitlines()
                            if '-m methods.clustering.' + module + ' ' in line]
                flags = {token.partition('=')[0] for token in command if token.startswith('--')}
                self.assertFalse(flags - supported, flags - supported)
                if module == 'k_means':
                    self.assertEqual(command[command.index('--use_ssb_splits') + 1], 'True')
                else:
                    self.assertNotIn('--use_ssb_splits', flags)

    def test_preflight_image_verification(self):
        # Run just the wrapper's Python preflight, never the evaluation commands.
        preflight = WRAPPER.split("<<'PY'\n", 1)[1].split('\nPY\n', 1)[0]
        with tempfile.TemporaryDirectory() as folder:
            base = Path(folder)
            cub = base / 'CUB_200_2011'
            (cub / 'images').mkdir(parents=True)
            (cub / 'attributes').mkdir()
            for name in ('model.pt', 'model_best.pt', 'model_proj_head.pt',
                         'model_proj_head_best.pt', 'dino.pth'):
                (base / name).touch()
            for name in ('image_class_labels.txt', 'train_test_split.txt',
                         'attributes/image_attribute_labels.txt'):
                (cub / name).touch()
            (cub / 'images.txt').write_text('1 first.jpg\n2 missing.jpg\n')
            (cub / 'images/first.jpg').touch()
            env = dict(os.environ, MODEL=str(base / 'model.pt'), SELEX_DATA_ROOT=folder,
                       SELEX_OUTPUT_ROOT=str(base / 'out'), SELEX_PRETRAINED_PATH=str(base / 'dino.pth'))
            env.pop('SELEX_VERIFY_ALL_IMAGES', None)

            def run():
                return subprocess.run([sys.executable, '-c', preflight], env=env,
                                      capture_output=True, text=True)

            for setting in (None, '0', 'true'):
                if setting is not None:
                    env['SELEX_VERIFY_ALL_IMAGES'] = setting
                result = run()
                self.assertEqual(result.returncode, 0, result.stderr)
            env['SELEX_VERIFY_ALL_IMAGES'] = '1'
            result = run()
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing dataset image:', result.stderr)
            self.assertIn('missing.jpg', result.stderr)
            (cub / 'images/missing.jpg').touch()
            self.assertEqual(run().returncode, 0)
