"""Exercise the real workload guards independently of optional SciPy dependencies."""
import ast
import builtins
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).absolute().parents[1]


class ImportGuardCompatibilityTests(unittest.TestCase):
    def test_relative_models_and_absolute_workloads(self):
        for filename, definition, original_name in [
            ('test_online_cub_runner.py', 'def guard(', 'original'),
            ('test_online_experiment.py', 'def guarded_import(', 'original_import')]:
            tree = ast.parse((ROOT / 'tests' / filename).read_text(encoding='utf-8-sig'))
            source = next(n.value for n in ast.walk(tree) if isinstance(n, ast.Constant)
                          and isinstance(n.value, str) and definition in n.value)
            start = source.index(original_name + ' = builtins.__import__')
            stop = (source.index('from online_cub_fixtures') if 'from online_cub_fixtures' in source
                    else source.index('def audit('))
            original_import = builtins.__import__
            with self.subTest(guard=filename):
                try:
                    # Includes the synthetic relative-module collision and explicit
                    # forbidden absolute imports embedded in the actual CLI test.
                    exec(source[start:stop], {'builtins': builtins, 'sys': sys})
                    try:
                        import charset_normalizer
                    except ModuleNotFoundError:
                        pass  # Synthetic collision above still runs on minimal environments.
                    else:
                        builtins.__import__('models', {'__package__': 'charset_normalizer'}, None, (), 1)
                finally:
                    builtins.__import__ = original_import
                    sys.modules.pop('guard_relative_fixture.models', None)
                    sys.modules.pop('guard_relative_fixture', None)


if __name__ == '__main__':
    unittest.main()
