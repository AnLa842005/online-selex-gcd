import ast
import os
from pathlib import Path
import pickle
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from online_gcd.ssb import load_cub_ssb_classes

ROOT = Path(__file__).resolve().parents[1]


class OnlineSSBTests(unittest.TestCase):
    def test_reader_matches_existing_offline_function_without_dataset_import(self):
        # Execute only the existing split function, with a synthetic pickle.
        # No dataset classes, images, torch or real numeric SSB membership needed.
        source = ast.parse((ROOT / 'data/get_datasets.py').read_text())
        function = next(node for node in source.body
                        if isinstance(node, ast.FunctionDef) and node.name == 'get_class_splits')
        namespace = {'pickle': pickle, 'os': os, 'osr_split_dir': 'unused'}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<split parity>', 'exec'), namespace)
        fixture = {'known_classes': [9, 2],
                   'unknown_classes': {'Hard': [7, 3], 'Medium': [8], 'Easy': [1]}}
        from unittest.mock import mock_open
        content = pickle.dumps(fixture)
        import hashlib
        with patch('builtins.open', mock_open(read_data=content)):
            args = namespace['get_class_splits'](SimpleNamespace(dataset_name='cub', use_ssb_splits=True))
        with patch('pathlib.Path.read_bytes', return_value=content), patch(
                'online_gcd.ssb.CUB_SPLIT_SHA256', hashlib.sha256(content).hexdigest()):
            known, novel = load_cub_ssb_classes()
        self.assertEqual(known, tuple(args.train_classes))
        self.assertEqual(novel, tuple(args.unlabeled_classes))

    def test_changed_split_rejected_before_deserialization(self):
        with patch('pathlib.Path.read_bytes', return_value=b'wrong'), patch('pickle.loads') as load:
            with self.assertRaisesRegex(ValueError, 'checksum'):
                load_cub_ssb_classes()
            load.assert_not_called()


if __name__ == '__main__':
    unittest.main()
