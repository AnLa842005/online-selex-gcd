import ast
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from online_cub_fixtures import fixture_bundle, metadata_fixture
from online_gcd.artifacts import seal
from online_gcd.clean_protocol import CleanConfig
from online_gcd.cub_manifest import (build_bundle, build_cub_bundle, labelled_known_indices,
                                      learner_session, load_bundle, read_metadata, save_bundle, validate_bundle)
from online_gcd.ssb import load_cub_ssb_classes

ROOT = Path(__file__).resolve().parents[1]


class CUBManifestTests(unittest.TestCase):
    def test_original_global_half_selection_and_rng_parity(self):
        source = ast.parse((ROOT / 'data/data_utils.py').read_text())
        function = next(n for n in source.body if getattr(n, 'name', None) == 'subsample_instances')
        namespace = {'np': np}
        exec(compile(ast.Module(body=[function], type_ignores=[]), '<original sampling>', 'exec'), namespace)
        state = np.random.get_state()
        try:
            for count in (5, 23, 100):
                expected = namespace['subsample_instances'](range(count), .5)
                np.testing.assert_array_equal(labelled_known_indices(count), expected)
        finally:
            np.random.set_state(state)
        labelled_known_indices(17)
        after = np.random.get_state()
        np.testing.assert_array_equal(state[1], after[1])
        self.assertEqual(state[2:], after[2:])

    def test_real_ssb_reader_reused_on_fake_metadata(self):
        with tempfile.TemporaryDirectory() as folder:
            root = metadata_fixture(folder, classes=200, train_per_class=40, test_per_class=1)
            with patch('online_gcd.cub_manifest.load_cub_ssb_classes', wraps=load_cub_ssb_classes) as read:
                bundle = build_cub_bundle(root, CleanConfig())
                read.assert_called_once()
            known, novel = load_cub_ssb_classes()
            self.assertEqual(bundle['partition']['known_classes'], list(known))
            self.assertEqual(bundle['partition']['novel_classes'], list(novel))
            validate_bundle(bundle, CleanConfig())

    def test_partition_coverage_exact_subset_and_safe_views(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = fixture_bundle(folder)
            p = bundle['partition']
            base, stream, probes = (bundle[k]['records'] for k in ('base', 'stream_evaluator', 'probe_evaluator'))
            sets = [{r['sample_id'] for r in records} for records in (base, stream, probes)]
            self.assertEqual(sum(map(len, sets)), len(p['source_records']))
            self.assertEqual(len(set.union(*sets)), len(p['source_records']))
            self.assertEqual(len(base), 30)
            self.assertEqual(len(stream), 180)
            self.assertEqual(len(probes), 21)
            self.assertEqual({r['known_class_id'] for r in base}, {1, 5})
            self.assertTrue(all(r['official_split'] == 'test' for r in probes))
            from online_gcd.artifacts import digest
            sources = {'cub-' + digest(['CUB-200-2011', r['image_id']]): r for r in p['source_records']}
            known_train = [r for r in p['source_records'] if r['official_split'] == 'train' and r['class_id'] in (1, 5)]
            chosen = {known_train[i]['image_id'] for i in labelled_known_indices(len(known_train))}
            self.assertEqual({sources[r['sample_id']]['image_id'] for r in base}, chosen)
            self.assertTrue(all(sources[r['sample_id']]['official_split'] == 'train' for r in base + stream))
            self.assertTrue(all(sources[r['sample_id']]['official_split'] == 'test' for r in probes))
            for r in stream:
                self.assertGreaterEqual(r['session_id'], r['introduction_session'])
            for session in range(1, 6):
                for row in learner_session(bundle, session):
                    self.assertEqual(set(row), {'sample_id', 'session_id', 'feature_row'})
                    self.assertEqual(row['session_id'], session)
            schedule = dict(p['class_introduction_schedule'])
            self.assertTrue(all(r['eligibility_session'] == schedule.get(r['class_id'], 0) for r in probes))

    def test_determinism_checksums_and_cross_view_tampering(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle = fixture_bundle(folder)
            p = bundle['partition']
            self.assertEqual(bundle, build_bundle(p['source_records'], p['metadata_hashes'],
                                                  p['known_classes'], p['novel_classes'], CleanConfig()))
            different = build_bundle(p['source_records'], p['metadata_hashes'], p['known_classes'],
                                     p['novel_classes'], CleanConfig(seed=2))
            for name in bundle:
                self.assertNotEqual(bundle[name]['checksum'], different[name]['checksum'])
            changed = deepcopy(p['source_records'])
            changed[0]['path'] = 'different.jpg'
            different = build_bundle(changed, p['metadata_hashes'], p['known_classes'], p['novel_classes'], CleanConfig())
            self.assertNotEqual(p['checksum'], different['partition']['checksum'])
            for change in ('class_id', 'feature_row', 'session_id'):
                altered = deepcopy(bundle)
                altered['stream_learner']['records'][0][change] = 999
                altered['stream_learner'] = seal({k: v for k, v in altered['stream_learner'].items() if k != 'checksum'})
                with self.assertRaises(ValueError):
                    validate_bundle(altered, CleanConfig(), official=False)
            output = Path(folder) / 'manifests'
            save_bundle(bundle, output)
            self.assertEqual(load_bundle(output, CleanConfig(), official=False), bundle)
            with self.assertRaises(ValueError):
                load_bundle(output, CleanConfig())  # Tiny fixture is not real SSB.
            with self.assertRaises(ValueError):
                save_bundle(bundle, output)

    def test_malformed_metadata_is_rejected_without_images(self):
        for name, extra in (('images.txt', '1 duplicated.jpg\n'),
                            ('image_class_labels.txt', '999 1\n'),
                            ('train_test_split.txt', '999 2\n')):
            with self.subTest(name=name), tempfile.TemporaryDirectory() as folder:
                root = metadata_fixture(folder)
                with (root / name).open('a') as handle:
                    handle.write(extra)
                with self.assertRaises(ValueError):
                    read_metadata(root)
        with tempfile.TemporaryDirectory() as folder:
            root = metadata_fixture(folder)
            path = root / 'images.txt'
            path.write_text(path.read_text().replace('001.class_name/img_0.jpg', '../outside.jpg'))
            with self.assertRaisesRegex(ValueError, 'relative'):
                read_metadata(root)


if __name__ == '__main__':
    unittest.main()
