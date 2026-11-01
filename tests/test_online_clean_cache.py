from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from online_cub_fixtures import fixture_cache
from online_gcd.artifacts import seal
from online_gcd.clean_dino import load_clean_encoder, normalized_batches, verify_checkpoint, evaluation_transform
from online_gcd.clean_protocol import CleanConfig
from online_gcd.feature_cache import load_cache, write_cache
from online_gcd.stream import load_config

ROOT = Path(__file__).resolve().parents[1]


class CleanCacheTests(unittest.TestCase):
    def test_cache_is_label_free_readonly_and_fixture_cannot_be_real(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _, cache, features, metadata = fixture_cache(folder)
            loaded, actual = load_cache(cache, bundle, allow_fixture=True)
            self.assertEqual(actual, metadata)
            np.testing.assert_array_equal(loaded, features)
            self.assertFalse(loaded.flags.writeable)
            self.assertEqual({p.name for p in cache.iterdir()}, {'features.npy', 'sample_ids.json', 'cache.json'})
            self.assertNotIn('class_id', (cache / 'cache.json').read_text())
            with self.assertRaisesRegex(ValueError, 'fixture'):
                load_cache(cache, bundle)
            del loaded

    def test_stale_cache_hashes_dimensions_order_and_contents(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _, cache, _, metadata = fixture_cache(folder)
            for key, value in (('manifest_checksum', 'bad'), ('checkpoint_sha256', 'bad'),
                               ('preprocessing_sha256', 'bad'), ('feature_dim', 2),
                               ('metadata_sha256', 'bad'), ('ssb_sha256', 'bad'), ('normalization', 'none')):
                changed = {k: v for k, v in metadata.items() if k != 'checksum'}
                changed[key] = value
                (cache / 'cache.json').write_text(json.dumps(seal(changed)))
                with self.subTest(key=key), self.assertRaisesRegex(ValueError, key):
                    load_cache(cache, bundle, allow_fixture=True)
            (cache / 'cache.json').write_text(json.dumps(metadata))
            ids = json.loads((cache / 'sample_ids.json').read_text())
            (cache / 'sample_ids.json').write_text(json.dumps(ids[::-1]))
            with self.assertRaisesRegex(ValueError, 'IDs/order'):
                load_cache(cache, bundle, allow_fixture=True)
            (cache / 'sample_ids.json').write_text(json.dumps(ids))
            with (cache / 'features.npy').open('ab') as handle:
                handle.write(b'corruption')
            with self.assertRaisesRegex(ValueError, 'data checksum'):
                load_cache(cache, bundle, allow_fixture=True)

    def test_partial_or_malformed_extraction_never_publishes_cache(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _, _, features, _ = fixture_cache(folder)
            for index, batches in enumerate(([features[:-1]], [features, features[:1]],
                                              [features.astype(np.float64)], [features[:, :2]],
                                              [np.zeros_like(features)])):
                target = Path(folder) / f'rejected{index}'
                with self.assertRaises(ValueError):
                    write_cache(target, bundle, batches, {'kind': 'fixture'})
                self.assertFalse(target.exists())
            self.assertFalse(list(Path(folder).glob('.features-*')))

    def test_missing_real_extraction_provenance_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            bundle, _, cache, _, metadata = fixture_cache(folder)
            altered = {k: v for k, v in metadata.items() if k != 'checksum'}
            altered['extraction'] = {'kind': 'frozen_dino'}
            (cache / 'cache.json').write_text(json.dumps(seal(altered)))
            with self.assertRaisesRegex(ValueError, 'provenance'):
                load_cache(cache, bundle)

    def test_clean_config_and_checkpoint_allowlist_cannot_select_selex(self):
        config = load_config(ROOT / 'configs/online_selex/cub_frozen_dino_v0.yaml', CleanConfig)
        for key, value in (('checkpoint_role', 'selex'), ('checkpoint_sha256', 'anything'),
                           ('ema_momentum', 0.5), ('encoder_update', 'trainable')):
            altered = config.snapshot()
            altered[key] = value
            with self.assertRaises(ValueError):
                CleanConfig.from_dict(altered)
        with tempfile.TemporaryDirectory() as folder:
            for name in ('model.pt', 'dino_vitbase16_pretrain.pth'):
                path = Path(folder) / name
                path.write_bytes(b'not the approved pretrained checkpoint')
                with patch('torch.load') as load:
                    with self.assertRaisesRegex(ValueError, 'pretrained backbone'):
                        load_clean_encoder(path)
                    load.assert_not_called()

    def test_tiny_mock_encoder_is_frozen_and_inference_only(self):
        import torch
        class Tiny(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.scale = torch.nn.Parameter(torch.ones(1))
                self.observations = []

            def forward(self, x):
                self.observations.append((self.training, torch.is_grad_enabled()))
                return x.repeat(1, 384) * self.scale
        model = Tiny()
        with patch('online_gcd.clean_dino.verify_checkpoint'), patch('torch.load', return_value=model.state_dict()) as load, \
                patch('models.vision_transformer.vit_base', return_value=model) as factory:
            encoder = load_clean_encoder('mock-only')
            factory.assert_called_once_with(patch_size=16)
            self.assertTrue(load.call_args.kwargs['weights_only'])
        before = encoder.scale.detach().clone()
        output = list(normalized_batches(encoder, [torch.ones(2, 2)], 'cpu', 2))
        self.assertEqual(encoder.observations, [(False, False)])
        self.assertFalse(encoder.scale.requires_grad)
        self.assertTrue(torch.equal(before, encoder.scale))
        self.assertEqual(output[0].dtype, np.float32)
        np.testing.assert_allclose(np.linalg.norm(output[0], axis=1), 1, atol=1e-6)
        encoder.train()
        with self.assertRaises(ValueError):
            list(normalized_batches(encoder, [torch.ones(1, 2)], 'cpu', 1))

    def test_preprocessing_matches_existing_evaluation_transform(self):
        from data.augmentations import get_transform
        from types import SimpleNamespace
        from PIL import Image
        import torch
        image = Image.fromarray(np.random.default_rng(0).integers(0, 255, (280, 300, 3), dtype=np.uint8))
        old = get_transform('imagenet', 224, SimpleNamespace(interpolation=3, crop_pct=.875))[1]
        self.assertTrue(torch.equal(evaluation_transform()(image), old(image)))


if __name__ == '__main__':
    unittest.main()
