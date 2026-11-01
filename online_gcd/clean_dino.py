"""Explicit frozen-DINO extraction entrypoints; torch imported only when requested."""
from datetime import datetime, timezone
from pathlib import Path
import os
import platform

import numpy as np

from online_gcd.artifacts import code_commit, digest, file_sha256
from online_gcd.clean_protocol import DINO_SHA256
from online_gcd.cub_manifest import build_cub_bundle
from online_gcd.feature_cache import load_cache, write_cache


def verify_checkpoint(path):
    if file_sha256(path) != DINO_SHA256:
        raise ValueError('only the pinned official DINOv1 pretrained backbone is allowed; SelEx/fine-tuned weights rejected')


def load_clean_encoder(path):
    # Verify before deserialization or model construction, even for renamed files.
    verify_checkpoint(path)
    import torch
    from models.vision_transformer import vit_base
    state = torch.load(path, map_location='cpu', weights_only=True)
    model = vit_base(patch_size=16)
    model.load_state_dict(state, strict=True)
    model.requires_grad_(False)
    model.eval()
    return model


def evaluation_transform():
    from types import SimpleNamespace
    from data.augmentations import get_transform
    # Same PIL resize, center crop and ImageNet normalization as offline extraction.
    return get_transform('imagenet', image_size=224,
                         args=SimpleNamespace(interpolation=3, crop_pct=0.875))[1]


class ManifestImages:
    """Private extraction dataset: no true labels returned, no download hooks."""
    def __init__(self, cub_root, bundle):
        self.root = (Path(cub_root) / 'images').resolve()
        p = bundle['partition']
        by_id = {'cub-' + digest(['CUB-200-2011', r['image_id']]): r['path'] for r in p['source_records']}
        self.paths = [self.root / by_id[sid] for sid in p['feature_sample_ids']]
        for path in self.paths:
            if not path.resolve().is_relative_to(self.root) or not path.is_file():
                raise ValueError(f'image missing or outside CUB root: {path}')
        self.transform = evaluation_transform()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        from PIL import Image
        with Image.open(self.paths[index]) as image:
            return self.transform(image.convert('RGB'))


def normalized_batches(model, loader, device, total):
    import torch
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise ValueError('encoder must be frozen and in eval mode')
    processed = 0
    with torch.inference_mode():
        for images in loader:
            if model.training:
                raise ValueError('encoder left eval mode')
            output = model(images.to(device))  # final layer-normalized CLS, no head
            if output.ndim != 2 or output.shape[1] != 768 or not torch.isfinite(output).all():
                raise ValueError('encoder output must be finite [batch, 768] CLS features')
            output = output.float()
            norms = torch.linalg.vector_norm(output, dim=1, keepdim=True)
            if not torch.isfinite(norms).all() or torch.any(norms == 0):
                raise ValueError('invalid CLS norm')
            features = (output / norms).cpu().numpy()
            processed += len(features)
            print(f'Extracted {processed}/{total} images', flush=True)
            yield features


def extract(cub_root, checkpoint, bundle, config, output, *, device='cpu', batch_size=64, workers=0, reuse=False):
    from online_gcd.prototype_bank import integer
    integer(batch_size, 'batch_size', 1)
    integer(workers, 'workers')
    if not (device == 'cpu' or device.startswith('cuda:') and device[5:].isdigit()):
        raise ValueError('device must be cpu or cuda:<index>')
    verify_checkpoint(checkpoint)
    if build_cub_bundle(cub_root, config) != bundle:
        raise ValueError('live metadata differs from the input manifest')
    if reuse:
        features, metadata = load_cache(output, bundle)
        del features
        print('Validated existing cache; no images or model loaded.')
        return metadata
    if Path(output).exists():
        raise ValueError('cache output exists; refusing to overwrite')
    # Set before CUDA initialization. This command owns its extraction process.
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    import torch
    import torchvision
    import PIL
    torch.manual_seed(config.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if device.startswith('cuda') and not torch.cuda.is_available():
        raise ValueError('requested CUDA is unavailable; no automatic fallback')
    dataset = ManifestImages(cub_root, bundle)
    model = load_clean_encoder(checkpoint).to(device)
    generator = torch.Generator().manual_seed(config.seed)
    loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, num_workers=workers,
                                         shuffle=False, drop_last=False, generator=generator)
    provenance = {'kind': 'frozen_dino', 'checkpoint_path': str(Path(checkpoint).resolve()),
                  'repository_commit': code_commit(), 'timestamp_utc': datetime.now(timezone.utc).isoformat(),
                  'device': device, 'batch_size': batch_size, 'workers': workers,
                  'seed': config.seed, 'deterministic_algorithms': True, 'tf32': False,
                  'versions': {'python': platform.python_version(), 'torch': torch.__version__,
                               'torchvision': torchvision.__version__, 'pillow': PIL.__version__, 'numpy': np.__version__}}
    return write_cache(output, bundle, normalized_batches(model, loader, device, len(dataset)), provenance)
