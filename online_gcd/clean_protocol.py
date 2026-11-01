"""Fixed clean-DINO protocol. Asset locations are CLI/environment inputs only."""
from copy import deepcopy
from dataclasses import dataclass

from online_gcd.artifacts import digest
from online_gcd.prototype_bank import integer

PROTOCOL = 'online-selex-cub-frozen-dino-v0'
# Existing official asset identity in BASELINE_SETUP.md and download_assets.py.
DINO_SHA256 = 'bf34ad0f424b9029b593e8dc3ed553bf26e88bcba0d32bf3e62a6209cb64c85e'
PREPROCESSING = {'image_mode': 'RGB', 'resize_shorter_side': 256, 'center_crop': 224,
                 'interpolation': 'PIL_bicubic', 'tensor_scale': 'uint8_to_float32_0_1',
                 'mean': [0.485, 0.456, 0.406], 'std': [0.229, 0.224, 0.225]}
FIXED = {'protocol_version': PROTOCOL, 'dataset': 'CUB-200-2011', 'class_split': 'existing_SSB_100_100',
         'labelled_subset': 'global_known_train_50pct_numpy_randomstate0', 'num_online_sessions': 5,
         'backbone': 'DINOv1_ViT_B_16', 'checkpoint_role': 'official_imagenet_pretrained_backbone',
         'checkpoint_sha256': DINO_SHA256, 'feature_layer': 'final_layernorm_CLS', 'feature_dim': 768,
         'preprocessing': PREPROCESSING, 'feature_normalization': 'L2_float32',
         'threshold_calibration': 'known_only_loo_balanced_accuracy_v0', 'ema_momentum': 0.9,
         'replay_enabled': False, 'encoder_update': 'frozen', 'memory_budget': 'zero_replay_unbounded_prototypes_logged',
         'cluster_merge': False, 'cluster_split': False, 'cluster_delete': False}


@dataclass(frozen=True)
class CleanConfig:
    seed: int = 1

    def __post_init__(self):
        integer(self.seed, 'seed')

    def snapshot(self):
        return dict(deepcopy(FIXED), seed=self.seed)

    @classmethod
    def from_dict(cls, values):
        if not isinstance(values, dict) or 'seed' not in values:
            raise ValueError('clean protocol configuration requires seed and all fixed fields')
        result = cls(values['seed'])
        if digest(values) != digest(result.snapshot()):
            raise ValueError('clean protocol settings differ from the fixed v0 contract (no SelEx initialization)')
        return result
