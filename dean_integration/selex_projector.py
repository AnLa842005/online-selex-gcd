"""Three-layer DINOHead, matching the audited SelEx architecture and initialization."""
import torch
from torch import nn
from torch.nn import functional as F


class SelExProjector(nn.Module):
    def __init__(self, config, input_dim=768):
        super().__init__()
        self.epsilon = config.epsilon
        self.mlp = nn.Sequential(nn.Linear(input_dim, config.hidden_dim), nn.GELU(),
                                 nn.Linear(config.hidden_dim, config.hidden_dim), nn.GELU(),
                                 nn.Linear(config.hidden_dim, config.bottleneck_dim))
        self.mlp.apply(self._init_weights)
        # Reference initializes this Linear after MLP trunc_normal initialization.
        self.last_layer = nn.utils.weight_norm(nn.Linear(config.bottleneck_dim, config.output_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        self.last_layer.weight_g.requires_grad_(False)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            nn.init.trunc_normal_(module.weight, std=0.02)
            nn.init.zeros_(module.bias)

    def forward(self, features):
        bottleneck = self._normalize(self.mlp(features))
        return self._normalize(self.last_layer(bottleneck))

    def _normalize(self, features):
        with torch.autocast(device_type=features.device.type, enabled=False):
            if features.dtype in (torch.float16, torch.bfloat16):
                features = features.float()
            return F.normalize(features, dim=-1, eps=self.epsilon)
