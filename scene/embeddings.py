import torch
import torch.nn as nn

from torch import Tensor


class SineCosineEncoder(nn.Module):
    def __init__(self, num_frequencies=16, max_freq=10.0, log_sampling=True):
        super().__init__()
        self.num_frequencies = num_frequencies

        if log_sampling:
            freq_bands = 2.0 ** torch.linspace(1.0, max_freq, num_frequencies)
        else:
            freq_bands = torch.linspace(1.0, max_freq, num_frequencies)

        self.register_buffer('freq_bands', freq_bands)

    def forward(self, x: Tensor) -> Tensor:
        sin_features = torch.sin(x * self.freq_bands[None])
        cos_features = torch.cos(x * self.freq_bands[None])
        return torch.cat([sin_features, cos_features], dim=-1)