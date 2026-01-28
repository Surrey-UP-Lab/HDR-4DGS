import math
import torch
import torch.nn as nn

from torch import Tensor
from .embeddings import SineCosineEncoder
from typing import Tuple


class Classic3DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super(Classic3DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

    def forward(self, hdr: Tensor, exposure_time: Tensor) -> Tensor:
        hdr = hdr + exposure_time.log()
        assert hdr.shape[-1] == 3, "hdr should have 3 channels"
        ldr = []

        for c in range(3):
            ldr_c = self.crf[c](hdr[..., c:c+1])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1) # Nx3
        return ldr


class Spatial3DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, feat_dim: int = 4):
        super(Spatial3DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(1 + feat_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

    def forward(self, hdr: Tensor, feat: Tensor, exposure_time: Tensor) -> Tensor:
        hdr = hdr + exposure_time.log()

        ldr = []
        for c in range(3):
            ldr_c = self.crf[c](torch.cat([hdr[..., c:c+1], feat], dim=-1))
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1)
        return ldr


class Temporal3DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, time_dim: int = 2):
        super(Temporal3DToneMapper, self).__init__()

        self.crf = nn.ModuleList([

           nn.Sequential(
               nn.Linear(1 + time_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
           ) for _ in range(3)
        ])
        self.time_embedding = SineCosineEncoder(time_dim // 2, log_sampling=True)

    def forward(self, hdr: Tensor, time: Tensor, exposure_time: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=hdr.device, dtype=torch.float32)
        time_embedded = self.time_embedding(time)
        time_embedded = time_embedded.repeat(hdr.shape[0], 1)

        hdr = hdr + exposure_time.log()

        ldr = []
        for c in range(3):
            ldr_c = self.crf[c](torch.cat([hdr[..., c:c+1], time_embedded], dim=-1))
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1)
        return ldr


class SpatialTemporal3DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, feat_dim: int = 4, time_dim: int = 2):
        super(SpatialTemporal3DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(1 + feat_dim + time_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

        self.time_embedding = SineCosineEncoder(time_dim // 2, log_sampling=True)

    def forward(self, hdr: Tensor, feat: Tensor, time: Tensor, exposure_time: Tensor) -> Tensor:
        hdr = hdr + exposure_time.log()
        time = torch.as_tensor(time, device=hdr.device, dtype=torch.float32)

        ldr = []
        time_embedded = self.time_embedding(time)
        time_embedded = time_embedded.repeat(hdr.shape[0], 1)

        for c in range(3):
            ldr_c = self.crf[c](torch.cat([hdr[..., c:c+1], feat, time_embedded], dim=-1))
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1)
        return ldr


class Classic2DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super(Classic2DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

    def forward(self, hdr_image: Tensor, time: Tensor, exposure_time: Tensor) -> Tensor:
        hdr = hdr_image + exposure_time.log()  # [3, 400, 400]
        hdr = hdr.permute(1, 2, 0)             # [400, 400, 3]
        assert hdr.shape[-1] == 3, "hdr should have 3 channels"
        ldr = []

        for c in range(3):
            ldr_c = self.crf[c](hdr[..., c:c+1])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1) # [400, 400, 3]
        ldr = ldr.permute(2, 0, 1)     # [3, 400, 400]
        return ldr


class Spatial2DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super(Spatial2DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Conv2d(1, hidden_dim, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

    def forward(self, hdr_image: Tensor, time: Tensor, exposure_time: Tensor) -> Tensor:
        hdr = hdr_image + exposure_time.log()  # [3, 400, 400]
        hdr = hdr[None]  # [1, 3, 400, 400]
        assert hdr.shape[1] == 3, "hdr should have 3 channels"
        ldr = []

        for c in range(3):
            ldr_c = self.crf[c](hdr[:, c:c+1])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=1)[0] # [1, 3, 400, 400]
        return ldr


class Temporal2DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, time_dim: int = 2):
        super(Temporal2DToneMapper, self).__init__()

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Conv2d(1 + time_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

        self.time_embedding = SineCosineEncoder(time_dim // 2, log_sampling=True)

    def forward(self, hdr_image: Tensor, time: Tensor, exposure_time: Tensor) -> Tensor:
        time = torch.as_tensor(time, device=hdr_image.device, dtype=torch.float32)
        time_embedded = self.time_embedding(time)

        hdr = hdr_image + exposure_time.log()  # [3, 400, 400]
        ldr = []

        for c in range(3):
            ldr_c = self.crf[c](torch.cat([hdr[c:c+1], time_embedded[None].permute(2, 0, 1).repeat(1, hdr.shape[1], hdr.shape[2])], dim=0)[None])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=1)[0] # [3, 400, 400]
        return ldr


class ScaledTemporal2DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, time_dim: int = 2, exp_times = (0.125, 2.0, 32.0)):
        super(ScaledTemporal2DToneMapper, self).__init__()

        assert len(exp_times) >= 1, "At least one exposure time should be provided"
        if len(exp_times) == 1:
            self.r = 1
            self.s = 0
        else:
            log_times = [math.log(t) for t in exp_times]
            self.r = 1.0 / (max(log_times) - min(log_times))  # scale factor
            self.s = -(max(log_times) + min(log_times)) / 2  # offset

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Conv2d(1 + time_dim, hidden_dim, kernel_size=3, padding=1, stride=1),
                    nn.ReLU(),
                    nn.Conv2d(hidden_dim, 1, kernel_size=3, padding=1, stride=1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

        self.time_embedding = SineCosineEncoder(time_dim // 2, log_sampling=True)

    def forward(self, hdr_image: Tensor, time: Tensor, exposure_time: Tensor) -> Tuple[Tensor, Tensor]:
        time = torch.as_tensor(time, device=hdr_image.device, dtype=torch.float32)
        time_embedded = self.time_embedding(time)

        hdr = hdr_image + self. r * exposure_time.log() + self.s  # [3, 400, 400]
        ldr = []

        for c in range(3):
            ldr_c = self.crf[c](torch.cat([hdr[c:c+1], time_embedded[None].permute(2, 0, 1).repeat(1, hdr.shape[1], hdr.shape[2])], dim=0)[None])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=1)[0] # [3, 400, 400]
        hdr = torch.exp((hdr_image + self.s) / self.r)
        return hdr, ldr


class Scaled3DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64, time_dim: int = 2):
        super(Scaled3DToneMapper, self).__init__()

        self.scale_predictor = nn.Sequential(
            nn.Linear(1, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(3, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

        self.time_embedding = SineCosineEncoder(time_dim // 2, log_sampling=True)

    def forward(self, hdr: Tensor, time: Tensor, exposure_time: Tensor) -> Tuple[Tensor, Tensor]:
        luminance = hdr + exposure_time.log()
        mean_luminance = torch.mean(luminance, dim=1, keepdim=True)
        r, s = self.scale_predictor(mean_luminance).split([1, 1], dim=1)
        scale_r, scale_s = torch.sigmoid(r), torch.tanh(s)
        scale_luminance = scale_r * luminance + scale_s

        time_embedded = self.time_embedding(time)
        time_embedded = time_embedded.repeat(hdr.shape[0], 1)

        ldr = []
        for c in range(3):
            ldr_c = self.crf[c](
                torch.cat([scale_luminance[..., c:c+1], time_embedded], dim=-1)
            )
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1) # Nx3
        hdr = torch.exp(scale_r * hdr + scale_s)

        return hdr, ldr


class Scaled2DToneMapper(nn.Module):
    def __init__(self, hidden_dim: int = 64):
        super(Scaled2DToneMapper, self).__init__()

        self.scale_predictor = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

        self.crf = nn.ModuleList([
               nn.Sequential(
                   nn.Linear(1, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(hidden_dim, 1),
                    nn.Sigmoid()
               )
            for _ in range(3)
        ])

    def forward(self, hdr: Tensor, time: Tensor, exposure_time: Tensor) -> Tuple[Tensor, Tensor]:
        luminance = hdr + exposure_time.log()
        mean_luminance = torch.mean(luminance, dim=[1, 2], keepdim=False)[None] # [1, 3]
        r, s = self.scale_predictor(mean_luminance).split([1, 1], dim=1)
        scale_r, scale_s = torch.sigmoid(r), torch.tanh(s)
        scale_luminance = scale_r * luminance + scale_s  # [3, 400, 400]
        scale_luminance = scale_luminance.permute(1, 2, 0)  # [400, 400, 3]

        ldr = []
        for c in range(3):
            ldr_c = self.crf[c](scale_luminance[..., c:c+1])
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1) # Nx3
        ldr = ldr.permute(2, 0, 1)     # [3, 400, 400]
        hdr = torch.exp(scale_r * hdr + scale_s)

        return hdr, ldr


class GaussianGRUToneMapper(nn.Module):
    def __init__(self, input_dim=3, gru_dim=2, hidden_dim=64, gru_layers=1):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=gru_dim,
            num_layers=gru_layers,
            batch_first=True
        )

        self.mlp = nn.ModuleList([
            nn.Sequential(
                nn.Linear(1 + gru_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 1),
                nn.Sigmoid()
            ) for _ in range(3)
        ])

    def forward(self, hdr: Tensor, x_seq: Tensor, time: Tensor, exposure_time: Tensor, iteration: int = 0):
        """
        x_seq: [batch_size, seq_len, N, input_dim]
        """
        if iteration >= -1:
            seq = x_seq.reshape(1, -1, 3).clone()
            _, h_n = self.gru(seq)
            
            if hdr.ndim == 2:
                gru_output = h_n[-1].repeat(hdr.shape[0], 1)  # 取最后一层隐藏状态
            else:
                assert hdr.shape[2] == 3, f"Expect RGB channel last, but got shape {hdr.shape}"
                gru_output = h_n[-1][None].repeat(hdr.shape[0], hdr.shape[1], 1)
        else:
            if hdr.ndim == 2:
                gru_output = torch.zeros(hdr.shape[0], 2, device=hdr.device)
            else:
                assert hdr.shape[2] == 3, f"Expect RGB channel last, but got shape {hdr.shape}"
                gru_output = torch.zeros(hdr.shape[0], hdr.shape[1], 2, device=hdr.device)
                
        assert exposure_time > 0, "exposure_time should be positive"
        lumin = hdr + exposure_time.log()

        ldr = []
        for c in range(3):
            ldr_c = self.mlp[c](
                torch.cat([lumin[..., c:c+1], gru_output], dim=-1)
            )
            ldr.append(ldr_c)

        ldr = torch.cat(ldr, dim=-1)  # [B*N, 3]
        return ldr
