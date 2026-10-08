"""Optional conditional 3D MRI flow-matching velocity network."""
from __future__ import annotations

import math
import torch
from torch import nn
from torch.nn import functional as F

from .encoder import _groups


class _FlowResidual(nn.Module):
    def __init__(self, channels, condition_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(channels), channels)
        self.conv1 = nn.Conv3d(channels, channels, 3, padding=1)
        self.norm2 = nn.GroupNorm(_groups(channels), channels)
        self.conv2 = nn.Conv3d(channels, channels, 3, padding=1)
        self.modulation = nn.Linear(condition_dim, channels * 2)

    def forward(self, value, condition):
        shift, scale = self.modulation(condition).chunk(2, -1)
        value1 = self.norm1(value)
        value1 = value1 * (1 + scale[:, :, None, None, None]) + shift[:, :, None, None, None]
        value1 = self.conv1(F.silu(value1))
        return value + self.conv2(F.silu(self.norm2(value1)))


class ConditionalMRIFlow(nn.Module):
    """Small 3D conditional UNet predicting FM velocity, not image/noise itself.

    x0 is Gaussian noise, x1 is the source-normalized future MRI, and
    x_tau=(1-tau)*x0+tau*x1. The target velocity is x1-x0. Conditioning uses
    the latest *observed* MRI and a predicted spatial grid, never target MRI.
    Biological visit stage and flow tau have separate embeddings.
    """

    def __init__(self, cfg):
        super().__init__()
        self.grid = tuple(cfg.token_grid)
        channels = cfg.fm_channels
        self.spatial_condition = nn.Conv3d(cfg.dim, channels, 1)
        self.time = nn.Sequential(nn.Linear(4, cfg.dim), nn.SiLU(), nn.Linear(cfg.dim, cfg.dim))
        self.stage = nn.Embedding(4, cfg.dim)
        self.stem = nn.Conv3d(6 + channels, channels, 3, padding=1)
        self.high = _FlowResidual(channels, cfg.dim)
        self.down = nn.Conv3d(channels, 2 * channels, 3, stride=2, padding=1)
        self.low = _FlowResidual(2 * channels, cfg.dim)
        self.up = nn.Conv3d(3 * channels, channels, 3, padding=1)
        self.finish = _FlowResidual(channels, cfg.dim)
        self.output = nn.Conv3d(channels, 3, 3, padding=1)

    def forward(self, x_tau, tau, source_mri, future_tokens, stage):
        spatial = future_tokens.transpose(1, 2).reshape(len(x_tau), -1, *self.grid)
        spatial = F.interpolate(self.spatial_condition(spatial), size=x_tau.shape[-3:],
                                mode="trilinear", align_corners=False)
        tau_features = torch.stack((tau, tau.square(), torch.sin(math.pi * tau),
                                    torch.cos(math.pi * tau)), -1)
        condition = self.time(tau_features) + self.stage(stage) + future_tokens.mean(1)
        high = self.high(self.stem(torch.cat((x_tau, source_mri, spatial), 1)), condition)
        low = self.low(self.down(high), condition)
        low = F.interpolate(low, size=high.shape[-3:], mode="trilinear", align_corners=False)
        value = self.finish(self.up(torch.cat((high, low), 1)), condition)
        return self.output(value)
