"""Validated model hyperparameters; voxel-flow generation is optional."""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RawJEPAConfig:
    clinical_dim: int = 4
    dim: int = 128
    token_grid: tuple[int, int, int] = (2, 4, 4)
    encoder_depth: int = 2
    fusion_depth: int = 2
    predictor_depth: int = 2
    heads: int = 4
    state_queries: int = 8
    pcr_queries: int = 4
    fm_channels: int = 24
    ema_decay: float = .99
    dropout: float = 0.
    enable_flow: bool = True

    def validate(self):
        for name in ("clinical_dim", "dim", "encoder_depth", "fusion_depth",
                     "predictor_depth", "heads", "state_queries", "pcr_queries",
                     "fm_channels"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if self.dim % self.heads:
            raise ValueError("dim must be divisible by heads")
        if (len(self.token_grid) != 3 or
                any(isinstance(v, bool) or not isinstance(v, int) or v < 1
                    for v in self.token_grid)):
            raise ValueError("token_grid must have three positive integer dimensions")
        if not math.isfinite(self.ema_decay) or not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must lie in [0,1)")
        if not math.isfinite(self.dropout) or not 0 <= self.dropout < 1:
            raise ValueError("dropout must lie in [0,1)")
        if not isinstance(self.enable_flow, bool):
            raise ValueError("enable_flow must be boolean")
        return self


@dataclass
class RawVLAJEPAConfig(RawJEPAConfig):
    enable_flow: bool = False
