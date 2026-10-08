"""Input-only raw-MRI prefixes and separate longitudinal supervision."""
from __future__ import annotations

from dataclasses import dataclass, fields
import torch


class TensorRecord:
    def to(self, device, *, non_blocking=False):
        return type(self)(**{field.name: getattr(self, field.name).to(device, non_blocking=non_blocking)
                            for field in fields(self)})

    def pin_memory(self):
        values = {field.name: getattr(self, field.name) for field in fields(self)}
        return type(self)(**{name: value if value.is_pinned() else value.pin_memory()
                            for name, value in values.items()})


@dataclass(frozen=True)
class RawMRIInput(TensorRecord):
    images: torch.Tensor                 # [B,4,3,D,H,W]; hidden visits are zero
    observed_mask: torch.Tensor          # [B,4]
    landmark: torch.Tensor               # [B], canonical T0..T3
    clinical: torch.Tensor               # [B,C], unavailable values are zero
    clinical_mask: torch.Tensor          # [B,C]
    arm_id: torch.Tensor                 # [B], 0 unknown, 1..13 official arms
    arm_mask: torch.Tensor               # [B]
    arm_known_at: torch.Tensor           # [B], -1 unknown or T0..T3
    query_mask: torch.Tensor             # [B,4], requested future, NOT target availability

    def validate(self):
        if self.images.ndim != 6 or self.images.shape[1:3] != (4, 3):
            raise ValueError("Raw MRI must have shape [B,4,3,D,H,W]")
        if any(size < 1 for size in self.images.shape):
            raise ValueError("Raw MRI batch and spatial dimensions must be nonempty")
        batch = len(self.images)
        for name in ("observed_mask", "query_mask"):
            value = getattr(self, name)
            if value.shape != (batch, 4) or value.dtype != torch.bool:
                raise ValueError(f"{name} must be boolean [B,4]")
        for name in ("landmark", "arm_id", "arm_known_at"):
            value = getattr(self, name)
            if value.shape != (batch,) or value.dtype != torch.long:
                raise ValueError(f"{name} must be int64 [B]")
        if self.arm_mask.shape != (batch,) or self.arm_mask.dtype != torch.bool:
            raise ValueError("arm_mask must be boolean [B]")
        if self.clinical.ndim != 2 or self.clinical.shape[0] != batch:
            raise ValueError("clinical must have shape [B,C]")
        if self.clinical_mask.shape != self.clinical.shape or self.clinical_mask.dtype != torch.bool:
            raise ValueError("clinical_mask must match clinical")
        if ((self.landmark < 0) | (self.landmark > 3)).any():
            raise ValueError("landmark must be a canonical stage 0..3")
        stages = torch.arange(4, device=self.landmark.device)[None]
        if (self.observed_mask & (stages > self.landmark[:, None])).any():
            raise ValueError("Observed inputs cannot include future visits")
        if not self.observed_mask.any(1).all():
            raise ValueError("Each patient needs at least one observed MRI")
        if (self.query_mask & (stages <= self.landmark[:, None])).any():
            raise ValueError("Queries must be strictly after the landmark")
        if self.images[~self.observed_mask].count_nonzero():
            raise ValueError("Unobserved MRI slots must be zero, not future ground truth")
        if self.clinical[~self.clinical_mask].count_nonzero():
            raise ValueError("Unavailable clinical values must be zero")
        if not torch.isfinite(self.images).all() or not torch.isfinite(self.clinical).all():
            raise ValueError("Inputs must contain finite values")
        if ((self.arm_id < 0) | (self.arm_id > 13)).any():
            raise ValueError("Arm IDs must be 0 unknown or 1..13")
        if (self.arm_mask & ((self.arm_id == 0) | (self.arm_known_at < 0)
                            | (self.arm_known_at > self.landmark))).any():
            raise ValueError("Available Arm must have a known category and legal known-at stage")
        if ((~self.arm_mask) & (self.arm_id != 0)).any():
            raise ValueError("Unavailable Arm must use ID zero")
        if ((~self.arm_mask) & (self.arm_known_at != -1)).any():
            raise ValueError("Unavailable Arm must use known-at stage -1")
        return self


@dataclass(frozen=True)
class RawMRISupervision(TensorRecord):
    future: torch.Tensor                 # [B,4,3,D,H,W]
    future_mask: torch.Tensor            # [B,4], only real future scans
    label: torch.Tensor                  # [B], final binary pCR
    label_mask: torch.Tensor             # [B]

    def validate(self, inp, *, future_values=True):
        if self.future.shape != inp.images.shape:
            raise ValueError("Future MRI shape must match the canonical input grid")
        batch = len(inp.images)
        if self.future_mask.shape != (batch, 4) or self.future_mask.dtype != torch.bool:
            raise ValueError("future_mask must be boolean [B,4]")
        if (self.future_mask & ~inp.query_mask).any():
            raise ValueError("Future supervision must address requested future stages")
        if self.label.shape != (batch,) or self.label_mask.shape != (batch,):
            raise ValueError("Labels and masks must have shape [B]")
        if self.label_mask.dtype != torch.bool:
            raise ValueError("label_mask must be boolean")
        if future_values and not torch.isfinite(self.future[self.future_mask]).all():
            raise ValueError("Observed targets must be finite")
        if not torch.isfinite(self.label[self.label_mask]).all():
            raise ValueError("Observed labels must be finite")
        if ((self.label[self.label_mask] != 0) & (self.label[self.label_mask] != 1)).any():
            raise ValueError("pCR labels must be binary")
        return self
