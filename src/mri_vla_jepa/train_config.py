"""Strict configuration for fixed-T0 and dynamic MRI VLA-JEPA training."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
import math
import yaml

from .io import stable_hash
from .model_config import RawVLAJEPAConfig


@dataclass
class RawVLATrainConfig:
    schema: str = "responsewm_raw_vla_jepa_train_v1"
    model: RawVLAJEPAConfig = field(default_factory=RawVLAJEPAConfig)
    image_shape: tuple = (32, 128, 128)
    schedule: str = "one_stage"
    training_unit: str = "steps"
    max_epochs: int = 200
    early_stopping_patience: int = 50
    early_stopping_min_delta: float = 0.0
    representation_steps: int = 0
    joint_steps: int = 5000
    freeze_encoder_after_representation: bool = False
    landmarks: str = "all_observed"  # patient-uniform; or fixed T0
    task_weight: float = 1.0
    jepa_weight: float = 0.5
    flow_weight: float = 0.0
    reconstruction_weight: float = 0.1
    variance_weight: float = 0.01
    variance_definition: str = "legacy"
    diagnostics_every_epochs: int = 0
    diagnostics_patients: int = 32
    diagnostics_batch_size: int = 16
    diagnostics_seed: int = 20261004
    batch_size: int = 1
    accumulation: int = 4
    lr: float = 1e-4
    lr_scheduler: str = "none"
    plateau_factor: float = 0.5
    plateau_patience: int = 5
    plateau_min_lr: float = 1e-6
    plateau_threshold: float = 0.0
    image_cache: str | None = None
    prefetch_batches: int = 0
    loader_workers: int = 2
    pin_memory: bool = False
    non_blocking_transfer: bool = False
    weight_decay: float = 0.01
    grad_clip: float = 1.0
    validation_every: int = 100
    checkpoint_every: int = 100
    validation_batch_size: int = 1
    device: str = "cuda"
    precision: str = "bf16"
    seed: int = 20261003
    threads: int = 4
    deterministic: bool = False
    allow_synthetic: bool = False

    def validate(self):
        if self.schema != "responsewm_raw_vla_jepa_train_v1":
            raise ValueError("Unsupported raw VLA-JEPA training schema")
        if not isinstance(self.model, RawVLAJEPAConfig):
            raise ValueError("VLA training requires RawVLAJEPAConfig")
        self.model.validate()
        if self.schedule not in {"one_stage", "two_stage"}:
            raise ValueError("schedule must be one_stage or two_stage")
        if self.training_unit not in {"steps", "epochs"}:
            raise ValueError("training_unit must be steps or epochs")
        if self.training_unit == "epochs" and (self.schedule != "one_stage" or self.accumulation != 1):
            raise ValueError("Epoch training requires one_stage and accumulation=1")
        if self.landmarks not in {"all_observed", "t0"}:
            raise ValueError("landmarks must be all_observed or t0")
        if self.variance_definition not in {"legacy", "off", "patient_axis_stage_token_fp32"}:
            raise ValueError("Unsupported variance_definition")
        if self.variance_definition == "off" and self.variance_weight != 0:
            raise ValueError("variance_definition=off requires variance_weight=0")
        if type(self.diagnostics_every_epochs) is not int or self.diagnostics_every_epochs < 0:
            raise ValueError("diagnostics_every_epochs must be a nonnegative integer")
        if self.diagnostics_every_epochs and self.training_unit != "epochs":
            raise ValueError("Epoch diagnostics require epoch training")
        for name in ("diagnostics_patients", "diagnostics_batch_size"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.diagnostics_seed) is not int or self.diagnostics_seed < 0:
            raise ValueError("diagnostics_seed must be a nonnegative integer")
        if self.lr_scheduler not in {"none", "plateau"}:
            raise ValueError("lr_scheduler must be none or plateau")
        if self.lr_scheduler == "plateau" and self.training_unit != "epochs":
            raise ValueError("plateau lr_scheduler requires epoch training")
        if type(self.plateau_patience) is not int or self.plateau_patience < 0:
            raise ValueError("plateau_patience must be a nonnegative integer")
        if self.image_cache is not None and (not isinstance(self.image_cache, str) or not self.image_cache.strip()):
            raise ValueError("image_cache must be a nonempty directory string or null")
        if type(self.prefetch_batches) is not int or not 0 <= self.prefetch_batches <= 4:
            raise ValueError("prefetch_batches must be an integer from zero to four")
        if type(self.loader_workers) is not int or not 1 <= self.loader_workers <= 8:
            raise ValueError("loader_workers must be an integer from one to eight")
        if self.prefetch_batches and self.training_unit != "epochs":
            raise ValueError("Batch prefetch requires epoch training")
        if self.pin_memory and not self.device.startswith("cuda"):
            raise ValueError("pin_memory requires a CUDA training device")
        if self.non_blocking_transfer and not self.pin_memory:
            raise ValueError("non_blocking_transfer requires pin_memory")
        for name in ("joint_steps", "batch_size", "accumulation", "validation_every", "checkpoint_every", "validation_batch_size", "threads", "max_epochs", "early_stopping_patience"):
            if type(getattr(self, name)) is not int or getattr(self, name) < 1:
                raise ValueError(f"{name} must be a positive integer")
        if type(self.representation_steps) is not int or self.representation_steps < 0:
            raise ValueError("representation_steps must be nonnegative")
        if self.schedule == "two_stage" and self.representation_steps < 1:
            raise ValueError("two_stage requires representation steps")
        if len(self.image_shape) != 3 or any(type(v) is not int or v < 4 for v in self.image_shape):
            raise ValueError("image_shape must be D,H,W integers >=4")
        for name in ("task_weight", "jepa_weight", "flow_weight", "reconstruction_weight", "variance_weight", "lr", "weight_decay", "grad_clip", "early_stopping_min_delta", "plateau_factor", "plateau_min_lr", "plateau_threshold"):
            val = getattr(self, name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or val < 0:
                raise ValueError(f"Invalid {name}")
        if self.task_weight <= 0 or self.lr <= 0 or self.grad_clip <= 0:
            raise ValueError("Task weight, lr and grad_clip must be positive")
        if not 0 < self.plateau_factor < 1:
            raise ValueError("plateau_factor must be strictly between zero and one")
        if self.plateau_min_lr <= 0 or (self.lr_scheduler == "plateau" and self.plateau_min_lr > self.lr):
            raise ValueError("plateau_min_lr must be positive and no greater than the initial plateau lr")
        if self.schedule == "two_stage" and not any((self.jepa_weight, self.reconstruction_weight,
                                                     self.variance_weight)):
            raise ValueError("Representation-only training requires an active auxiliary loss")
        if self.flow_weight > 0 and not self.model.enable_flow:
            raise ValueError("Positive flow loss requires enable_flow=true")
        if self.precision not in {"fp32", "bf16"}:
            raise ValueError("precision must be fp32 or CUDA bf16")
        if self.precision == "bf16" and not self.device.startswith("cuda"):
            raise ValueError("bf16 configuration currently requires CUDA")
        for name in ("freeze_encoder_after_representation", "deterministic", "allow_synthetic",
                     "pin_memory", "non_blocking_transfer"):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if self.device.startswith("mps") and self.deterministic:
            raise ValueError("MPS indexed backward lacks strict determinism; set deterministic=false")
        return self

    def to_dict(self):
        return asdict(self)

    @property
    def digest(self):
        return stable_hash(self.to_dict())

    @property
    def total_steps(self):
        return self.joint_steps + (self.representation_steps if self.schedule == "two_stage" else 0)

    def phase(self, completed):
        return "representation" if self.schedule == "two_stage" and completed < self.representation_steps else "joint"


def from_dict(value):
    if not isinstance(value, dict):
        raise ValueError("VLA training configuration must be a mapping")
    data = dict(value)
    unknown = set(data) - {f.name for f in fields(RawVLATrainConfig)}
    if unknown:
        raise ValueError(f"Unknown VLA training keys: {sorted(unknown)}")
    if "model" in data:
        if not isinstance(data["model"], dict):
            raise ValueError("VLA model configuration must be a mapping")
        model = dict(data["model"])
        unknown = set(model) - {f.name for f in fields(RawVLAJEPAConfig)}
        if unknown:
            raise ValueError(f"Unknown VLA model keys: {sorted(unknown)}")
        if "token_grid" in model:
            model["token_grid"] = tuple(model["token_grid"])
        data["model"] = RawVLAJEPAConfig(**model)
    if "image_shape" in data:
        data["image_shape"] = tuple(data["image_shape"])
    return RawVLATrainConfig(**data).validate()


def load_config(path):
    return from_dict(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
