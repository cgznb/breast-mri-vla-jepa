"""VLA-style raw-MRI JEPA with isolated teacher-forced world supervision.

The shared backbone sees only the legal observed prefix and emits dynamics
and direct pCR features. A separate world predictor trains on true previous
states, while forecasting freely rolls its own predictions forward. Neither
world targets nor world predictions are read by the pCR head.
"""
from __future__ import annotations

import copy
import math
from dataclasses import replace

import torch
from torch import nn
from torch.nn import functional as F

from .contracts import RawMRIInput, RawMRISupervision
from .model_config import RawJEPAConfig, RawVLAJEPAConfig
from .encoder import VisitMRIEncoder
from .flow import ConditionalMRIFlow





class TimeCausalWorldPredictor(nn.Module):
    """[z(0->1),S0,z(1->2),S1,z(2->3),S2] -> [predicted S1,S2,S3].

    Within one time block, all action/state positions can attend each other.
    Across blocks attention is causal. A learned root key is always available
    even if an early block is missing. Inactive inputs are zeroed *before*
    embedding, so their NaNs or contents cannot affect active outputs.
    """

    def __init__(self, cfg):
        super().__init__()
        self.dim = cfg.dim
        self.tokens = math.prod(cfg.token_grid)
        self.queries = cfg.state_queries
        self.state_embed = nn.Linear(cfg.dim, cfg.dim)
        self.action_embed = nn.Linear(cfg.dim, cfg.dim)
        self.spatial_position = nn.Parameter(torch.randn(1, self.tokens, cfg.dim) * .02)
        self.query_position = nn.Parameter(torch.randn(1, self.queries, cfg.dim) * .02)
        self.time = nn.Embedding(3, cfg.dim)
        self.root = nn.Parameter(torch.randn(1, 1, cfg.dim) * .02)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.blocks = nn.TransformerEncoder(layer, cfg.predictor_depth,
                                             enable_nested_tensor=False)
        self.output = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, cfg.dim))

    def forward(self, source_states, transitions, source_mask):
        batch = len(source_states)
        if source_states.shape != (batch, 3, self.tokens, self.dim):
            raise ValueError("World source states must have shape [B,3,N,dim]")
        if transitions.shape != (batch, 3, self.queries, self.dim):
            raise ValueError("World transitions must have shape [B,3,K,dim]")
        if source_mask.shape != (batch, 3) or source_mask.dtype != torch.bool:
            raise ValueError("World source mask must be boolean [B,3]")
        active = source_mask[:, :, None, None]
        source_states = source_states.masked_fill(~active, 0)
        transitions = transitions.masked_fill(~active, 0)
        temporal = self.time.weight[None, :, None]
        state = self.state_embed(source_states) + self.spatial_position[:, None] + temporal
        action = self.action_embed(transitions) + self.query_position[:, None] + temporal
        per_block = self.queries + self.tokens
        packed = torch.cat((action, state), 2).flatten(1, 2)
        packed = torch.cat((self.root.expand(batch, -1, -1), packed), 1)
        key_padding = torch.cat((torch.zeros(batch, 1, dtype=torch.bool, device=packed.device),
                                 ~source_mask[:, :, None].expand(-1, -1, per_block).flatten(1)), 1)
        block_time = torch.cat((torch.tensor([-1], device=packed.device),
                                torch.arange(3, device=packed.device).repeat_interleave(per_block)))
        blocked = block_time[None, :] > block_time[:, None]
        hidden = self.blocks(packed, mask=blocked, src_key_padding_mask=key_padding)
        spatial = hidden[:, 1:].reshape(batch, 3, per_block, self.dim)[:, :, self.queries:]
        return self.output(spatial).masked_fill(~active, 0)


class RawVLAJEPA(nn.Module):
    """Source-only shared backbone, direct pCR, teacher-forced auxiliary world head."""

    def __init__(self, cfg: RawVLAJEPAConfig):
        super().__init__()
        self.cfg = cfg.validate()
        self.encoder = VisitMRIEncoder(cfg)
        self.teacher_encoder = copy.deepcopy(self.encoder).requires_grad_(False)
        self.teacher_encoder.eval()
        self.stage = nn.Embedding(4, cfg.dim)
        self.clinical = nn.Linear(2 * cfg.clinical_dim, cfg.dim)
        self.arm = nn.Embedding(14, cfg.dim)
        self.arm_known_at = nn.Embedding(4, cfg.dim)
        # Slot 0 is unused; slot t conditions the transition (t-1) -> t.
        self.dynamics_queries = nn.Parameter(torch.randn(4, cfg.state_queries, cfg.dim) * .02)
        self.pcr_queries = nn.Parameter(torch.randn(cfg.pcr_queries, cfg.dim) * .02)
        layer = nn.TransformerEncoderLayer(cfg.dim, cfg.heads, cfg.dim * 4, cfg.dropout,
                                           activation="gelu", batch_first=True, norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.fusion_depth,
                                             enable_nested_tensor=False)
        self.fusion_norm = nn.LayerNorm(cfg.dim)
        self.pcr_head = nn.Sequential(nn.LayerNorm(cfg.dim), nn.Linear(cfg.dim, 1))
        self.world_predictor = TimeCausalWorldPredictor(cfg)
        self.reconstruction = nn.Conv3d(cfg.dim, 3, 1)
        self.flow = ConditionalMRIFlow(cfg) if cfg.enable_flow else None

    def train(self, mode=True):
        super().train(mode)
        self.teacher_encoder.eval()
        return self

    def _backbone(self, inp, *, validated=False):
        if not validated:
            inp.validate()
        if inp.clinical.shape[1] != self.cfg.clinical_dim:
            raise ValueError("Input clinical dimension differs from model configuration")
        batch, tokens_per_visit = len(inp.images), math.prod(self.cfg.token_grid)
        observed = self.encoder(inp.images[inp.observed_mask])
        spatial = observed.new_zeros(batch, 4, tokens_per_visit, self.cfg.dim)
        spatial[inp.observed_mask] = observed
        spatial = spatial + self.stage.weight[None, :, None]
        clinical = self.clinical(torch.cat((inp.clinical, inp.clinical_mask.to(observed.dtype)), -1))
        arm = (self.arm(inp.arm_id) + self.arm_known_at(inp.arm_known_at.clamp(0, 3))
               * inp.arm_mask[:, None].to(observed.dtype))
        context = torch.cat((spatial.flatten(1, 2), clinical[:, None], arm[:, None]), 1)
        context_padding = torch.cat((~inp.observed_mask[:, :, None]
                                     .expand(-1, -1, tokens_per_visit).flatten(1),
                                     torch.zeros(batch, 2, dtype=torch.bool, device=observed.device)), 1)
        dynamic = self.dynamics_queries[None] + self.stage.weight[None, :, None]
        dynamic = dynamic.expand(batch, -1, -1, -1)
        pcr = self.pcr_queries[None].expand(batch, -1, -1) + self.stage(inp.landmark)[:, None]
        packed = torch.cat((context, dynamic.flatten(1, 2), pcr), 1)
        padding = torch.cat((context_padding,
                             ~inp.query_mask[:, :, None].expand(-1, -1, self.cfg.state_queries).flatten(1),
                             torch.zeros(batch, self.cfg.pcr_queries, dtype=torch.bool,
                                         device=observed.device)), 1)
        length, context_length = packed.shape[1], context.shape[1]
        causal = torch.ones(length, length, dtype=torch.bool, device=observed.device).triu(1)
        causal[:context_length, :context_length] = False
        hidden = self.fusion_norm(self.fusion(packed, mask=causal, src_key_padding_mask=padding))
        dynamics = hidden[:, context_length:context_length + 4 * self.cfg.state_queries]
        dynamics = dynamics.reshape(batch, 4, self.cfg.state_queries, self.cfg.dim)
        dynamics = dynamics.masked_fill(~inp.query_mask[:, :, None, None], 0)
        pcr_features = hidden[:, -self.cfg.pcr_queries:]
        output = {"pcr_logit": self.pcr_head(pcr_features.mean(1)).squeeze(-1),
                  "dynamics_features": dynamics, "pcr_features": pcr_features}
        return output, observed

    def backbone(self, inp: RawMRIInput):
        """Only legal observed data; never runs a teacher, world predictor or flow."""
        return self._backbone(inp)[0]

    def forward(self, inp: RawMRIInput):
        return self.backbone(inp)

    @torch.no_grad()
    def _teacher_states(self, inp, sup):
        known = inp.observed_mask | sup.future_mask
        scans = torch.zeros_like(inp.images)
        scans[inp.observed_mask] = inp.images[inp.observed_mask]
        # Index before encoding: masked future NaNs are never read by the teacher.
        scans[sup.future_mask] = sup.future[sup.future_mask]
        states = inp.images.new_zeros(len(inp.images), 4, math.prod(self.cfg.token_grid), self.cfg.dim)
        states[known] = self.teacher_encoder(scans[known]).to(states.dtype)
        return states, known

    def teacher_forcing_loss(self, inp, sup, dynamics, return_details=False):
        """True predecessor -> next state with patient-mean valid-pair reduction.

        For landmark k only transitions j>=k and requested j+1 are active.
        Missing target or source removes that pair; no invented adjacent visit.
        The teacher-forced predictions never condition the pCR branch.
        """
        inp.validate()
        sup.validate(inp)
        return self._teacher_forcing_loss(inp, sup, dynamics, return_details)

    def _teacher_forcing_loss(self, inp, sup, dynamics, return_details=False):
        expected = (len(inp.images), 4, self.cfg.state_queries, self.cfg.dim)
        if dynamics.shape != expected:
            raise ValueError("Dynamics must have shape [B,4,K,dim]")
        states, known = self._teacher_states(inp, sup)
        stage = torch.arange(3, device=inp.images.device)[None]
        source_mask = known[:, :-1] & (stage >= inp.landmark[:, None]) & inp.query_mask[:, 1:]
        pair_mask = source_mask & known[:, 1:]
        prediction = self.world_predictor(states[:, :-1], dynamics[:, 1:], source_mask)
        pair_losses = (prediction - states[:, 1:]).abs().mean((-1, -2))
        pair_counts = pair_mask.sum(1)
        patient_loss = pair_losses.masked_fill(~pair_mask, 0).sum(1) / pair_counts.clamp_min(1)
        loss = patient_loss[pair_counts > 0].mean() if (pair_counts > 0).any() else dynamics.sum() * 0
        if return_details:
            return loss, {"prediction": prediction, "pair_mask": pair_mask,
                          "source_mask": source_mask, "teacher_states": states,
                          "source_states": states[:, :-1], "targets": states[:, 1:],
                          "pair_losses": pair_losses, "known_mask": known}
        return loss

    def _rollout(self, inp, dynamics=None):
        """Free rollout from real S_k; no access to supervision or future MRI."""
        inp.validate()
        batch, count = len(inp.images), math.prod(self.cfg.token_grid)
        needs_future = inp.query_mask.any(1)
        at_landmark = inp.observed_mask.gather(1, inp.landmark[:, None]).squeeze(1)
        if (needs_future & ~at_landmark).any():
            raise ValueError("Forecast requires a real MRI at the requested landmark; pCR can still use legal history")
        result = inp.images.new_zeros(batch, 4, count, self.cfg.dim)
        if not needs_future.any():
            return result
        # Sparse output requests may still need intermediate transitions.
        full_query = torch.arange(4, device=inp.images.device)[None] > inp.landmark[:, None]
        if dynamics is None or not torch.equal(inp.query_mask, full_query):
            dynamics = self.backbone(replace(inp, query_mask=full_query))["dynamics_features"]
        states = inp.images.new_zeros(batch, 4, count, self.cfg.dim)
        current_rows = needs_future.nonzero().flatten()
        with torch.no_grad():
            initial = self.teacher_encoder(inp.images[current_rows, inp.landmark[current_rows]])
        states[current_rows, inp.landmark[current_rows]] = initial.to(states.dtype)
        source_stages = torch.arange(3, device=inp.images.device)[None]
        rows = torch.arange(batch, device=inp.images.device)
        for source_stage in range(3):
            updating = needs_future & (inp.landmark <= source_stage)
            if not updating.any():
                continue
            active = (source_stages >= inp.landmark[:, None]) & (source_stages <= source_stage) & needs_future[:, None]
            prediction = self.world_predictor(states[:, :-1], dynamics[:, 1:], active)
            # Clone avoids changing a predecessor tensor needed by backward.
            states = states.clone()
            states[rows[updating], source_stage + 1] = prediction[updating, source_stage].to(states.dtype)
        return states.masked_fill(~inp.query_mask[:, :, None, None], 0)

    @torch.no_grad()
    def forecast_states(self, inp: RawMRIInput):
        return self._rollout(inp)

    @torch.no_grad()
    def forecast(self, inp: RawMRIInput):
        output = self(inp)
        return {**output, "state_prediction": self._rollout(inp, output["dynamics_features"]),
                "query_mask": inp.query_mask.clone()}

    @staticmethod
    def _latest_mri(inp):
        stages = torch.arange(4, device=inp.images.device)[None]
        latest = stages.expand_as(inp.observed_mask).masked_fill(~inp.observed_mask, -1).max(1).values
        return inp.images[torch.arange(len(inp.images), device=inp.images.device), latest]

    def compute_loss(self, inp: RawMRIInput, sup: RawMRISupervision, *,
                     task_weight=1., jepa_weight=1., flow_weight=.1,
                     reconstruction_weight=.1, variance_weight=.01,
                     representation_only=False, variance_definition="legacy"):
        inp.validate()
        weights = (task_weight, jepa_weight, flow_weight, reconstruction_weight, variance_weight)
        if any(not math.isfinite(w) or w < 0 for w in weights):
            raise ValueError("Loss weights must be finite and nonnegative")
        if variance_definition not in {"legacy", "off", "patient_axis_stage_token_fp32"}:
            raise ValueError("Unsupported variance_definition")
        if variance_definition == "off" and variance_weight:
            raise ValueError("variance_definition=off requires variance_weight=0")
        sup.validate(inp, future_values=bool(jepa_weight or
                     (self.flow is not None and flow_weight and not representation_only)))
        output, observed = self._backbone(inp, validated=True)
        zero = observed.sum() * 0
        task = world = flow = reconstruction = variance = zero
        pair_count = zero.detach()
        if not representation_only and task_weight and sup.label_mask.any():
            task = F.binary_cross_entropy_with_logits(output["pcr_logit"][sup.label_mask],
                                                      sup.label[sup.label_mask].to(observed.dtype))
        if jepa_weight:
            world, details = self._teacher_forcing_loss(inp, sup, output["dynamics_features"],
                                                       return_details=True)
            pair_count = details["pair_mask"].sum().to(observed.dtype)
        if reconstruction_weight:
            spatial = observed.transpose(1, 2).reshape(len(observed), self.cfg.dim, *self.cfg.token_grid)
            reconstructed = F.interpolate(self.reconstruction(spatial), size=inp.images.shape[-3:],
                                            mode="trilinear", align_corners=False)
            reconstruction = F.mse_loss(reconstructed, inp.images[inp.observed_mask].detach())
        if variance_weight:
            variance = self.variance_penalty(observed, inp.observed_mask, variance_definition)
        if not representation_only and self.flow is not None and flow_weight and sup.future_mask.any():
            # Never feed teacher-forced future predictions to FM conditions.
            forecast = self._rollout(inp, output["dynamics_features"])
            selected = sup.future_mask
            x1 = sup.future[selected].detach()
            x0 = torch.randn_like(x1)
            tau = torch.rand(len(x1), device=x1.device, dtype=x1.dtype)
            t = tau[:, None, None, None, None]
            x_tau = (1 - t) * x0 + t * x1
            source = self._latest_mri(inp)[:, None].expand(-1, 4, -1, -1, -1, -1)[selected]
            stage = torch.arange(4, device=x1.device)[None].expand(len(inp.images), -1)[selected]
            velocity = self.flow(x_tau, tau, source, forecast[selected], stage)
            flow = F.mse_loss(velocity, x1 - x0)
        loss = (task_weight * task + jepa_weight * world + flow_weight * flow
                + reconstruction_weight * reconstruction + variance_weight * variance)
        return loss, {"loss": loss, "task": task, "jepa": world, "world_model": world,
                      "flow": flow, "reconstruction": reconstruction, "variance": variance,
                      "label_count": sup.label_mask.sum().to(observed.dtype),
                      "future_count": sup.future_mask.sum().to(observed.dtype),
                      "pair_count": pair_count}

    @staticmethod
    def variance_penalty(observed, observed_mask, definition="legacy"):
        if definition == "legacy":
            std = (observed.flatten(0, 1).var(0, unbiased=False) + 1e-4).sqrt()
            return F.relu(1 - std).mean()
        if definition == "off":
            return observed.sum() * 0
        if definition != "patient_axis_stage_token_fp32":
            raise ValueError("Unsupported variance_definition")
        # Compare patients at matching stage/token/channel; spatial diversity cannot satisfy this term.
        states = observed.new_zeros(*observed_mask.shape, *observed.shape[1:], dtype=torch.float32)
        states[observed_mask] = observed.float()
        losses = []
        for stage in range(4):
            valid = observed_mask[:, stage]
            if valid.sum() >= 2:
                std = (states[valid, stage].var(0, unbiased=False) + 1e-4).sqrt()
                losses.append(F.relu(1 - std).mean())
        return torch.stack(losses).mean() if losses else observed.float().sum() * 0

    @torch.no_grad()
    def update_teacher(self):
        for teacher, student in zip(self.teacher_encoder.parameters(), self.encoder.parameters(), strict=True):
            teacher.lerp_(student, 1 - self.cfg.ema_decay)
        for teacher, student in zip(self.teacher_encoder.buffers(), self.encoder.buffers(), strict=True):
            teacher.copy_(student)

    @torch.no_grad()
    def generate(self, inp: RawMRIInput, steps=4, seed=0, method="heun"):
        """Optional MRI flow, conditioned on free forecasts, never teacher forcing."""
        if self.flow is None:
            raise ValueError("Generation is disabled in this model configuration")
        if isinstance(steps, bool) or not isinstance(steps, int) or steps < 1:
            raise ValueError("steps must be a positive integer")
        if method not in {"euler", "heun"}:
            raise ValueError("method must be euler or heun")
        training = self.training
        self.eval()
        try:
            output = self.forecast(inp)
            selected = inp.query_mask
            images = torch.zeros_like(inp.images)
            if selected.any():
                noise_device = "cpu" if inp.images.device.type == "mps" else inp.images.device
                generator = torch.Generator(device=noise_device).manual_seed(seed)
                value = torch.randn(inp.images[selected].shape, dtype=inp.images.dtype,
                                    device=noise_device, generator=generator).to(inp.images.device)
                source = self._latest_mri(inp)[:, None].expand(-1, 4, -1, -1, -1, -1)[selected]
                future = output["state_prediction"][selected]
                stage = torch.arange(4, device=inp.images.device)[None].expand(len(inp.images), -1)[selected]
                dt = 1 / steps
                for step in range(steps):
                    tau = value.new_full((len(value),), step * dt)
                    velocity = self.flow(value, tau, source, future, stage)
                    if method == "heun":
                        proposal = value + dt * velocity
                        velocity = .5 * (velocity + self.flow(proposal, tau + dt, source, future, stage))
                    value = value + dt * velocity
                images[selected] = value
            return {"future_images": images, "pcr_probability": output["pcr_logit"].sigmoid(),
                    "query_mask": inp.query_mask.clone()}
        finally:
            self.train(training)
