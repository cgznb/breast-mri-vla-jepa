"""Evaluate trained world/copy states without training or exporting patient rows."""
from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.diagnostics import evaluate_world
from mri_vla_jepa.io import stable_hash, write_json
from mri_vla_jepa.training import (LEGACY_CHECKPOINT_SCHEMA, _patient_partitions,
                                  _source_snapshot, load_trained)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "val", "test"), default="val")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--allow-synthetic", action="store_true")
    args = parser.parse_args()
    identity = (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns)
    model, saved_cfg, state = load_trained(args.checkpoint, args.device)
    torch.set_num_threads(saved_cfg.threads)
    cfg = replace(saved_cfg, device=args.device, precision="fp32")
    store = RawMRIStore(args.manifest, cfg.image_shape, args.allow_synthetic,
                        state["normalization"], image_cache=cfg.image_cache)
    partitions = _patient_partitions(store)
    legacy = state["schema"] == LEGACY_CHECKPOINT_SCHEMA
    if legacy:
        partitions = {split: sorted(stable_hash(key) for key in keys) for split, keys in partitions.items()}
    if partitions != state["patient_partitions"]:
        raise ValueError("World evaluation patient partitions differ from checkpoint")
    if not legacy and state["data_signature"] != store.signature():
        raise ValueError("World evaluation data assets or manifest differ from checkpoint")
    result = evaluate_world(model, store, cfg, split=args.split, batch_size=args.batch_size)
    if identity != (args.checkpoint.stat().st_size, args.checkpoint.stat().st_mtime_ns):
        raise ValueError("Checkpoint changed while evaluating")
    result.update(checkpoint=str(args.checkpoint.resolve()),
                  checkpoint_identity={"path": str(args.checkpoint.resolve()),
                                       "size_bytes": identity[0], "mtime_ns": identity[1]},
                  checkpoint_epoch=(state.get("epoch_state") or {}).get("completed_epochs"),
                  checkpoint_steps=state["completed_steps"],
                  synthetic=bool(store.manifest.get("synthetic", False)),
                  runtime_source_matches_checkpoint=state.get("source_snapshot") == _source_snapshot(),
                  checkpoint_file_unchanged=True)
    write_json(args.output, result)
    print(f"World evaluation {result['status']}; aggregate output: {args.output}")


if __name__ == "__main__":
    main()
