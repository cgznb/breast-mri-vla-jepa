"""Dedicated CLI for teacher-forced MRI VLA-JEPA and direct pCR."""
from __future__ import annotations

import argparse
from pathlib import Path
from importlib import resources
import json
import yaml
from .io import write_json
from .data import RawMRIStore, make_raw_synthetic
from .training import evaluate, predict, train
from .train_config import from_dict, load_config

COMMANDS = {"train-vla-jepa", "smoke-vla-jepa", "evaluate-vla-jepa", "predict-vla-jepa"}


def _landmark(value):
    if value == "all_observed":
        return value
    try:
        stage = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("landmark must be 0, 1, 2, 3 or all_observed") from exc
    if str(stage) != value or not 0 <= stage <= 3:
        raise argparse.ArgumentTypeError("landmark must be 0, 1, 2, 3 or all_observed")
    return stage


def parser():
    root = argparse.ArgumentParser(description="MRI VLA-JEPA: shared task trunk and teacher-forced world model")
    sub = root.add_subparsers(dest="command", required=True)
    training = sub.add_parser("train-vla-jepa")
    training.add_argument("--manifest", required=True)
    training.add_argument("--config", required=True)
    training.add_argument("--output", required=True)
    training.add_argument("--resume", action="store_true")
    training.add_argument("--stop-after", type=int)
    smoke = sub.add_parser("smoke-vla-jepa")
    smoke.add_argument("--config", help="Defaults to the packaged CPU fixed-T0 smoke configuration")
    smoke.add_argument("--output", required=True)
    evaluation = sub.add_parser("evaluate-vla-jepa")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--manifest", required=True)
    evaluation.add_argument("--output", required=True)
    evaluation.add_argument("--split", choices=("train", "val", "test"), default="test")
    evaluation.add_argument("--landmark", type=_landmark,
                            help="0..3 or all_observed; defaults to the checkpoint profile")
    evaluation.add_argument("--bootstrap", type=int, default=1000)
    evaluation.add_argument("--device", default="cpu")
    evaluation.add_argument("--allow-synthetic", action="store_true")
    prediction = sub.add_parser("predict-vla-jepa")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--manifest", required=True)
    prediction.add_argument("--patient-key", required=True)
    prediction.add_argument("--landmark", type=int, choices=range(4), default=0)
    prediction.add_argument("--output", required=True)
    prediction.add_argument("--device", default="cpu")
    prediction.add_argument("--allow-synthetic", action="store_true")
    prediction.add_argument("--forecast-states", action="store_true",
                            help="Explicitly run autonomous latent-state rollout and save .states.npz")
    prediction.add_argument("--generate", action="store_true")
    prediction.add_argument("--steps", type=int, default=20)
    prediction.add_argument("--seed", type=int, default=0)
    return root


def main(argv=None):
    args = parser().parse_args(argv)
    if args.command == "train-vla-jepa":
        cfg = load_config(args.config)
        store = RawMRIStore(args.manifest, cfg.image_shape, cfg.allow_synthetic, image_cache=cfg.image_cache)
        report = train(store, cfg, args.output, resume=args.resume, stop_after=args.stop_after)
    elif args.command == "smoke-vla-jepa":
        if args.config is None:
            resource = resources.files("mri_vla_jepa").joinpath(
                "resources", "configs", "raw_vla_jepa_smoke_t0.yaml")
            cfg = from_dict(yaml.safe_load(resource.read_text(encoding="utf-8")))
        else:
            cfg = load_config(args.config)
        if not cfg.allow_synthetic or cfg.device != "cpu":
            raise ValueError("VLA smoke requires explicit synthetic CPU configuration")
        out = Path(args.output).resolve()
        if (out / "data/manifest.json").exists() or (out / "run").exists():
            raise FileExistsError("Choose a new smoke output directory")
        manifest = make_raw_synthetic(out / "data", image_shape=cfg.image_shape, seed=cfg.seed)
        store = RawMRIStore(manifest, cfg.image_shape, allow_synthetic=True)
        trained = train(store, cfg, out / "run")
        checkpoint = out / "run/best.pt"
        evaluated = evaluate(checkpoint, manifest, out / "evaluation.json", split="val",
                             allow_synthetic=True, bootstrap=0)
        prediction = predict(checkpoint, manifest, "synthetic_0", out / "prediction.json",
                             allow_synthetic=True,
                             forecast_states=cfg.jepa_weight > 0 or cfg.flow_weight > 0,
                             generate=cfg.model.enable_flow and cfg.flow_weight > 0,
                             steps=2, seed=cfg.seed)
        report = {"engineering_only": True, "clinical_validation": False, "training": trained,
                  "evaluation": {k: v for k, v in evaluated.items() if k != "rows"}, "prediction": prediction}
        write_json(out / "smoke_report.json", report)
    elif args.command == "evaluate-vla-jepa":
        report = evaluate(args.checkpoint, args.manifest, args.output, split=args.split,
                          landmark=args.landmark, bootstrap=args.bootstrap, device=args.device,
                          allow_synthetic=args.allow_synthetic)
    else:
        report = predict(args.checkpoint, args.manifest, args.patient_key, args.output,
                         landmark=args.landmark, device=args.device, allow_synthetic=args.allow_synthetic,
                         forecast_states=args.forecast_states, generate=args.generate,
                         steps=args.steps, seed=args.seed)
    print(json.dumps({k: v for k, v in report.items() if k != "rows"}, indent=2, ensure_ascii=False))
    return report


def entrypoint():
    """Console scripts require a None return value for a successful exit."""
    main()


if __name__ == "__main__":
    entrypoint()
