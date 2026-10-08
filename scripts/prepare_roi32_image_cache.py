#!/usr/bin/env python3
"""Create and verify exact ROI32 NPY image views from the original NPZ files."""
import argparse
from pathlib import Path

from mri_vla_jepa.data import RawMRIStore
from mri_vla_jepa.data_pipeline import prepare_image_cache
from mri_vla_jepa.io import write_json

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=ROOT / "data/registered_roi32/manifest.json")
    parser.add_argument("--output", type=Path, default=ROOT / "data/registered_roi32/image_cache_20261004")
    parser.add_argument("--report", type=Path, default=ROOT / "reports/roi32_image_cache_20261004.json")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 8:
        parser.error("workers must be 1..8")
    store = RawMRIStore(args.manifest)
    report = prepare_image_cache(store, args.output, args.workers)
    write_json(args.report, report)
    print(f"ROI32 image cache ready: {report['images']} images at {args.output}", flush=True)


if __name__ == "__main__":
    main()
