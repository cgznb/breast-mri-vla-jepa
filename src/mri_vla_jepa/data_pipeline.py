"""Bounded, deterministic batch loading and metadata-bound ROI32 image caches."""
from __future__ import annotations

from collections import OrderedDict, deque
from concurrent.futures import ThreadPoolExecutor
import copy
import os
from pathlib import Path
import threading

import numpy as np

from .io import read_json, write_json

CACHE_SCHEMA = "registered_roi32_image_cache_v1"
IMAGE_SHAPE = (3, 32, 128, 128)


def file_identity(path):
    path = Path(path).resolve()
    stat = path.stat()
    return {"path": str(path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}


class ROI32ImageCache:
    def __init__(self, directory, *, normalization, phase_order, max_open=64):
        self.directory = Path(directory).resolve()
        self.path = self.directory / "index.json"
        self.index = read_json(self.path)
        if (self.index.get("schema") != CACHE_SCHEMA or self.index.get("shape") != list(IMAGE_SHAPE)
                or self.index.get("dtype") != "float32"
                or self.index.get("image_preprocessing") != {"mode": "preprocessed_roi32"}
                or self.index.get("image_normalization") != normalization
                or self.index.get("phase_order") != phase_order):
            raise ValueError("ROI32 cache preprocessing/normalization/shape declaration differs")
        self.entries = {}
        for entry in self.index["images"]:
            key = entry["source"]["path"]
            if key in self.entries:
                raise ValueError("Duplicate source image in ROI32 cache")
            cached = Path(entry["cached"]["path"]).resolve()
            if not cached.is_relative_to(self.directory) or cached.suffix != ".npy":
                raise ValueError("ROI32 cached images must be NPY files inside the cache directory")
            self.entries[key] = entry
        self.max_open = max_open
        self.arrays = OrderedDict()
        self.lock = threading.Lock()

    def read(self, source):
        key = str(Path(source).resolve())
        with self.lock:
            if key not in self.entries:
                raise ValueError("Image is absent from the declared ROI32 cache")
            entry = self.entries[key]
            if (file_identity(source) != entry["source"]
                    or file_identity(entry["cached"]["path"]) != entry["cached"]):
                raise ValueError("ROI32 image cache is stale: source or cached file identity changed")
            if key not in self.arrays:
                # Copy-on-write mapping is writable for torch views without changing the file.
                array = np.load(entry["cached"]["path"], mmap_mode="c", allow_pickle=False)
                if array.shape != IMAGE_SHAPE or array.dtype != np.float32 or not np.isfinite(array).all():
                    raise ValueError("ROI32 cached image must be finite float32 [3,32,128,128]")
                self.arrays[key] = array
                if len(self.arrays) > self.max_open:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(key)
            return self.arrays[key]

    def signature(self):
        if read_json(self.path) != self.index:
            raise ValueError("ROI32 cache index changed during the run")
        for entry in self.index["images"]:
            for kind in ("source", "cached"):
                if file_identity(entry[kind]["path"]) != entry[kind]:
                    raise ValueError("ROI32 image cache is stale: file identity changed")
        return {"directory": str(self.directory), "index": copy.deepcopy(self.index)}


def prepare_image_cache(store, directory, workers=4):
    if not store.preprocessed_roi32 or store.image_cache is not None:
        raise ValueError("Cache preparation requires the original preprocessed ROI32 NPZ store")
    if type(workers) is not int or not 1 <= workers <= 8:
        raise ValueError("Cache workers must be an integer from one to eight")
    visits = sorted(((str(store.resolve(visit["image"])), visit)
                     for patient in store.patients for visit in patient["visits"] if visit is not None))
    directory = Path(directory).resolve()
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    if (directory / "index.json").exists():
        existing = ROI32ImageCache(directory, normalization=store.image_normalization,
                                  phase_order=store.manifest["phase_order"])
        if set(existing.entries) != {source for source, _ in visits}:
            raise ValueError("ROI32 image cache population differs from the manifest")
        existing.signature()
        return {"images": len(existing.entries), "reused": True, "directory": str(directory)}

    def prepare(item):
        index, (source, visit) = item
        before = file_identity(source)
        image, _ = store._read_scan(visit)
        path = directory / f"image_{index:04d}.npy"
        if path.exists():
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if saved.shape != image.shape or saved.dtype != image.dtype or not np.array_equal(saved, image):
                raise ValueError("An existing partial cache image differs from its source")
        else:
            temporary = path.with_suffix(".npy.tmp")
            with temporary.open("wb") as handle:
                os.chmod(temporary, 0o600)
                np.save(handle, image, allow_pickle=False)
            os.replace(temporary, path)
            saved = np.load(path, mmap_mode="r", allow_pickle=False)
            if not np.array_equal(saved, image):
                raise ValueError("Derived ROI32 cache image differs from its source")
        if file_identity(source) != before:
            raise ValueError("Source image changed while preparing its cache")
        return {"source": before, "cached": file_identity(path)}

    entries = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        for entry in executor.map(prepare, enumerate(visits)):
            entries.append(entry)
            if len(entries) % 250 == 0:
                print(f"Cache verified {len(entries)}/{len(visits)} images", flush=True)
    index = {"schema": CACHE_SCHEMA, "shape": list(IMAGE_SHAPE), "dtype": "float32",
             "image_preprocessing": copy.deepcopy(store.image_preprocessing),
             "image_normalization": copy.deepcopy(store.image_normalization),
             "phase_order": copy.deepcopy(store.manifest["phase_order"]), "images": entries}
    write_json(directory / "index.json", index)
    (directory / "index.json").chmod(0o600)
    return {"images": len(entries), "reused": False, "directory": str(directory),
            "total_bytes": sum(e["cached"]["size"] for e in entries),
            "all_images_value_equal_to_npz": True, "zero_background_preserved": True}


class BatchLoader:
    """Workers consume preplanned tasks; the training sampler never advances here."""
    def __init__(self, store, *, prefetch_batches=0, workers=2, pin_memory=False,
                 future_supervision=True):
        self.store, self.depth, self.pin_memory = store, prefetch_batches, pin_memory
        self.future_supervision = future_supervision
        self.executor = ThreadPoolExecutor(max_workers=workers) if self.depth else None
        self.pending = deque()
        self.planned = False
        self.tasks = iter(())

    def __enter__(self):
        return self

    def __exit__(self, *args):
        if self.executor:
            self.executor.shutdown(wait=True, cancel_futures=True)
        self.pending.clear()

    def _batch(self, tasks):
        if not self.future_supervision:
            return self.store.batch(tasks, pin_memory=self.pin_memory, future_supervision=False)
        if self.pin_memory:
            return self.store.batch(tasks, pin_memory=True)
        return self.store.batch(tasks)

    def _fill(self):
        while len(self.pending) < self.depth:
            tasks = next(self.tasks, None)
            if tasks is None:
                break
            self.pending.append((tasks, self.executor.submit(self._batch, tasks)))

    def plan(self, tasks):
        if self.pending:
            raise RuntimeError("Cannot replace a batch plan while unconsumed batches remain")
        self.tasks, self.planned = iter(tasks), True
        self._fill()

    def fetch(self, tasks):
        if self.executor is None:
            return self._batch(tasks)
        if not self.pending:
            raise RuntimeError("Prefetch plan exhausted before the consumed training cursor")
        expected, future = self.pending.popleft()
        if expected != tasks:
            raise RuntimeError("Prefetched tasks differ from the main training sampler")
        result = future.result()
        self._fill()
        return result

    def finish_epoch(self):
        if self.pending:
            raise RuntimeError("Unconsumed prefetched batches at the end of an epoch")
        self.planned = False
