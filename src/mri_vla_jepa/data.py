"""Source-only raw three-phase MRI reads for canonical T0..T3 JEPA prefixes."""
from __future__ import annotations

from collections import defaultdict
from pathlib import Path
import copy
import math
import re

import numpy as np
import torch
import torch.nn.functional as F

from .arm import CLINICAL_FEATURES, OFFICIAL_ARMS, make_arm_record, normalize_patient_key, stage_index, validate_arm_record, visible_arm
from .constants import PHASES
from .io import read_json, write_json
from .contracts import RawMRIInput, RawMRISupervision

SCHEMA = "responsewm_raw_mri_v1"
ROI32_SHAPE = (3, 32, 128, 128)
IMAGE_NORMALIZATION_KEYS = (
    "mean", "std", "fit_split", "scope", "background_policy", "already_normalized", "normalization_policy",
)


def manifest_without_legacy_checksums(value):
    if isinstance(value, dict):
        return {key: manifest_without_legacy_checksums(item) for key, item in value.items()
                if not re.search(r"(?:^|_)(?:sha\d*|hash(?:es)?|digest|checksum)(?:_|$)", key, re.IGNORECASE)}
    if isinstance(value, list):
        return [manifest_without_legacy_checksums(item) for item in value]
    return copy.deepcopy(value)


def canonical_image_normalization(value):
    """Keep the declared training scale, without copying unrelated cache metadata."""
    if not isinstance(value, dict) or any(key not in value for key in IMAGE_NORMALIZATION_KEYS):
        raise ValueError("ROI32 requires the complete image_normalization declaration")
    result = {key: copy.deepcopy(value[key]) for key in IMAGE_NORMALIZATION_KEYS}
    if (type(result["mean"]) not in {int, float} or not math.isfinite(result["mean"])
            or type(result["std"]) not in {int, float} or not math.isfinite(result["std"])
            or result["std"] <= 0 or result["fit_split"] != "train"
            or result["background_policy"] != "preserve_zero" or result["already_normalized"] is not True
            or any(not isinstance(result[key], str) or not result[key].strip()
                   for key in ("scope", "normalization_policy"))):
        raise ValueError("ROI32 image_normalization requires a finite training scale and preserved zero background")
    return result


class RawMRIStore:
    def __init__(self, manifest, image_shape=(32, 128, 128), allow_synthetic=False,
                 normalization_state=None, image_preprocessing=None, image_cache=None):
        self.path = Path(manifest).resolve()
        self.manifest = read_json(self.path)
        self.image_shape = tuple(image_shape)
        if len(self.image_shape) != 3 or any(type(v) is not int or v < 1 for v in self.image_shape):
            raise ValueError("image_shape must contain three positive integer D,H,W sizes")
        m = self.manifest
        self.image_preprocessing = copy.deepcopy(m.get("image_preprocessing", {"mode": "raw_t0_zscore"}))
        if (not isinstance(self.image_preprocessing, dict) or set(self.image_preprocessing) != {"mode"}
                or self.image_preprocessing["mode"] not in {"raw_t0_zscore", "preprocessed_roi32"}):
            raise ValueError("image_preprocessing.mode must be raw_t0_zscore or preprocessed_roi32")
        if image_preprocessing is not None and image_preprocessing != self.image_preprocessing:
            raise ValueError("Requested image_preprocessing does not match the manifest")
        self.preprocessed_roi32 = self.image_preprocessing["mode"] == "preprocessed_roi32"
        self.image_normalization = None
        if self.preprocessed_roi32:
            if self.image_shape != ROI32_SHAPE[1:]:
                raise ValueError("preprocessed_roi32 requires image_shape [32,128,128]; resizing is forbidden")
            self.image_normalization = canonical_image_normalization(m.get("image_normalization"))
            if m["image_normalization"] != self.image_normalization:
                raise ValueError("ROI32 manifest image_normalization must contain only the declared scale fields")
        elif "image_normalization" in m:
            raise ValueError("Pre-normalized image_normalization requires image_preprocessing.mode=preprocessed_roi32")
        if (m.get("schema") != SCHEMA or m.get("canonical_stages") != [0, 1, 2, 3]
                or any(type(v) is not int for v in m.get("canonical_stages", []))
                or m.get("phase_order") != PHASES or m.get("time_basis") != "stage_index"):
            raise ValueError("Expected raw MRI schema with canonical stages, phase order and stage-index time")
        if m.get("synthetic", False) and not allow_synthetic:
            raise ValueError("Synthetic raw MRI requires explicit allow_synthetic")
        if m.get("clinical_features") != list(CLINICAL_FEATURES):
            raise ValueError("Raw MRI clinical features must be age, HR, HER2, MP in the declared order")
        self.clinical_dim = len(CLINICAL_FEATURES)
        self.patients = m.get("patients", [])
        if not self.patients:
            raise ValueError("Empty raw MRI cohort")
        self.by_split = defaultdict(list)
        keys, assets = set(), {}
        for index, patient in enumerate(self.patients):
            key = patient.get("patient_key")
            if not isinstance(key, str) or not key or normalize_patient_key(key) in keys:
                raise ValueError("Duplicate patient or patient overlap across splits")
            keys.add(normalize_patient_key(key))
            split = patient.get("split")
            if split not in {"train", "val", "test"}:
                raise ValueError("Patient split must be train, val or test")
            self.by_split[split].append(index)
            geometry = patient.get("geometry", {})
            if (geometry.get("source_only") is not True or type(geometry.get("source_stage")) is not int
                    or geometry.get("source_stage") != 0
                    or geometry.get("roi_source") != "T0" or geometry.get("orientation") != "RAS"
                    or geometry.get("resampling") != "source_grid" or not geometry.get("grid_id")):
                raise ValueError("MRI geometry/ROI must declare a T0 source-only RAS grid and source-grid resampling")
            visits = patient.get("visits")
            if not isinstance(visits, list) or len(visits) != 4 or visits[0] is None:
                raise ValueError("Four canonical visit slots with an actual T0 are required; missing visits use null")
            for stage, visit in enumerate(visits):
                if visit is None:
                    continue
                if (type(visit.get("stage")) is not int or visit.get("stage") != stage
                        or type(visit.get("available_at")) is not int or not stage <= visit["available_at"] <= 3):
                    raise ValueError("Visit stage/availability must agree with canonical chronology")
                if visit.get("grid_id") != geometry["grid_id"]:
                    raise ValueError("MRI visits must share the declared source grid; future-derived ROI is forbidden")
                paths = self._visit_paths(visit)
                if self.preprocessed_roi32 and ("image" not in visit or not paths[0].endswith(".npz")):
                    raise ValueError("preprocessed_roi32 requires a stacked .npz image for each observed visit")
                if not self.preprocessed_roi32 and any(path.endswith(".npz") for path in paths):
                    raise ValueError("NPZ ROI32 input requires image_preprocessing.mode=preprocessed_roi32")
                for path in paths:
                    canonical = str(self.resolve(path))
                    if canonical in assets:
                        raise ValueError("An MRI asset overlaps patients, stages or splits")
                    assets[canonical] = (normalize_patient_key(key), stage)
            if visits[0]["available_at"] != 0:
                raise ValueError("T0 source geometry requires baseline available at stage zero")
            self._clinical_values(patient, 3)
            validate_arm_record(patient.get("arm"))
            # The target is intentionally not inspected here: inference never
            # needs a label and must work even if target records are inaccessible.
        self._normalization = self._fit_clinical() if normalization_state is None else self._restore_normalization(normalization_state)
        self._validate_normalization()
        self.image_cache = None
        if image_cache is not None:
            if not self.preprocessed_roi32:
                raise ValueError("An ROI32 image cache requires preprocessed_roi32 mode")
            from .data_pipeline import ROI32ImageCache
            path = Path(image_cache)
            if not path.is_absolute():
                path = self.path.parent / path
            self.image_cache = ROI32ImageCache(path, normalization=self.image_normalization,
                                              phase_order=m["phase_order"])
            sources = {str(self.resolve(visit["image"])) for patient in self.patients
                       for visit in patient["visits"] if visit is not None}
            if sources != set(self.image_cache.entries):
                raise ValueError("ROI32 image cache population differs from the manifest")

    def resolve(self, path):
        p = Path(path)
        return p.resolve() if p.is_absolute() else (self.path.parent / p).resolve()

    @staticmethod
    def _visit_paths(visit):
        image, phases = visit.get("image"), visit.get("phases")
        if (image is None) == (phases is None):
            raise ValueError("Each visit requires either one stacked image or three ordered phase paths")
        paths = [image] if image is not None else phases
        if not isinstance(paths, list) or len(paths) != (1 if image is not None else 3):
            raise ValueError("Exactly three phase paths are required")
        if any(not isinstance(p, str) or not p for p in paths):
            raise ValueError("MRI paths must be nonempty strings")
        if image is not None and not image.endswith((".npy", ".npz")):
            raise ValueError("Stacked [3,D,H,W] images must be .npy or .npz; use phases for NIfTI")
        return paths

    def _clinical_values(self, patient, landmark):
        clinical = patient.get("clinical", {})
        values, known = clinical.get("values", []), clinical.get("known_at", [])
        if len(values) != self.clinical_dim or len(known) != self.clinical_dim:
            raise ValueError("Clinical values/known_at must follow the four declared features")
        mask = []
        for i, (value, when) in enumerate(zip(values, known)):
            if value is None:
                if when is not None:
                    raise ValueError("Missing clinical value requires null known-at")
                mask.append(False)
                continue
            if type(value) not in {int, float} or not math.isfinite(value):
                raise ValueError("Clinical values must be finite or missing")
            stage_index(when)
            if i > 0 and value not in (0, 1):
                raise ValueError("HR/HER2/MP must be binary or missing")
            mask.append(when <= landmark)
        return np.asarray([v if v is not None else 0 for v in values], np.float32), np.asarray(mask, bool)

    def _fit_clinical(self):
        indices = self.by_split["train"]
        if not indices:
            raise ValueError("Train-only normalization must be supplied for an inference-only cohort")
        pairs = [self._clinical_values(self.patients[i], 0) for i in indices]
        values, masks = np.stack([v for v, _ in pairs]), np.stack([m for _, m in pairs])
        counts = masks.sum(0)
        mean = (values * masks).sum(0) / counts.clip(1)
        std = np.sqrt((((values - mean) * masks) ** 2).sum(0) / counts.clip(1))
        # Age is continuous; retain official binary coding for other features.
        mean[1:], std[1:] = 0, 1
        std = np.where(std < 1e-6, 1, std)
        state = {"schema": "raw_mri_normalization_v2", "fit_split": "train",
                "clinical_features": list(CLINICAL_FEATURES), "clinical_mean": mean.tolist(),
                "clinical_std": std.tolist(), "clinical_counts": counts.tolist(),
                "image_preprocessing": copy.deepcopy(self.image_preprocessing),
                "intensity": self._intensity_description()}
        if self.image_normalization is not None:
            state["image_normalization"] = copy.deepcopy(self.image_normalization)
        return state

    def _restore_normalization(self, state):
        if not isinstance(state, dict):
            raise ValueError("Normalization must use the declared training clinical coordinates")
        if state.get("schema") == "raw_mri_normalization_v1":
            if self.preprocessed_roi32:
                raise ValueError("Legacy raw normalization cannot load preprocessed_roi32 images")
            result = {key: copy.deepcopy(state.get(key)) for key in (
                "fit_split", "clinical_features", "clinical_mean", "clinical_std", "clinical_counts",
            )}
            result.update(schema="raw_mri_normalization_v2", image_preprocessing={"mode": "raw_t0_zscore"},
                          intensity=self._intensity_description())
            return result
        return copy.deepcopy(state)

    def _intensity_description(self):
        return ("Existing training-set ROI32 scale; image arrays passed through unchanged, zero background preserved"
                if self.preprocessed_roi32 else
                "Per-phase mean/std from that patient's T0 only; shared across visits")

    def _validate_normalization(self):
        st = self._normalization
        if (st.get("schema") != "raw_mri_normalization_v2" or st.get("fit_split") != "train"
                or st.get("clinical_features") != list(CLINICAL_FEATURES)):
            raise ValueError("Normalization must use the declared training clinical coordinates")
        mean, std = np.asarray(st.get("clinical_mean", [])), np.asarray(st.get("clinical_std", []))
        if (mean.shape != (4,) or std.shape != (4,) or not np.isfinite(mean).all()
                or not np.isfinite(std).all() or (std <= 0).any()):
            raise ValueError("Invalid train-only clinical scaling")
        counts = st.get("clinical_counts", [])
        if (not isinstance(counts, list) or len(counts) != 4
                or any(type(count) is not int or count < 0 for count in counts)):
            raise ValueError("Invalid training clinical feature counts")
        if st.get("image_preprocessing") != self.image_preprocessing:
            raise ValueError("Checkpoint image_preprocessing does not match the manifest")
        if st.get("image_normalization") != self.image_normalization:
            raise ValueError("Checkpoint image_normalization does not match the manifest")
        allowed = {"schema", "fit_split", "clinical_features", "clinical_mean", "clinical_std",
                   "clinical_counts", "image_preprocessing", "image_normalization", "intensity"}
        if set(st) - allowed:
            raise ValueError("Normalization v2 contains undeclared fields")

    def normalization_state(self):
        return copy.deepcopy(self._normalization)

    def allowed_landmarks(self, index):
        """Stages where a real MRI becomes available, with no invented visits."""
        return sorted({visit["available_at"] for visit in self.patients[index]["visits"] if visit is not None})

    def signature(self):
        """Bind the manifest structure and MRI filesystem identities for resume."""
        paths = sorted({str(self.resolve(path)) for patient in self.patients
                        for visit in patient["visits"] if visit is not None
                        for path in self._visit_paths(visit)})
        files = []
        for path in paths:
            stat = Path(path).stat()
            files.append({"path": path, "size": stat.st_size, "mtime_ns": stat.st_mtime_ns})
        result = {"manifest": manifest_without_legacy_checksums(self.manifest),
                "manifest_on_disk": manifest_without_legacy_checksums(read_json(self.path)),
                "image_shape": list(self.image_shape), "arrays": files}
        if self.image_cache is not None:
            result["image_cache"] = self.image_cache.signature()
        return result

    @staticmethod
    def _is_nifti(path):
        return str(path).endswith((".nii", ".nii.gz"))

    def _read_scan(self, visit, reference=None):
        paths = [self.resolve(p) for p in self._visit_paths(visit)]
        if "image" in visit:
            if paths[0].suffix == ".npz":
                if self.image_cache is not None:
                    array = self.image_cache.read(paths[0])
                else:
                    with np.load(paths[0], allow_pickle=False) as archive:
                        if "image" not in archive.files:
                            raise ValueError("ROI32 NPZ requires an image array")
                        array = archive["image"]
                if array.shape != ROI32_SHAPE or array.dtype != np.float32:
                    raise ValueError("ROI32 NPZ image must be float32 with exact shape [3,32,128,128]")
                meta = {"kind": "npz", "shape": array.shape[1:]}
            else:
                array = np.asarray(np.load(paths[0], allow_pickle=False), np.float32)
                if array.ndim != 4 or array.shape[0] != 3:
                    raise ValueError("Stacked raw MRI must have shape [3,D,H,W]")
                meta = {"kind": "npy", "shape": array.shape[1:]}
        elif all(self._is_nifti(p) for p in paths):
            try:
                import nibabel as nib
                from nibabel.processing import resample_from_to
            except ImportError as exc:
                raise ImportError("NIfTI MRI loading requires nibabel/scipy data dependencies") from exc
            scans = [nib.as_closest_canonical(nib.load(str(p))) for p in paths]
            if any(len(scan.shape) != 3 for scan in scans):
                raise ValueError("Each NIfTI phase must be a three-dimensional volume")
            if reference is not None and reference["kind"] != "nifti":
                raise ValueError("Do not mix NIfTI and array geometry across visits")
            shape = reference["shape"] if reference is not None else scans[0].shape
            affine = reference["affine"] if reference is not None else scans[0].affine
            arrays = []
            for scan in scans:
                if scan.shape != shape or not np.allclose(scan.affine, affine, atol=1e-5):
                    scan = resample_from_to(scan, (shape, affine), order=1)
                arrays.append(np.asarray(scan.get_fdata(dtype=np.float32)).transpose(2, 1, 0))
            array = np.stack(arrays)
            meta = {"kind": "nifti", "shape": tuple(shape), "affine": np.asarray(affine)}
        elif all(str(p).endswith(".npy") for p in paths):
            arrays = [np.asarray(np.load(p, allow_pickle=False), np.float32) for p in paths]
            if any(v.ndim != 3 or v.shape != arrays[0].shape for v in arrays):
                raise ValueError("Array phases require matching three-dimensional source grids")
            array = np.stack(arrays)
            meta = {"kind": "npy", "shape": array.shape[1:]}
        else:
            raise ValueError("Phase files must all be .npy or all be NIfTI; mixed geometry is unsupported")
        if self.image_cache is None and not np.isfinite(array).all():
            raise ValueError("MRI must contain finite intensities")
        if reference is not None:
            if meta["kind"] != reference["kind"] or (meta["kind"] in {"npy", "npz"} and meta["shape"] != reference["shape"]):
                raise ValueError("Array visits must already share the T0 source grid; do not resize independent ROIs")
        return array, meta

    def _resize_normalized(self, array, mean, std):
        if self.preprocessed_roi32:
            return torch.from_numpy(np.ascontiguousarray(array))
        value = torch.from_numpy(np.ascontiguousarray((array - mean) / std)).float()[None]
        return F.interpolate(value, self.image_shape, mode="trilinear", align_corners=False)[0]

    def batch(self, tasks, supervised=True, *, pin_memory=False, future_supervision=True):
        if not tasks:
            raise ValueError("Empty raw MRI batch")
        if type(pin_memory) is not bool:
            raise ValueError("pin_memory must be boolean")
        if type(future_supervision) is not bool:
            raise ValueError("future_supervision must be boolean")
        b = len(tasks)
        images = torch.zeros((b, 4, 3, *self.image_shape), pin_memory=pin_memory)
        observed = torch.zeros(b, 4, dtype=torch.bool)
        queries = torch.zeros_like(observed)
        clinical = torch.zeros(b, 4)
        cmask = torch.zeros(b, 4, dtype=torch.bool)
        arm_ids, arm_masks, arm_known, landmarks = [], [], [], []
        sources = []
        for row, (index, landmark) in enumerate(tasks):
            landmark = stage_index(landmark)
            if type(index) is not int or not 0 <= index < len(self.patients):
                raise ValueError("Invalid patient index")
            patient = self.patients[index]
            baseline, reference = self._read_scan(patient["visits"][0])
            mean, std = None, None
            if not self.preprocessed_roi32:
                mean = baseline.mean(axis=(1, 2, 3), keepdims=True)
                std = baseline.std(axis=(1, 2, 3), keepdims=True)
                std = np.where(std < 1e-6, 1, std)
            sources.append((mean, std, reference))
            for stage, visit in enumerate(patient["visits"]):
                if stage <= landmark and visit is not None and visit["available_at"] <= landmark:
                    scan = baseline if stage == 0 else self._read_scan(visit, reference)[0]
                    images[row, stage] = self._resize_normalized(scan, mean, std)
                    observed[row, stage] = True
            queries[row] = torch.arange(4) > landmark
            values, mask = self._clinical_values(patient, landmark)
            standardized = (values - np.asarray(self._normalization["clinical_mean"])) / np.asarray(self._normalization["clinical_std"])
            clinical[row] = torch.from_numpy(np.where(mask, standardized, 0).astype(np.float32))
            cmask[row] = torch.from_numpy(mask)
            identifier, available, when = visible_arm(patient["arm"], landmark)
            arm_ids.append(identifier); arm_masks.append(available); arm_known.append(when); landmarks.append(landmark)
        inp = RawMRIInput(images, observed, torch.tensor(landmarks, dtype=torch.long), clinical, cmask,
                          torch.tensor(arm_ids, dtype=torch.long), torch.tensor(arm_masks, dtype=torch.bool),
                          torch.tensor(arm_known, dtype=torch.long), queries).validate()
        if pin_memory:
            inp = inp.pin_memory()
        if not supervised:
            return inp, None
        future = torch.zeros(images.shape, dtype=images.dtype, pin_memory=pin_memory)
        fmask = torch.zeros_like(observed)
        labels, lmask = torch.zeros(b), torch.zeros(b, dtype=torch.bool)
        for row, (index, landmark) in enumerate(tasks):
            patient = self.patients[index]
            mean, std, reference = sources[row]
            if future_supervision:
                for stage, visit in enumerate(patient["visits"]):
                    if stage > landmark and visit is not None:
                        future[row, stage] = self._resize_normalized(self._read_scan(visit, reference)[0], mean, std)
                        fmask[row, stage] = True
            target = patient.get("target", {})
            label = target.get("pcr")
            if label is not None:
                if type(label) not in {int, float} or label not in (0, 1):
                    raise ValueError("Final pCR supervision must be binary or null")
                if not isinstance(target.get("label_source"), str) or not target["label_source"]:
                    raise ValueError("Final pCR supervision requires label provenance")
                labels[row], lmask[row] = label, True
        sup = RawMRISupervision(future, fmask, labels, lmask).validate(inp)
        return inp, sup.pin_memory() if pin_memory else sup


def make_raw_synthetic(output, n_train=6, n_val=2, image_shape=(8, 16, 16), seed=123):
    """Write fake-patient MRI fixtures for integration, never clinical evidence."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if n_train < 1 or n_val < 1:
        raise ValueError("Synthetic train and validation cohorts must be nonempty")
    rng = np.random.default_rng(seed)
    patients = []
    grid = np.indices(image_shape).astype(np.float32)
    lesion = np.exp(-sum(((grid[i] - (image_shape[i] - 1) / 2) / max(image_shape[i] / 5, 1)) ** 2 for i in range(3)))
    for index in range(n_train + n_val):
        label = index % 2
        visits = []
        for stage in range(4):
            path = output / f"synthetic_{index}_T{stage}.npy"
            scan = rng.normal(0, .1, (3, *image_shape)).astype(np.float32)
            scan += (1 + label) * (1 - .2 * stage * label) * lesion[None]
            scan += label * .3
            np.save(path, scan)
            visits.append({"stage": stage, "available_at": stage, "grid_id": f"fake_grid_{index}", "image": path.name})
        patients.append({"patient_key": f"synthetic_{index}", "split": "train" if index < n_train else "val",
                         "clinical": {"values": [35 + index, label, 1 - label, label], "known_at": [0] * 4},
                         "arm": make_arm_record(OFFICIAL_ARMS[index % 13], 0, "assigned_arm_scenario"),
                         "geometry": {"source_stage": 0, "source_only": True, "grid_id": f"fake_grid_{index}",
                                      "orientation": "RAS", "roi_source": "T0", "resampling": "source_grid"},
                         "visits": visits, "target": {"pcr": label, "label_source": "synthetic_fixture"}})
    manifest = {"schema": SCHEMA, "time_basis": "stage_index", "canonical_stages": [0, 1, 2, 3],
                "phase_order": PHASES, "clinical_features": list(CLINICAL_FEATURES), "synthetic": True,
                "provenance": {"meaning": "Synthetic engineering fixtures, not patient data"}, "patients": patients}
    path = output / "manifest.json"
    write_json(path, manifest)
    return path
