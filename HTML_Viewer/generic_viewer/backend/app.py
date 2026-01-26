#!/usr/bin/env python3
"""
Flask backend for the T2/FF viewer.

Serves slice PNGs and patient metadata from datasets/patient_data_regression_setup.
"""

from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk
from scipy.ndimage import binary_erosion, generate_binary_structure
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image
import yaml

DATA_DIR = Path(os.environ.get("HTML_VIEWER_DATA_DIR", "datasets/patient_data_regression_setup"))
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config_pixel_level" / "train_config.yaml"
CONFIG_PATH = Path(os.environ.get("HTML_VIEWER_CONFIG", DEFAULT_CONFIG_PATH))
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")
app.logger.setLevel(os.environ.get("HTML_VIEWER_LOG_LEVEL", "INFO"))

_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
_CACHE_ORDER: list[Tuple[str, str]] = []
_CACHE_MAX = 8
_FF_MEDIAN_CACHE: Dict[Tuple[str, str, int], float] = {}
_SAMPLER_CACHE: Dict[Tuple[str, str, Tuple[float, ...], Tuple[float, ...], int], dict] = {}
_TRAINING_CACHE: Dict[str, dict] = {}
_TRAINING_SAMPLER_CACHE: Dict[Tuple[str, str], dict] = {}
_TRAINING_VOLUME_CACHE: Dict[Tuple[str, str, str], np.ndarray] = {}


def _cache_get(key: Tuple[str, str]) -> np.ndarray | None:
    return _CACHE.get(key)


def _cache_set(key: Tuple[str, str], value: np.ndarray) -> None:
    if key in _CACHE:
        return
    _CACHE[key] = value
    _CACHE_ORDER.append(key)
    if len(_CACHE_ORDER) > _CACHE_MAX:
        old_key = _CACHE_ORDER.pop(0)
        _CACHE.pop(old_key, None)


def _load_training_config() -> dict:
    cached = _TRAINING_CACHE.get("config")
    if cached is not None:
        return cached
    if not CONFIG_PATH.exists():
        app.logger.warning("Training config not found: %s", CONFIG_PATH)
        _TRAINING_CACHE["config"] = {}
        return {}
    with open(CONFIG_PATH, "r") as f:
        config = yaml.safe_load(f)
    _TRAINING_CACHE["config"] = config
    app.logger.info("Loaded training config: %s", CONFIG_PATH)
    return config


def _training_data_dir() -> Path:
    cached = _TRAINING_CACHE.get("data_dir")
    if cached is not None:
        return cached
    config = _load_training_config()
    data_dir = Path(config.get("data", {}).get("data_dir", DATA_DIR))
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    _TRAINING_CACHE["data_dir"] = data_dir
    app.logger.info("Using training data dir: %s", data_dir)
    return data_dir


def _training_data_cfg() -> dict:
    return _load_training_config().get("data", {})


def _training_fat_sampling_cfg() -> dict:
    return _training_data_cfg().get("fat_sampling", {})


def _find_file(patient_id: str, filename: str, data_dir: Path | None = None) -> Path | None:
    root = data_dir or DATA_DIR
    candidate = root / patient_id / filename
    if candidate.exists():
        return candidate
    if filename.endswith(".nii.gz"):
        alt = root / patient_id / filename.replace(".nii.gz", ".nii")
        if alt.exists():
            return alt
    return None


def _erode_mask(volume: np.ndarray, radius: int = 3) -> np.ndarray:
    image = sitk.GetImageFromArray((volume > 0).astype(np.uint8))
    eroded = sitk.BinaryErode(image, [radius, radius, radius])
    return sitk.GetArrayFromImage(eroded).astype(np.uint8)


def _erode_mask_scipy(volume: np.ndarray, iterations: int) -> np.ndarray:
    if iterations <= 0:
        return (volume > 0).astype(np.uint8)
    mask = (volume > 0).astype(np.uint8)
    struct = generate_binary_structure(3, 1)
    return binary_erosion(mask, structure=struct, iterations=iterations).astype(np.uint8)


def _load_array(path: Path) -> np.ndarray:
    image = sitk.ReadImage(str(path))
    return sitk.GetArrayFromImage(image).astype(np.float32)


def _load_volume(patient_id: str, kind: str, data_dir: Path | None = None) -> np.ndarray:
    key = (patient_id, kind)
    cached = _cache_get(key)
    if cached is not None:
        return cached

    if kind == "t2":
        filename = f"{patient_id}_t2_original.nii.gz"
    elif kind == "t2_seg":
        filename = f"{patient_id}_t2_original_segmentation.nii.gz"
    elif kind == "ff":
        filename = f"{patient_id}_ff_normalized.nii.gz"
    elif kind == "ff_seg":
        filename = f"{patient_id}_segmentation.nii.gz"
    elif kind == "ff_seg_eroded":
        filename = f"{patient_id}_segmentation.nii.gz"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    path = _find_file(patient_id, filename, data_dir=data_dir)
    if path is None:
        raise FileNotFoundError(f"Missing file for {patient_id}: {filename}")

    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image).astype(np.float32)  # (z, y, x)
    if kind == "ff_seg_eroded":
        volume = _erode_mask(volume)
    _cache_set(key, volume)
    return volume


def _normalize_t2(slice_2d: np.ndarray) -> np.ndarray:
    p1, p99 = np.percentile(slice_2d, [1, 99])
    if p99 <= p1:
        return np.zeros_like(slice_2d, dtype=np.uint8)
    scaled = (slice_2d - p1) / (p99 - p1)
    scaled = np.clip(scaled, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _normalize_ff(slice_2d: np.ndarray) -> np.ndarray:
    if slice_2d.max() > 1.5:
        slice_2d = slice_2d / 100.0
    scaled = np.clip(slice_2d, 0.0, 1.0)
    return (scaled * 255).astype(np.uint8)


def _normalize_ff_volume(volume: np.ndarray) -> np.ndarray:
    if volume.max() > 1.5:
        volume = volume / 100.0
    return np.clip(volume, 0.0, 1.0)


def _resolve_data_path(
    patient_id: str,
    suffix: str,
    data_dir: Path,
    data_cfg: dict,
    subdir_key: str | None = None,
) -> Path | None:
    use_subdirs = bool(data_cfg.get("use_subdirs", False))
    use_patient_subdirs = bool(data_cfg.get("use_patient_subdirs", False))
    filename = f"{patient_id}{suffix}" if not use_subdirs else patient_id
    root = data_dir
    if use_subdirs and subdir_key:
        root = data_dir / data_cfg.get(subdir_key, "")
    if use_patient_subdirs:
        root = root / patient_id
    path_gz = root / f"{filename}.nii.gz"
    path_nii = root / f"{filename}.nii"
    if path_gz.exists():
        return path_gz
    if path_nii.exists():
        return path_nii
    return None


def _mask_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    mask = (slice_2d > 0).astype(np.uint8) * 255
    return mask


def _load_training_volume(patient_id: str, kind: str) -> np.ndarray:
    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    cache_key = (str(data_dir), patient_id, kind)
    cached = _TRAINING_VOLUME_CACHE.get(cache_key)
    if cached is not None:
        return cached

    if kind == "t2":
        suffix = data_cfg.get("t2_suffix", "_t2_aligned")
        subdir = "t2_subdir"
    elif kind == "ff":
        suffix = data_cfg.get("ff_suffix", "_ff_normalized")
        subdir = "ff_subdir"
    elif kind == "ff_seg":
        suffix = data_cfg.get("mask_suffix", "_segmentation")
        subdir = "mask_subdir"
    elif kind == "t2_seg":
        suffix = data_cfg.get("input_mask_suffix") or data_cfg.get("mask_suffix", "_segmentation")
        subdir = "t2_subdir"
    elif kind == "ff_seg_eroded":
        suffix = data_cfg.get("mask_suffix", "_segmentation")
        subdir = "mask_subdir"
    else:
        raise ValueError(f"Unknown kind: {kind}")

    path = _resolve_data_path(patient_id, suffix, data_dir, data_cfg, subdir)
    if path is None:
        raise FileNotFoundError(f"Missing file for {patient_id}: {suffix}")

    volume = _load_array(path)
    if kind == "ff_seg_eroded":
        volume = _erode_mask(volume)
    _TRAINING_VOLUME_CACHE[cache_key] = volume
    return volume


def _list_patient_ids(data_dir: Path | None = None) -> list[str]:
    root = data_dir or DATA_DIR
    if not root.exists():
        return []
    patients = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        pid = p.name
        ff_path = _find_file(pid, f"{pid}_ff_normalized.nii.gz", data_dir=root)
        mask_path = _find_file(pid, f"{pid}_segmentation.nii.gz", data_dir=root)
        if ff_path and mask_path:
            patients.append(pid)
    return patients


def _load_splits(data_dir: Path | None = None, split_filename: str | None = None) -> dict:
    root = data_dir or DATA_DIR
    if split_filename:
        candidate = root / split_filename
        if candidate.exists():
            with open(candidate, "r") as f:
                return json.load(f)
    splits_path = root / "data_splits.json"
    if not splits_path.exists():
        return {}
    with open(splits_path, "r") as f:
        return json.load(f)


def _get_split_patient_ids(split: str, data_dir: Path | None = None) -> list[str]:
    root = data_dir or DATA_DIR
    all_ids = _list_patient_ids(root)
    if split == "all":
        return all_ids
    splits = _load_splits(root)
    split_ids = splits.get(split)
    if not split_ids:
        return all_ids
    available = set(all_ids)
    return [pid for pid in split_ids if pid in available]


def _get_training_split_ids(split: str, data_dir: Path) -> list[str]:
    data_cfg = _training_data_cfg()
    fat_split_cfg = data_cfg.get("fat_split", {})
    split_file = None
    if fat_split_cfg.get("enabled", False):
        split_file = fat_split_cfg.get("split_file", "data_splits_fatassigned.json")
    splits = _load_splits(data_dir, split_filename=split_file)
    if not splits:
        return []
    if split == "all":
        return splits.get("train", []) + splits.get("val", []) + splits.get("test", [])
    return splits.get(split, [])


def _load_training_splits(data_dir: Path) -> dict:
    data_cfg = _training_data_cfg()
    fat_split_cfg = data_cfg.get("fat_split", {})
    split_file = None
    if fat_split_cfg.get("enabled", False):
        split_file = fat_split_cfg.get("split_file", "data_splits_fatassigned.json")
    splits = _load_splits(data_dir, split_filename=split_file)
    if splits:
        app.logger.info(
            "Loaded split file %s (train=%d, val=%d, test=%d, median_ff=%s).",
            split_file or "data_splits.json",
            len(splits.get("train", [])),
            len(splits.get("val", [])),
            len(splits.get("test", [])),
            "yes" if "median_ff" in splits else "no",
        )
    else:
        app.logger.warning(
            "No split file found at %s.",
            (data_dir / (split_file or "data_splits.json")),
        )
    return splits


def _discover_training_patient_ids(data_dir: Path, data_cfg: dict) -> list[str]:
    use_subdirs = bool(data_cfg.get("use_subdirs", False))
    use_patient_subdirs = bool(data_cfg.get("use_patient_subdirs", False))
    t2_suffix = data_cfg.get("t2_suffix", "_t2_aligned")
    t2_subdir = data_cfg.get("t2_subdir", "t2_images")

    if use_subdirs:
        t2_dir = data_dir / t2_subdir
        t2_files = sorted(list(t2_dir.glob("*.nii.gz")) + list(t2_dir.glob("*.nii")))
        app.logger.info("Discovered %d T2 files in %s", len(t2_files), t2_dir)
        return [f.stem.replace(".nii", "") for f in t2_files]
    if use_patient_subdirs:
        patient_ids = []
        for subdir in sorted(data_dir.iterdir()):
            if subdir.is_dir():
                t2_file = list(subdir.glob(f"*{t2_suffix}.nii.gz"))
                if t2_file:
                    patient_ids.append(subdir.name)
        app.logger.info("Discovered %d patient subdirs in %s", len(patient_ids), data_dir)
        return patient_ids

    t2_files = sorted(
        list(data_dir.glob(f"*{t2_suffix}.nii.gz")) + list(data_dir.glob(f"*{t2_suffix}.nii"))
    )
    app.logger.info("Discovered %d flat T2 files in %s", len(t2_files), data_dir)
    patient_ids = []
    for f in t2_files:
        filename = f.stem.replace(".nii", "")
        if filename.endswith(t2_suffix):
            patient_ids.append(filename[: -len(t2_suffix)])
    return patient_ids


def _validate_patient_ids(
    patient_ids: list[str],
    data_dir: Path,
    t2_suffix: str,
    ff_suffix: str,
    mask_suffix: str,
    min_size: int = 8,
) -> list[str]:
    data_cfg = _training_data_cfg()
    valid = []
    missing = 0
    too_small = 0
    for pid in patient_ids:
        t2_path = _resolve_data_path(pid, t2_suffix, data_dir, data_cfg, "t2_subdir")
        ff_path = _resolve_data_path(pid, ff_suffix, data_dir, data_cfg, "ff_subdir")
        mask_path = _resolve_data_path(pid, mask_suffix, data_dir, data_cfg, "mask_subdir")
        if not t2_path or not ff_path or not mask_path:
            missing += 1
            continue
        try:
            image = sitk.ReadImage(str(t2_path))
            size = image.GetSize()
            if len(size) < 3 or any(dim < min_size for dim in size):
                too_small += 1
                continue
        except Exception:
            continue
        valid.append(pid)
    app.logger.info(
        "Validated %d/%d ids (missing=%d, too_small=%d).",
        len(valid),
        len(patient_ids),
        missing,
        too_small,
    )
    return valid


def _bin_index(value: float, bin_edges: list[float]) -> int:
    for i in range(len(bin_edges) - 1):
        if bin_edges[i] <= value < bin_edges[i + 1]:
            return i
    return len(bin_edges) - 2


def _compute_median_ff(patient_id: str, erosion_iters: int, data_dir: Path | None = None) -> float:
    cache_key = (str(data_dir or DATA_DIR), patient_id, erosion_iters)
    cached = _FF_MEDIAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ff = _load_volume(patient_id, "ff", data_dir=data_dir)
    mask = _load_volume(patient_id, "ff_seg", data_dir=data_dir)
    ff = _normalize_ff_volume(ff)
    mask = _erode_mask_scipy(mask, erosion_iters)
    masked_values = ff[mask > 0]
    if masked_values.size == 0:
        median = float("nan")
    else:
        median = float(np.median(masked_values))
    _FF_MEDIAN_CACHE[cache_key] = median
    return median


def _compute_median_ff_training(
    patient_id: str,
    data_dir: Path,
    data_cfg: dict,
) -> float:
    erosion_iters = int(data_cfg.get("mask_erosion", 0))
    cache_key = (str(data_dir), patient_id, erosion_iters, "training")
    cached = _FF_MEDIAN_CACHE.get(cache_key)
    if cached is not None:
        return cached
    ff_path = _resolve_data_path(
        patient_id, data_cfg.get("ff_suffix", "_ff_normalized"), data_dir, data_cfg, "ff_subdir"
    )
    mask_path = _resolve_data_path(
        patient_id, data_cfg.get("mask_suffix", "_segmentation"), data_dir, data_cfg, "mask_subdir"
    )
    if not ff_path or not mask_path:
        return float("nan")
    ff = _normalize_ff_volume(_load_array(ff_path))
    mask = _erode_mask_scipy(_load_array(mask_path), erosion_iters)
    masked_values = ff[mask > 0]
    if masked_values.size == 0:
        median = float("nan")
    else:
        median = float(np.median(masked_values))
    _FF_MEDIAN_CACHE[cache_key] = median
    return median


def _compute_sampling(
    patient_ids: list[str],
    bin_edges: list[float],
    bin_probs: list[float],
    erosion_iters: int,
    data_dir: Path | None = None,
) -> tuple[list[str], list[float], list[float], list[int], list[float]]:
    if len(bin_edges) != len(bin_probs) + 1:
        raise ValueError("bin_edges must be one longer than bin_probs")
    prob_sum = float(np.sum(bin_probs))
    if prob_sum <= 0:
        raise ValueError("bin_probs must sum to > 0")
    bin_probs = [p / prob_sum for p in bin_probs]

    filtered_ids = []
    medians = []
    bins = []
    for pid in patient_ids:
        median = _compute_median_ff(pid, erosion_iters, data_dir=data_dir)
        if not np.isfinite(median):
            continue
        filtered_ids.append(pid)
        medians.append(median)
        bins.append(_bin_index(median, bin_edges))

    counts = [0] * len(bin_probs)
    for b in bins:
        counts[b] += 1

    bin_weights = []
    for idx, count in enumerate(counts):
        if count == 0:
            bin_weights.append(0.0)
        else:
            bin_weights.append(bin_probs[idx] / count)

    weights = [bin_weights[b] for b in bins]
    mean_weight = float(np.mean([w for w in weights if w > 0])) if weights else 1.0
    if mean_weight > 0:
        weights = [w / mean_weight for w in weights]

    return filtered_ids, medians, weights, bins, bin_weights


def _compute_bins_and_weights(
    medians: list[float],
    bin_edges: list[float],
    bin_probs: list[float],
) -> tuple[list[int], list[float], list[float]]:
    if len(bin_edges) != len(bin_probs) + 1:
        raise ValueError("bin_edges must be one longer than bin_probs")
    prob_sum = float(np.sum(bin_probs))
    if prob_sum <= 0:
        raise ValueError("bin_probs must sum to > 0")
    bin_probs = [p / prob_sum for p in bin_probs]

    bins = [_bin_index(m, bin_edges) for m in medians]
    counts = [0] * len(bin_probs)
    for b in bins:
        counts[b] += 1

    bin_weights = []
    for idx, count in enumerate(counts):
        if count == 0:
            bin_weights.append(0.0)
        else:
            bin_weights.append(bin_probs[idx] / count)

    weights = [bin_weights[b] for b in bins]
    mean_weight = float(np.mean([w for w in weights if w > 0])) if weights else 1.0
    if mean_weight > 0:
        weights = [w / mean_weight for w in weights]

    return bins, bin_weights, weights


def _sampler_key(
    split: str,
    bin_edges: list[float],
    bin_probs: list[float],
    erosion_iters: int,
    data_dir: Path,
) -> tuple:
    return (str(data_dir), split, tuple(bin_edges), tuple(bin_probs), erosion_iters)


def _build_sampler_state(
    split: str,
    bin_edges: list[float],
    bin_probs: list[float],
    erosion_iters: int,
    data_dir: Path,
    data_cfg: dict,
) -> dict:
    patient_ids = _get_training_split_ids(split, data_dir=data_dir)
    patient_ids = _validate_patient_ids(
        patient_ids,
        data_dir=data_dir,
        t2_suffix=data_cfg.get("t2_suffix", "_t2_aligned"),
        ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
        mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
    )
    patient_ids, medians, weights, bins, bin_weights = _compute_sampling(
        patient_ids, bin_edges, bin_probs, erosion_iters, data_dir=data_dir
    )
    weight_sum = float(np.sum(weights))
    probs = None
    if weight_sum > 0:
        probs = [w / weight_sum for w in weights]
    order = []
    if len(weights) > 0:
        order = list(np.random.choice(len(weights), size=len(weights), replace=True, p=probs))
    return {
        "patient_ids": patient_ids,
        "medians": medians,
        "weights": weights,
        "bins": bins,
        "bin_weights": bin_weights,
        "probs": probs,
        "order": order,
        "cursor": 0,
    }


def _build_training_sampler_state(split: str) -> dict:
    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    fat_cfg = _training_fat_sampling_cfg()
    enabled = bool(fat_cfg.get("enabled", False))

    app.logger.info("Building training sampler for split %s", split)
    split_ids = _get_training_split_ids(split, data_dir=data_dir)
    if not split_ids:
        split_ids = _discover_training_patient_ids(data_dir, data_cfg)
    split_ids = _validate_patient_ids(
        split_ids,
        data_dir=data_dir,
        t2_suffix=data_cfg.get("t2_suffix", "_t2_aligned"),
        ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
        mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
    )

    medians = []
    for pid in split_ids:
        median = _compute_median_ff_training(pid, data_dir, data_cfg)
        if np.isfinite(median):
            medians.append(median)
        else:
            medians.append(float("nan"))

    valid_indices = [i for i, m in enumerate(medians) if np.isfinite(m)]
    valid_ids = [split_ids[i] for i in valid_indices]
    valid_medians = [medians[i] for i in valid_indices]

    if not enabled:
        app.logger.info("fat_sampling disabled; using uniform shuffle for split %s", split)
        order = list(np.random.permutation(len(valid_ids))) if valid_ids else []
        return {
            "patient_ids": valid_ids,
            "medians": valid_medians,
            "bins": [],
            "bin_weights": [],
            "probs": None,
            "order": order,
            "cursor": 0,
            "enabled": False,
        }

    bin_edges = fat_cfg.get("bin_edges", [0.0, 0.1, 0.2, 1.0])
    bin_probs = fat_cfg.get("bin_probs", [0.2, 0.3, 0.5])
    bins, bin_weights, weights = _compute_bins_and_weights(valid_medians, bin_edges, bin_probs)
    app.logger.info(
        "fat_sampling enabled; split %s bins=%s probs=%s",
        split,
        bin_edges,
        bin_probs,
    )
    weight_sum = float(np.sum(weights))
    probs = None
    if weight_sum > 0:
        probs = [w / weight_sum for w in weights]
    order = list(np.random.choice(len(valid_ids), size=len(valid_ids), replace=True, p=probs))
    return {
        "patient_ids": valid_ids,
        "medians": valid_medians,
        "bins": bins,
        "bin_weights": bin_weights,
        "bin_edges": bin_edges,
        "bin_probs": bin_probs,
        "probs": probs,
        "order": order,
        "cursor": 0,
        "enabled": True,
    }


@app.route("/api/patients")
def list_patients():
    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    if not data_dir.exists():
        return jsonify({"patients": [], "error": f"Missing data dir: {data_dir}"}), 200
    patients = []
    split_ids = _get_training_split_ids("all", data_dir=data_dir)
    if not split_ids:
        split_ids = _discover_training_patient_ids(data_dir, data_cfg)
    for pid in split_ids:
        t2_path = _resolve_data_path(pid, data_cfg.get("t2_suffix", "_t2_aligned"), data_dir, data_cfg, "t2_subdir")
        ff_path = _resolve_data_path(pid, data_cfg.get("ff_suffix", "_ff_normalized"), data_dir, data_cfg, "ff_subdir")
        if t2_path and ff_path:
            patients.append(pid)
    return jsonify({"patients": patients})


@app.route("/api/patients_with_ff")
def list_patients_with_ff():
    data_dir = _training_data_dir()
    if not data_dir.exists():
        return jsonify({"patients": [], "error": f"Missing data dir: {data_dir}"}), 200
    split = request.args.get("split", "all")
    data_cfg = _training_data_cfg()
    erosion_iters = int(data_cfg.get("mask_erosion", request.args.get("erosion_iters", 3)))
    splits = _load_training_splits(data_dir)
    patient_ids = _get_training_split_ids(split, data_dir=data_dir)
    if not patient_ids:
        patient_ids = _discover_training_patient_ids(data_dir, data_cfg)
    patient_ids = _validate_patient_ids(
        patient_ids,
        data_dir=data_dir,
        t2_suffix=data_cfg.get("t2_suffix", "_t2_aligned"),
        ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
        mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
    )
    patients = []
    median_map = splits.get("median_ff", {}) if splits else {}
    for pid in patient_ids:
        if pid in median_map:
            median_ff = float(median_map[pid])
        else:
            median_ff = _compute_median_ff_training(pid, data_dir, data_cfg)
        patients.append(
            {
                "patient_id": pid,
                "median_ff": median_ff,
                "median_ff_percent": median_ff * 100.0 if np.isfinite(median_ff) else None,
            }
        )
    return jsonify({"patients": patients})


@app.route("/api/sampling/summary", methods=["POST"])
def sampling_summary():
    payload = request.get_json(silent=True) or {}
    bin_edges = payload.get("bin_edges", [0.0, 0.1, 0.2, 1.0])
    bin_probs = payload.get("bin_probs", [0.2, 0.3, 0.5])
    split = payload.get("split", "train")
    erosion_iters = int(payload.get("erosion_iters", 3))

    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    key = _sampler_key(split, bin_edges, bin_probs, erosion_iters, data_dir)
    sampler = _SAMPLER_CACHE.get(key)
    if sampler is None:
        sampler = _build_sampler_state(split, bin_edges, bin_probs, erosion_iters, data_dir, data_cfg)
        _SAMPLER_CACHE[key] = sampler

    medians = sampler["medians"]
    bins = sampler["bins"]
    bin_weights = sampler["bin_weights"]

    counts = [0] * len(bin_probs)
    for b in bins:
        counts[b] += 1

    bin_ranges = [
        f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)
    ]

    return jsonify(
        {
            "split": split,
            "total": len(medians),
            "bin_edges": bin_edges,
            "bin_probs": bin_probs,
            "bins": [
                {
                    "index": i,
                    "range": bin_ranges[i],
                    "count": counts[i],
                    "prob": bin_probs[i],
                    "weight": bin_weights[i],
                }
                for i in range(len(bin_probs))
            ],
        }
    )


@app.route("/api/sampling/sample", methods=["POST"])
def sampling_sample():
    payload = request.get_json(silent=True) or {}
    bin_edges = payload.get("bin_edges", [0.0, 0.1, 0.2, 1.0])
    bin_probs = payload.get("bin_probs", [0.2, 0.3, 0.5])
    split = payload.get("split", "train")
    erosion_iters = int(payload.get("erosion_iters", 3))

    reset = bool(payload.get("reset", False))
    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    key = _sampler_key(split, bin_edges, bin_probs, erosion_iters, data_dir)
    sampler = _SAMPLER_CACHE.get(key)
    if sampler is None or reset:
        sampler = _build_sampler_state(split, bin_edges, bin_probs, erosion_iters, data_dir, data_cfg)
        _SAMPLER_CACHE[key] = sampler

    patient_ids = sampler["patient_ids"]
    medians = sampler["medians"]
    bins = sampler["bins"]
    bin_weights = sampler["bin_weights"]
    order = sampler["order"]
    probs = sampler.get("probs")
    if not medians or not order:
        return jsonify({"error": "No valid patients for sampling"}), 400

    if sampler["cursor"] >= len(order):
        sampler["cursor"] = 0
        sampler["order"] = list(
            np.random.choice(len(medians), size=len(medians), replace=True, p=probs)
        )
        order = sampler["order"]

    idx = int(order[sampler["cursor"]])
    sampler["cursor"] += 1
    median = medians[idx]
    bin_idx = bins[idx]
    patient_id = patient_ids[idx]

    return jsonify(
        {
            "patient_id": patient_id,
            "median_ff": median,
            "median_ff_percent": median * 100.0,
            "bin_index": bin_idx,
            "bin_range": f"{bin_edges[bin_idx]:.2f}-{bin_edges[bin_idx + 1]:.2f}",
            "bin_prob": bin_probs[bin_idx],
            "bin_weight": bin_weights[bin_idx],
            "split": split,
        }
    )


@app.route("/api/training_sampler/summary")
def training_sampler_summary():
    split = request.args.get("split", "train")
    key = (str(_training_data_dir()), split)
    sampler = _TRAINING_SAMPLER_CACHE.get(key)
    if sampler is None:
        sampler = _build_training_sampler_state(split)
        _TRAINING_SAMPLER_CACHE[key] = sampler

    if not sampler["enabled"]:
        return jsonify(
            {
                "split": split,
                "enabled": False,
                "total": len(sampler["patient_ids"]),
                "message": "fat_sampling disabled; uniform shuffle per epoch",
            }
        )

    bin_edges = sampler.get("bin_edges", [])
    bin_probs = sampler.get("bin_probs", [])
    bins = sampler.get("bins", [])
    bin_weights = sampler.get("bin_weights", [])

    counts = [0] * len(bin_probs)
    for b in bins:
        counts[b] += 1

    bin_ranges = [
        f"{bin_edges[i]:.2f}-{bin_edges[i + 1]:.2f}" for i in range(len(bin_edges) - 1)
    ]

    return jsonify(
        {
            "split": split,
            "enabled": True,
            "total": len(sampler["patient_ids"]),
            "bin_edges": bin_edges,
            "bin_probs": bin_probs,
            "bins": [
                {
                    "index": i,
                    "range": bin_ranges[i],
                    "count": counts[i],
                    "prob": bin_probs[i],
                    "weight": bin_weights[i],
                }
                for i in range(len(bin_probs))
            ],
        }
    )


@app.route("/api/training_sampler/sample")
def training_sampler_sample():
    split = request.args.get("split", "train")
    reset = bool(request.args.get("reset", False))
    key = (str(_training_data_dir()), split)
    sampler = _TRAINING_SAMPLER_CACHE.get(key)
    if sampler is None or reset:
        sampler = _build_training_sampler_state(split)
        _TRAINING_SAMPLER_CACHE[key] = sampler

    patient_ids = sampler["patient_ids"]
    medians = sampler["medians"]
    order = sampler["order"]
    if not patient_ids or not order:
        return jsonify({"error": "No valid patients for sampling"}), 400

    if sampler["cursor"] >= len(order):
        sampler["cursor"] = 0
        if sampler["enabled"]:
            probs = sampler.get("probs")
            sampler["order"] = list(
                np.random.choice(len(patient_ids), size=len(patient_ids), replace=True, p=probs)
            )
        else:
            sampler["order"] = list(np.random.permutation(len(patient_ids)))
        order = sampler["order"]

    idx = int(order[sampler["cursor"]])
    sampler["cursor"] += 1
    median = medians[idx]

    return jsonify(
        {
            "patient_id": patient_ids[idx],
            "median_ff": median,
            "median_ff_percent": median * 100.0,
            "split": split,
            "enabled": sampler["enabled"],
        }
    )


@app.route("/api/info/<patient_id>")
def patient_info(patient_id: str):
    try:
        t2 = _load_training_volume(patient_id, "t2")
        ff = _load_training_volume(patient_id, "ff")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404

    data_dir = _training_data_dir()
    data_cfg = _training_data_cfg()
    t2_seg_exists = _resolve_data_path(
        patient_id,
        data_cfg.get("input_mask_suffix") or data_cfg.get("mask_suffix", "_segmentation"),
        data_dir,
        data_cfg,
        "t2_subdir",
    ) is not None
    ff_seg_exists = _resolve_data_path(
        patient_id,
        data_cfg.get("mask_suffix", "_segmentation"),
        data_dir,
        data_cfg,
        "mask_subdir",
    ) is not None

    return jsonify(
        {
            "t2": {"shape": list(t2.shape), "slices": int(t2.shape[0]), "has_seg": t2_seg_exists},
            "ff": {"shape": list(ff.shape), "slices": int(ff.shape[0]), "has_seg": ff_seg_exists},
        }
    )


@app.route("/api/slice/<patient_id>/<kind>/<int:slice_idx>")
def get_slice(patient_id: str, kind: str, slice_idx: int):
    try:
        volume = _load_training_volume(patient_id, kind)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404

    slice_idx = max(0, min(slice_idx, volume.shape[0] - 1))
    slice_2d = volume[slice_idx]

    if kind in ("t2",):
        img = _normalize_t2(slice_2d)
    elif kind in ("ff",):
        img = _normalize_ff(slice_2d)
    elif kind in ("t2_seg", "ff_seg", "ff_seg_eroded"):
        img = _mask_to_uint8(slice_2d)
    else:
        return jsonify({"error": f"Unknown kind: {kind}"}), 400

    pil_img = Image.fromarray(img, mode="L")
    buffer = io.BytesIO()
    pil_img.save(buffer, format="PNG")
    buffer.seek(0)
    return app.response_class(buffer, mimetype="image/png")


@app.route("/")
def index():
    return send_from_directory(str(FRONTEND_DIR), "index.html")


@app.route("/sampling")
def sampling_view():
    return send_from_directory(str(FRONTEND_DIR), "sampling.html")


if __name__ == "__main__":
    app.logger.info("Starting HTML viewer backend on http://0.0.0.0:8000")
    app.run(host="0.0.0.0", port=8000, debug=False)
