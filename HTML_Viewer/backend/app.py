#!/usr/bin/env python3
"""
Flask backend for the T2/FF viewer.

Serves slice PNGs and patient metadata from datasets/patient_data_regression_setup.
"""

from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import SimpleITK as sitk
from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

DATA_DIR = Path(os.environ.get("HTML_VIEWER_DATA_DIR", "datasets/patient_data_regression_setup"))
FRONTEND_DIR = Path(__file__).resolve().parents[1] / "frontend"

app = Flask(__name__, static_folder=str(FRONTEND_DIR), static_url_path="")

_CACHE: Dict[Tuple[str, str], np.ndarray] = {}
_CACHE_ORDER: list[Tuple[str, str]] = []
_CACHE_MAX = 8


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


def _find_file(patient_id: str, filename: str) -> Path | None:
    candidate = DATA_DIR / patient_id / filename
    if candidate.exists():
        return candidate
    if filename.endswith(".nii.gz"):
        alt = DATA_DIR / patient_id / filename.replace(".nii.gz", ".nii")
        if alt.exists():
            return alt
    return None


def _erode_mask(volume: np.ndarray, radius: int = 3) -> np.ndarray:
    image = sitk.GetImageFromArray((volume > 0).astype(np.uint8))
    eroded = sitk.BinaryErode(image, [radius, radius, radius])
    return sitk.GetArrayFromImage(eroded).astype(np.uint8)


def _load_volume(patient_id: str, kind: str) -> np.ndarray:
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

    path = _find_file(patient_id, filename)
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


def _mask_to_uint8(slice_2d: np.ndarray) -> np.ndarray:
    mask = (slice_2d > 0).astype(np.uint8) * 255
    return mask


@app.route("/api/patients")
def list_patients():
    if not DATA_DIR.exists():
        return jsonify({"patients": [], "error": f"Missing data dir: {DATA_DIR}"}), 200
    patients = []
    for p in sorted(DATA_DIR.iterdir()):
        if not p.is_dir():
            continue
        pid = p.name
        t2_path = _find_file(pid, f"{pid}_t2_original.nii.gz")
        ff_path = _find_file(pid, f"{pid}_ff_normalized.nii.gz")
        if t2_path and ff_path:
            patients.append(pid)
    return jsonify({"patients": patients})


@app.route("/api/info/<patient_id>")
def patient_info(patient_id: str):
    try:
        t2 = _load_volume(patient_id, "t2")
        ff = _load_volume(patient_id, "ff")
    except Exception as exc:
        return jsonify({"error": str(exc)}), 404

    t2_seg_exists = _find_file(patient_id, f"{patient_id}_t2_original_segmentation.nii.gz") is not None
    ff_seg_exists = _find_file(patient_id, f"{patient_id}_segmentation.nii.gz") is not None

    return jsonify(
        {
            "t2": {"shape": list(t2.shape), "slices": int(t2.shape[0]), "has_seg": t2_seg_exists},
            "ff": {"shape": list(ff.shape), "slices": int(ff.shape[0]), "has_seg": ff_seg_exists},
        }
    )


@app.route("/api/slice/<patient_id>/<kind>/<int:slice_idx>")
def get_slice(patient_id: str, kind: str, slice_idx: int):
    try:
        volume = _load_volume(patient_id, kind)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, debug=False)
