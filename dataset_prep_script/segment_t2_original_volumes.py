#!/usr/bin/env python3
"""
Run liver segmentation on original T2 NIfTI volumes and save masks alongside inputs.

Expected input structure:
  datasets/patient_data_regression_setup/
    <patient_id>/<patient_id>_t2_original.nii.gz

Output:
  datasets/patient_data_regression_setup/
    <patient_id>/<patient_id>_t2_original_segmentation.nii.gz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import SimpleITK as sitk

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from segmentation_module import LiverSegmenter  # noqa: E402


def load_nifti(path: Path) -> tuple[np.ndarray, tuple[float, float, float]]:
    image = sitk.ReadImage(str(path))
    volume = sitk.GetArrayFromImage(image)
    spacing = image.GetSpacing()
    return volume, spacing


def save_nifti(volume: np.ndarray, spacing: tuple[float, float, float], path: Path) -> None:
    image = sitk.GetImageFromArray(volume)
    image.SetSpacing(spacing)
    sitk.WriteImage(image, str(path))
    print(f"  Saved: {path}")


def process_patient(patient_dir: Path, segmenter: LiverSegmenter, overwrite: bool) -> None:
    t2_path = patient_dir / f"{patient_dir.name}_t2_original.nii.gz"
    if not t2_path.exists():
        print(f"Skipping {patient_dir.name}: no {t2_path.name} found")
        return

    output_path = patient_dir / f"{patient_dir.name}_t2_original_segmentation.nii.gz"
    if output_path.exists() and not overwrite:
        print(f"Skipping {patient_dir.name}: mask already exists")
        return

    print(f"\nProcessing {patient_dir.name}")
    volume, spacing = load_nifti(t2_path)
    mask = segmenter.segment_volume(volume)
    save_nifti(mask.astype(np.uint8), spacing, output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description="Segment original T2 volumes and save masks.")
    parser.add_argument(
        "--input_dir",
        type=str,
        default="datasets/patient_data_regression_setup",
        help="Directory with per-patient original T2 NIfTI files",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing masks")
    parser.add_argument("--checkpoint-path", default=None, help="nnU-Net checkpoint path (overrides config)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        raise SystemExit(f"Input directory does not exist: {input_dir}")

    segmenter = LiverSegmenter(checkpoint_path=args.checkpoint_path)
    patient_dirs = [p for p in input_dir.iterdir() if p.is_dir()]
    for patient_dir in sorted(patient_dirs):
        try:
            process_patient(patient_dir, segmenter, args.overwrite)
        except Exception as exc:
            print(f"Error processing {patient_dir.name}: {exc}")


if __name__ == "__main__":
    main()
