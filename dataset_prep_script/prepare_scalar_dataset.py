#!/usr/bin/env python3
"""
Prepare a scalar regression dataset with patient subfolders.

For each patient:
  - copy fat fraction and segmentation from datasets/preprocessed_images_fixed3
  - convert original T2 DICOM series to NIfTI
  - skip or resume partially processed patients
"""

from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pydicom
import SimpleITK as sitk


def load_dicom_volume(dicom_dir: Path) -> tuple[np.ndarray, dict]:
    print(f"Loading DICOM from: {dicom_dir}")

    dicom_files = []
    for root, _, files in os.walk(dicom_dir):
        for file in files:
            if file.startswith("."):
                continue
            filepath = os.path.join(root, file)
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                dicom_files.append(filepath)
            except Exception:
                continue

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    slices = []
    for filepath in dicom_files:
        dcm = pydicom.dcmread(filepath, force=True)
        slices.append((filepath, dcm))

    slices.sort(
        key=lambda x: float(x[1].ImagePositionPatient[2])
        if hasattr(x[1], "ImagePositionPatient")
        else float(x[1].SliceLocation)
        if hasattr(x[1], "SliceLocation")
        else int(x[1].InstanceNumber)
    )

    volume_slices = []
    for _, dcm in slices:
        try:
            pixel_array = dcm.pixel_array.astype(np.float32)
        except (AttributeError, KeyError):
            continue
        volume_slices.append(pixel_array)

    if not volume_slices:
        raise ValueError(f"No valid DICOM slices with pixel data found in {dicom_dir}")

    volume = np.stack(volume_slices, axis=0)

    reference_dcm = None
    for _, dcm in slices:
        if hasattr(dcm, "PixelSpacing") and hasattr(dcm, "ImagePositionPatient"):
            reference_dcm = dcm
            break
    if reference_dcm is None:
        reference_dcm = slices[0][1]

    metadata = {
        "pixel_spacing": [float(x) for x in reference_dcm.PixelSpacing]
        if hasattr(reference_dcm, "PixelSpacing")
        else [1.0, 1.0],
        "slice_thickness": float(reference_dcm.SliceThickness)
        if hasattr(reference_dcm, "SliceThickness")
        else 1.0,
    }
    return volume, metadata


def save_as_nifti(volume: np.ndarray, output_path: Path, spacing) -> None:
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing)
    sitk.WriteImage(sitk_image, str(output_path))
    print(f"  Saved: {output_path}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Prepare scalar regression dataset")
    parser.add_argument(
        "--preprocessed-dir",
        default="datasets/preprocessed_images_fixed3",
        help="Directory with per-patient FF/segmentation",
    )
    parser.add_argument(
        "--dicom-root",
        default="/home/homesOnMaster/dgeiger/repos/T2Liver/patient_data",
        help="Root directory with patient DICOM folders",
    )
    parser.add_argument(
        "--output-dir",
        default="datasets/patient_data_regression_setup",
        help="Output directory for assembled dataset",
    )
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    preprocessed_dir = Path(args.preprocessed_dir)
    dicom_root = Path(args.dicom_root)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    patient_dirs = sorted([p for p in preprocessed_dir.iterdir() if p.is_dir()])
    if not patient_dirs:
        raise ValueError(f"No patient subdirectories found in {preprocessed_dir}")

    for patient_dir in patient_dirs:
        patient_id = patient_dir.name
        ff_path = patient_dir / f"{patient_id}_ff_normalized.nii.gz"
        seg_path = patient_dir / f"{patient_id}_segmentation.nii.gz"
        dicom_dir = dicom_root / patient_id / "MR HR_T2_AX_MVXD"

        if not ff_path.exists() or not seg_path.exists():
            print(f"Skipping {patient_id}: missing FF or segmentation")
            continue
        if not dicom_dir.exists():
            print(f"Skipping {patient_id}: missing DICOM dir {dicom_dir}")
            continue

        out_patient_dir = output_dir / patient_id
        out_patient_dir.mkdir(parents=True, exist_ok=True)

        out_ff = out_patient_dir / ff_path.name
        out_seg = out_patient_dir / seg_path.name
        out_t2 = out_patient_dir / f"{patient_id}_t2_original.nii.gz"

        if out_patient_dir.exists() and not args.overwrite:
            missing = [
                name
                for name, path in {
                    "ff": out_ff,
                    "seg": out_seg,
                    "t2": out_t2,
                }.items()
                if not path.exists()
            ]
            if not missing:
                print(f"Skipping {patient_id}: outputs already exist")
                continue
            print(f"Resuming {patient_id}: missing {', '.join(missing)}")

        if args.overwrite or not out_ff.exists():
            shutil.copy2(ff_path, out_ff)
        if args.overwrite or not out_seg.exists():
            shutil.copy2(seg_path, out_seg)
        print(f"Copied FF and segmentation for {patient_id}")

        if args.overwrite or not out_t2.exists():
            volume, meta = load_dicom_volume(dicom_dir)
            spacing = [meta["pixel_spacing"][0], meta["pixel_spacing"][1], meta["slice_thickness"]]
            save_as_nifti(volume, out_t2, spacing)


if __name__ == "__main__":
    main()
