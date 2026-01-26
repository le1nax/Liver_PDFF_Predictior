#!/usr/bin/env python3
"""
Precompute fatty liver patients based on mean fat fraction in liver segmentation.
Creates a YAML file mapping patient IDs to their mean liver fat fraction.
"""

import os
import sys
import yaml
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from tqdm import tqdm


def compute_mean_ff_in_liver(ff_file, seg_file):
    """
    Compute mean fat fraction within liver segmentation mask.

    Args:
        ff_file: Path to fat fraction NIfTI file
        seg_file: Path to segmentation NIfTI file

    Returns:
        float: Mean fat fraction (0-1) in liver region, or None if error
    """
    try:
        # Load volumes
        ff_img = sitk.ReadImage(str(ff_file))
        seg_img = sitk.ReadImage(str(seg_file))

        ff_volume = sitk.GetArrayFromImage(ff_img)
        seg_volume = sitk.GetArrayFromImage(seg_img)

        # Create liver mask (segmentation value > 0)
        liver_mask = seg_volume > 0

        # Check if there are any liver pixels
        if not np.any(liver_mask):
            print(f"  Warning: No liver pixels found in segmentation")
            return None

        # Extract FF values in liver region
        ff_in_liver = ff_volume[liver_mask]

        # Compute mean
        mean_ff = np.mean(ff_in_liver)

        return float(mean_ff)

    except Exception as e:
        print(f"  Error processing: {e}")
        return None


def precompute_fatty_liver_patients(preprocessed_dir, output_yaml, threshold=0.1):
    """
    Scan all patients and identify those with fatty liver.

    Args:
        preprocessed_dir: Path to preprocessed_images directory
        output_yaml: Path to output YAML file
        threshold: Mean FF threshold for fatty liver classification (default 0.1 = 10%)
    """
    base_path = Path(preprocessed_dir)

    if not base_path.exists():
        print(f"Error: Directory not found: {preprocessed_dir}")
        return

    print(f"Scanning patients in: {preprocessed_dir}")
    print(f"Fatty liver threshold: {threshold * 100:.1f}%")
    print("=" * 80)

    results = {
        'threshold': threshold,
        'patients': {}
    }

    # Get all patient directories
    patient_dirs = sorted([d for d in base_path.iterdir() if d.is_dir()])

    print(f"Found {len(patient_dirs)} patient directories")
    print("Processing...")
    print()
    sys.stdout.flush()

    fatty_liver_count = 0

    for i, patient_dir in enumerate(tqdm(patient_dirs, desc="Analyzing patients")):
        patient_id = patient_dir.name

        # Look for the three expected files
        ff_file = patient_dir / f"{patient_id}_ff_normalized.nii.gz"
        seg_file = patient_dir / f"{patient_id}_segmentation.nii.gz"

        # Skip if files don't exist
        if not ff_file.exists() or not seg_file.exists():
            continue

        # Compute mean FF in liver
        mean_ff = compute_mean_ff_in_liver(ff_file, seg_file)

        if mean_ff is not None:
            # Ensure patient_id is stored as string to maintain leading zeros
            results['patients'][str(patient_id)] = {
                'mean_ff': round(mean_ff, 6),
                'mean_ff_percent': round(mean_ff * 100, 2),
                'has_fatty_liver': mean_ff >= threshold
            }

            if mean_ff >= threshold:
                fatty_liver_count += 1

    # Save results
    print()
    print("=" * 80)
    print(f"Analysis complete!")
    print(f"Total patients analyzed: {len(results['patients'])}")
    print(f"Fatty liver patients (≥{threshold * 100:.1f}%): {fatty_liver_count}")
    print(f"Normal liver patients (<{threshold * 100:.1f}%): {len(results['patients']) - fatty_liver_count}")
    print()

    # Write YAML file
    output_path = Path(output_yaml)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_yaml, 'w') as f:
        # Use safe_dump with explicit string quoting for numeric-looking strings
        yaml.dump(results, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"Results saved to: {output_yaml}")
    print()

    # Print some statistics
    if results['patients']:
        ff_values = [p['mean_ff'] for p in results['patients'].values()]
        print("Fat Fraction Statistics:")
        print(f"  Min:    {np.min(ff_values) * 100:.2f}%")
        print(f"  Max:    {np.max(ff_values) * 100:.2f}%")
        print(f"  Mean:   {np.mean(ff_values) * 100:.2f}%")
        print(f"  Median: {np.median(ff_values) * 100:.2f}%")
        print()

    # Print top 10 fatty liver patients
    fatty_patients = [(pid, data['mean_ff_percent'])
                      for pid, data in results['patients'].items()
                      if data['has_fatty_liver']]

    if fatty_patients:
        fatty_patients.sort(key=lambda x: x[1], reverse=True)
        print("Top 10 patients with highest liver fat fraction:")
        for i, (pid, ff_pct) in enumerate(fatty_patients[:10], 1):
            print(f"  {i:2d}. {pid}: {ff_pct:.2f}%")
        print()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description='Precompute fatty liver patients from preprocessed data'
    )
    parser.add_argument(
        '--input-dir',
        default='/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/preprocessed_images_fixed3',
        help='Path to preprocessed_images directory'
    )
    parser.add_argument(
        '--output',
        default='/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/preprocessed_images_fixed3/fatty_liver_patients.yaml',
        help='Path to output YAML file'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.1,
        help='Mean FF threshold for fatty liver (0-1 range, default 0.1 = 10%%)'
    )

    args = parser.parse_args()

    precompute_fatty_liver_patients(
        args.input_dir,
        args.output,
        args.threshold
    )
