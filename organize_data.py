#!/usr/bin/env python3
"""
Organize preprocessed patient data for training without copying files.
Creates symbolic links to avoid data duplication.

Usage:
    python organize_data.py --input_dir ./processed_data --output_dir ./training_data
    python organize_data.py --input_dir ./processed_data --output_dir ./training_data --method copy
"""

import argparse
import os
from pathlib import Path
import shutil


def organize_with_symlinks(input_dir: Path, output_dir: Path):
    """
    Create symbolic links to files instead of copying.
    Fast and doesn't duplicate data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        print(f"Linking {patient_id}...")

        for nifti_file in patient_dir.glob("*.nii.gz"):
            target = output_dir / nifti_file.name
            source = nifti_file.absolute()

            # Remove existing symlink/file if present
            if target.exists() or target.is_symlink():
                target.unlink()

            # Create symlink
            target.symlink_to(source)
            count += 1

    print(f"\n✓ Created {count} symbolic links in {output_dir}")
    print("Note: Symbolic links point to original files - no data duplication!")


def organize_with_copy(input_dir: Path, output_dir: Path):
    """
    Copy files to output directory.
    Slower but creates independent copy of data.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name
        print(f"Copying {patient_id}...")

        for nifti_file in patient_dir.glob("*.nii.gz"):
            target = output_dir / nifti_file.name
            shutil.copy2(nifti_file, target)
            count += 1

    print(f"\n✓ Copied {count} files to {output_dir}")


def organize_direct_mapping(input_dir: Path, output_file: Path):
    """
    Create a JSON mapping file instead of organizing files.
    The dataset loader can read directly from the original structure.
    """
    import json

    mapping = {}
    for patient_dir in sorted(input_dir.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # Find files
        t2_file = None
        ff_file = None
        seg_file = None

        for f in patient_dir.glob("*.nii.gz"):
            if "_t2_aligned" in f.name:
                t2_file = str(f.absolute())
            elif "_ff_normalized" in f.name:
                ff_file = str(f.absolute())
            elif "_segmentation" in f.name:
                seg_file = str(f.absolute())

        if t2_file and ff_file and seg_file:
            mapping[patient_id] = {
                "t2": t2_file,
                "fat_fraction": ff_file,
                "segmentation": seg_file
            }

    with open(output_file, 'w') as f:
        json.dump(mapping, f, indent=2)

    print(f"\n✓ Created mapping for {len(mapping)} patients")
    print(f"Mapping saved to: {output_file}")
    print("\nNo files moved - dataset can read from original locations!")


def main():
    parser = argparse.ArgumentParser(
        description="Organize preprocessed data for training"
    )
    parser.add_argument('--input_dir', type=str, required=True,
                       help='Input directory with patient subdirectories')
    parser.add_argument('--output_dir', type=str, default='./training_data',
                       help='Output directory for organized data')
    parser.add_argument('--method', type=str, default='symlink',
                       choices=['symlink', 'copy', 'mapping'],
                       help='Organization method: symlink (fast, recommended), copy (independent), or mapping (no file movement)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)

    if not input_dir.exists():
        print(f"Error: Input directory not found: {input_dir}")
        return

    print(f"Organizing data from: {input_dir}")
    print(f"Method: {args.method}\n")

    if args.method == 'symlink':
        output_dir = Path(args.output_dir)
        organize_with_symlinks(input_dir, output_dir)
        print(f"\nUpdate config/config_pixel_level/train_config.yaml:")
        print(f"  data_dir: \"{output_dir.absolute()}\"")

    elif args.method == 'copy':
        output_dir = Path(args.output_dir)
        organize_with_copy(input_dir, output_dir)
        print(f"\nUpdate config/config_pixel_level/train_config.yaml:")
        print(f"  data_dir: \"{output_dir.absolute()}\"")

    elif args.method == 'mapping':
        output_file = Path(args.output_dir).parent / 'patient_mapping.json'
        organize_direct_mapping(input_dir, output_file)
        print(f"\nTo use mapping, point dataset directly to processed_data:")
        print(f"  data_dir: \"{input_dir.absolute()}\"")
        print(f"  use_patient_subdirs: true  # Add this option")


if __name__ == '__main__':
    main()
