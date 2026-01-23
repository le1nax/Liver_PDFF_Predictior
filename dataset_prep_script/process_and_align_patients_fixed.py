#!/usr/bin/env python3
"""
CORRECTED VERSION: Process T2 and Fat Fraction DICOM images

Key fixes:
1. Added metadata validation to catch incorrect pixel spacing
2. Added detailed debug output for FOV calculations
3. Added verification that cropped regions have matching physical extents
4. Added sanity checks for crop coordinates

Usage:
    python process_and_align_patients_fixed.py --input_dir ./patient_data --output_dir ./output
"""

import os
import sys
import argparse
import numpy as np
import SimpleITK as sitk
from pathlib import Path
from scipy.ndimage import zoom
import pydicom

# Import segmentation module
from segmentation_module import LiverSegmenter


def load_dicom_volume(dicom_dir, select_image_type=None):
    """
    Load a DICOM volume and return the 3D array plus metadata

    Args:
        dicom_dir: Directory containing DICOM files
        select_image_type: For Dixon sequences, specify which image type to load ('FF' for Fat Fraction)

    Returns:
        volume: numpy array (D, H, W)
        metadata: dict with DICOM metadata
    """
    print(f"Loading DICOM from: {dicom_dir}")
    if select_image_type:
        print(f"  Filtering for image type containing: {select_image_type}")

    # Get all DICOM files
    dicom_files = []
    for root, dirs, files in os.walk(dicom_dir):
        for file in files:
            if file.startswith('.'):
                continue
            filepath = os.path.join(root, file)
            try:
                pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                dicom_files.append(filepath)
            except:
                continue

    if not dicom_files:
        raise ValueError(f"No DICOM files found in {dicom_dir}")

    print(f"  Found {len(dicom_files)} DICOM files")

    # If we need to filter by ImageType (for Dixon sequences)
    if select_image_type:
        filtered_files = []
        for filepath in dicom_files:
            try:
                dcm = pydicom.dcmread(filepath, stop_before_pixels=True, force=True)
                image_type = getattr(dcm, 'ImageType', None)
                if image_type and select_image_type in str(image_type):
                    filtered_files.append(filepath)
            except:
                continue

        if not filtered_files:
            raise ValueError(f"No DICOM files with ImageType containing '{select_image_type}' found")

        dicom_files = filtered_files
        print(f"  Filtered to {len(dicom_files)} files with ImageType containing '{select_image_type}'")

    # Sort files by ImagePositionPatient Z coordinate or SliceLocation or InstanceNumber
    slices = []
    for filepath in dicom_files:
        dcm = pydicom.dcmread(filepath, force=True)
        slices.append((filepath, dcm))

    slices.sort(key=lambda x: float(x[1].ImagePositionPatient[2]) if hasattr(x[1], 'ImagePositionPatient')
                else float(x[1].SliceLocation) if hasattr(x[1], 'SliceLocation')
                else int(x[1].InstanceNumber))

    volume_slices = []
    slice_positions = {}
    valid_idx = 0

    for filepath, dcm in slices:
        try:
            pixel_array = dcm.pixel_array.astype(np.float32)
        except (AttributeError, KeyError):
            continue

        # T2 images: Keep raw pixel values (no rescaling)
        # FF images: MUST have RealWorldValueMappingSequence, otherwise skip
        if select_image_type == 'FF' and not hasattr(dcm, 'RealWorldValueMappingSequence'):
            # Skip FF images without RWVM - cannot be properly scaled
            continue

        volume_slices.append(pixel_array)

        if hasattr(dcm, 'ImagePositionPatient'):
            slice_positions[valid_idx] = float(dcm.ImagePositionPatient[2])
        elif hasattr(dcm, 'SliceLocation'):
            slice_positions[valid_idx] = float(dcm.SliceLocation)
        else:
            slice_positions[valid_idx] = valid_idx * 5.0

        valid_idx += 1

    if not volume_slices:
        raise ValueError(f"No valid DICOM slices with pixel data found in {dicom_dir}")

    volume = np.stack(volume_slices, axis=0)

    # CRITICAL FIX: Find first slice with valid PixelSpacing and ImagePositionPatient
    # Don't just use slices[0] as it might have missing metadata
    reference_dcm = None
    for filepath, dcm in slices:
        if hasattr(dcm, 'PixelSpacing') and hasattr(dcm, 'ImagePositionPatient'):
            reference_dcm = dcm
            print(f"  Using {Path(filepath).name} as reference for metadata (has valid PixelSpacing and ImagePositionPatient)")
            break

    if reference_dcm is None:
        # Fallback to first slice if none have both attributes
        reference_dcm = slices[0][1]
        print(f"  WARNING: No slices with complete metadata found, using first slice")

    metadata = {
        'pixel_spacing': [float(x) for x in reference_dcm.PixelSpacing] if hasattr(reference_dcm, 'PixelSpacing') else [1.0, 1.0],
        'slice_thickness': float(reference_dcm.SliceThickness) if hasattr(reference_dcm, 'SliceThickness') else 1.0,
        'image_position': [float(x) for x in reference_dcm.ImagePositionPatient] if hasattr(reference_dcm, 'ImagePositionPatient') else [0, 0, 0],
        'image_orientation': [float(x) for x in reference_dcm.ImageOrientationPatient] if hasattr(reference_dcm, 'ImageOrientationPatient') else [1, 0, 0, 0, 1, 0],
        'slice_positions': slice_positions,
        'rows': int(reference_dcm.Rows),
        'cols': int(reference_dcm.Columns),
        'real_world_slope': None,
        'real_world_intercept': None
    }

    # Attempt to parse RealWorldValueMappingSequence for Fat Fraction scaling
    if hasattr(reference_dcm, 'RealWorldValueMappingSequence'):
        try:
            rwvm = reference_dcm.RealWorldValueMappingSequence[0]
            slope = float(rwvm.RealWorldValueSlope)
            intercept = float(rwvm.RealWorldValueIntercept)
            metadata['real_world_slope'] = slope
            metadata['real_world_intercept'] = intercept
            print(f"  Found RealWorldValueMappingSequence: slope={slope}, intercept={intercept}")
        except Exception:
            pass

    print(f"  Volume shape: {volume.shape}")
    print(f"  Pixel spacing: {metadata['pixel_spacing']}")
    print(f"  Slice thickness: {metadata['slice_thickness']}")
    print(f"  ImagePositionPatient: {metadata['image_position']}")

    # VALIDATION: Calculate and display FOV
    fov_x = metadata['cols'] * metadata['pixel_spacing'][0]
    fov_y = metadata['rows'] * metadata['pixel_spacing'][1]
    print(f"  Field of View: {fov_x:.2f} x {fov_y:.2f} mm")

    return volume, metadata


def calculate_xy_crop_params(t2_meta, ff_meta):
    """
    Calculate crop parameters to find overlapping FOV between T2 and FF images.

    Returns dict with crop coordinates in pixel space for both images.
    """
    try:
        t2_ipp = t2_meta['image_position']
        ff_ipp = ff_meta['image_position']
        t2_ps = t2_meta['pixel_spacing']
        ff_ps = ff_meta['pixel_spacing']
        t2_rows, t2_cols = t2_meta['rows'], t2_meta['cols']
        ff_rows, ff_cols = ff_meta['rows'], ff_meta['cols']

        # Calculate FOV extents in physical space (mm)
        t2_x_start, t2_y_start = t2_ipp[0], t2_ipp[1]
        t2_x_end = t2_x_start + t2_cols * t2_ps[0]
        t2_y_end = t2_y_start + t2_rows * t2_ps[1]

        ff_x_start, ff_y_start = ff_ipp[0], ff_ipp[1]
        ff_x_end = ff_x_start + ff_cols * ff_ps[0]
        ff_y_end = ff_y_start + ff_rows * ff_ps[1]

        print(f"\n  DEBUG: FOV Calculation")
        print(f"  T2 FOV: X[{t2_x_start:.2f} to {t2_x_end:.2f}], Y[{t2_y_start:.2f} to {t2_y_end:.2f}] mm")
        print(f"  FF FOV: X[{ff_x_start:.2f} to {ff_x_end:.2f}], Y[{ff_y_start:.2f} to {ff_y_end:.2f}] mm")

        # Calculate overlap region
        overlap_x_start = max(t2_x_start, ff_x_start)
        overlap_x_end = min(t2_x_end, ff_x_end)
        overlap_y_start = max(t2_y_start, ff_y_start)
        overlap_y_end = min(t2_y_end, ff_y_end)

        overlap_width = overlap_x_end - overlap_x_start
        overlap_height = overlap_y_end - overlap_y_start

        print(f"  Overlap: X[{overlap_x_start:.2f} to {overlap_x_end:.2f}], Y[{overlap_y_start:.2f} to {overlap_y_end:.2f}] mm")
        print(f"  Overlap size: {overlap_width:.2f} x {overlap_height:.2f} mm")

        # VALIDATION: Check for valid overlap
        if overlap_width <= 0 or overlap_height <= 0:
            raise ValueError(f"No overlap between T2 and FF FOVs! Overlap: {overlap_width:.2f} x {overlap_height:.2f} mm")

        # Convert overlap region to pixel coordinates
        t2_crop_x_start_px = int(round((overlap_x_start - t2_x_start) / t2_ps[0]))
        t2_crop_x_end_px = int(round((overlap_x_end - t2_x_start) / t2_ps[0]))
        t2_crop_y_start_px = int(round((overlap_y_start - t2_y_start) / t2_ps[1]))
        t2_crop_y_end_px = int(round((overlap_y_end - t2_y_start) / t2_ps[1]))

        ff_crop_x_start_px = int(round((overlap_x_start - ff_x_start) / ff_ps[0]))
        ff_crop_x_end_px = int(round((overlap_x_end - ff_x_start) / ff_ps[0]))
        ff_crop_y_start_px = int(round((overlap_y_start - ff_y_start) / ff_ps[1]))
        ff_crop_y_end_px = int(round((overlap_y_end - ff_y_start) / ff_ps[1]))

        # VALIDATION: Ensure crop coordinates are within bounds
        t2_crop_x_start_px = max(0, min(t2_crop_x_start_px, t2_cols))
        t2_crop_x_end_px = max(0, min(t2_crop_x_end_px, t2_cols))
        t2_crop_y_start_px = max(0, min(t2_crop_y_start_px, t2_rows))
        t2_crop_y_end_px = max(0, min(t2_crop_y_end_px, t2_rows))

        ff_crop_x_start_px = max(0, min(ff_crop_x_start_px, ff_cols))
        ff_crop_x_end_px = max(0, min(ff_crop_x_end_px, ff_cols))
        ff_crop_y_start_px = max(0, min(ff_crop_y_start_px, ff_rows))
        ff_crop_y_end_px = max(0, min(ff_crop_y_end_px, ff_rows))

        # Calculate cropped sizes
        t2_crop_width = t2_crop_x_end_px - t2_crop_x_start_px
        t2_crop_height = t2_crop_y_end_px - t2_crop_y_start_px
        ff_crop_width = ff_crop_x_end_px - ff_crop_x_start_px
        ff_crop_height = ff_crop_y_end_px - ff_crop_y_start_px

        # Calculate physical extents of cropped regions
        t2_crop_physical_width = t2_crop_width * t2_ps[0]
        t2_crop_physical_height = t2_crop_height * t2_ps[1]
        ff_crop_physical_width = ff_crop_width * ff_ps[0]
        ff_crop_physical_height = ff_crop_height * ff_ps[1]

        print(f"\n  DEBUG: Crop Results")
        print(f"  T2 crop: {t2_crop_width}x{t2_crop_height} pixels = {t2_crop_physical_width:.2f}x{t2_crop_physical_height:.2f} mm")
        print(f"  FF crop: {ff_crop_width}x{ff_crop_height} pixels = {ff_crop_physical_width:.2f}x{ff_crop_physical_height:.2f} mm")

        # VALIDATION: Check that physical extents match (within 1mm tolerance)
        width_diff = abs(t2_crop_physical_width - ff_crop_physical_width)
        height_diff = abs(t2_crop_physical_height - ff_crop_physical_height)

        if width_diff > 1.0 or height_diff > 1.0:
            print(f"  WARNING: Physical extent mismatch! Width diff: {width_diff:.2f}mm, Height diff: {height_diff:.2f}mm")
            print(f"  This indicates incorrect pixel spacing or crop calculation.")
        else:
            print(f"  ✓ Physical extents match within tolerance ({width_diff:.3f}mm, {height_diff:.3f}mm)")

    except Exception as e:
        print(f"  Error calculating crop params: {e}")
        import traceback
        traceback.print_exc()
        return None

    return {
        't2_crop': {
            'x_start': t2_crop_x_start_px,
            'x_end': t2_crop_x_end_px,
            'y_start': t2_crop_y_start_px,
            'y_end': t2_crop_y_end_px
        },
        'ff_crop': {
            'x_start': ff_crop_x_start_px,
            'x_end': ff_crop_x_end_px,
            'y_start': ff_crop_y_start_px,
            'y_end': ff_crop_y_end_px
        },
        'overlap_physical': {
            'width': overlap_width,
            'height': overlap_height
        }
    }


def align_t2_to_ff(t2_volume, ff_volume, t2_meta, ff_meta, segmentation=None):
    """
    Align T2 image to FF image by:
    1. Cropping both to overlapping FOV
    2. Resampling T2 to match FF resolution
    3. Matching slices in Z direction
    """
    print("\nAligning T2 to FF...")

    # Apply correct FF scaling if available
    slope = ff_meta.get('real_world_slope', None)
    intercept = ff_meta.get('real_world_intercept', None)

    if slope is not None and intercept is not None:
        print(f"  Applying FF scaling: value * {slope} + {intercept}")
        ff_scaled = ff_volume * slope + intercept
        ff_scaled = np.clip(ff_scaled, 0, 100)  # Clip to 0-100%
        ff_normalized = ff_scaled / 100.0       # Normalize to [0,1]
    else:
        print("  Warning: RealWorldValueMappingSequence missing, falling back to min-max normalization")
        ff_min = ff_volume.min()
        ff_max = ff_volume.max()
        if ff_max > ff_min:
            ff_normalized = (ff_volume - ff_min) / (ff_max - ff_min)
        else:
            print("  Warning: FF volume is constant; setting to zeros")
            ff_normalized = np.zeros_like(ff_volume)

    print(f"  Fat fraction value range after normalization: [{ff_normalized.min():.4f}, {ff_normalized.max():.4f}]")

    # Calculate crop parameters
    crop_params = calculate_xy_crop_params(t2_meta, ff_meta)
    if not crop_params:
        raise ValueError("Failed to calculate crop parameters")

    print(f"\n  Crop coordinates (pixel space):")
    print(f"  T2: X[{crop_params['t2_crop']['x_start']}:{crop_params['t2_crop']['x_end']}], "
          f"Y[{crop_params['t2_crop']['y_start']}:{crop_params['t2_crop']['y_end']}]")
    print(f"  FF: X[{crop_params['ff_crop']['x_start']}:{crop_params['ff_crop']['x_end']}], "
          f"Y[{crop_params['ff_crop']['y_start']}:{crop_params['ff_crop']['y_end']}]")

    # Z-direction slice matching
    t2_z_positions = t2_meta['slice_positions']
    ff_z_positions = ff_meta['slice_positions']

    t2_slices = sorted([(int(k), float(v)) for k, v in t2_z_positions.items()], key=lambda x: x[1])
    ff_slices = sorted([(int(k), float(v)) for k, v in ff_z_positions.items()], key=lambda x: x[1])

    print(f"\n  Z-direction matching:")
    print(f"  T2 Z range: {t2_slices[0][1]:.2f} to {t2_slices[-1][1]:.2f}mm ({len(t2_slices)} slices)")
    print(f"  FF Z range: {ff_slices[0][1]:.2f} to {ff_slices[-1][1]:.2f}mm ({len(ff_slices)} slices)")

    # Match T2 and FF slices
    matches = []
    used_t2_indices = set()

    for ff_idx, ff_z in ff_slices:
        best_t2_idx = None
        best_distance = float('inf')
        best_t2_z = None

        for t2_idx, t2_z in t2_slices:
            if t2_idx in used_t2_indices:
                continue
            distance = abs(t2_z - ff_z)
            if distance < best_distance:
                best_distance = distance
                best_t2_idx = t2_idx
                best_t2_z = t2_z

        # Use 3mm threshold for slice matching
        if best_t2_idx is not None and best_distance < 3.0:
            matches.append({
                't2_idx': best_t2_idx,
                'ff_idx': ff_idx,
                't2_z': best_t2_z,
                'ff_z': ff_z,
                'distance': best_distance
            })
            used_t2_indices.add(best_t2_idx)

    print(f"  Found {len(matches)} matched slice pairs")

    if len(matches) == 0:
        raise ValueError("No matching slices found between T2 and FF!")

    # Process matched slices
    aligned_t2_slices = []
    aligned_ff_slices = []
    aligned_seg_slices = [] if segmentation is not None else None

    for match in matches:
        # Get slices
        t2_slice = t2_volume[match['t2_idx']]
        ff_slice = ff_normalized[match['ff_idx']]

        # Crop to overlap region
        t2_cropped = t2_slice[
            crop_params['t2_crop']['y_start']:crop_params['t2_crop']['y_end'],
            crop_params['t2_crop']['x_start']:crop_params['t2_crop']['x_end']
        ]
        ff_cropped = ff_slice[
            crop_params['ff_crop']['y_start']:crop_params['ff_crop']['y_end'],
            crop_params['ff_crop']['x_start']:crop_params['ff_crop']['x_end']
        ]

        # Calculate zoom factors to resample T2 to FF resolution
        zoom_factors = (
            ff_cropped.shape[0] / t2_cropped.shape[0],
            ff_cropped.shape[1] / t2_cropped.shape[1]
        )

        # Resample T2 to match FF resolution
        t2_resampled = zoom(t2_cropped, zoom_factors, order=1)

        aligned_t2_slices.append(t2_resampled)
        aligned_ff_slices.append(ff_cropped)

        # Process segmentation if available
        if segmentation is not None:
            seg_slice = segmentation[match['t2_idx']]
            seg_cropped = seg_slice[
                crop_params['t2_crop']['y_start']:crop_params['t2_crop']['y_end'],
                crop_params['t2_crop']['x_start']:crop_params['t2_crop']['x_end']
            ]
            seg_resampled = zoom(seg_cropped, zoom_factors, order=0)  # Nearest neighbor for segmentation
            aligned_seg_slices.append(seg_resampled)

    # Stack into 3D volumes
    aligned_t2 = np.stack(aligned_t2_slices, axis=0)
    aligned_ff = np.stack(aligned_ff_slices, axis=0)
    aligned_seg = np.stack(aligned_seg_slices, axis=0) if segmentation is not None else None

    print(f"\n  Final aligned shapes:")
    print(f"  T2: {aligned_t2.shape}")
    print(f"  FF: {aligned_ff.shape}")
    if aligned_seg is not None:
        print(f"  Segmentation: {aligned_seg.shape}")

    # VALIDATION: Ensure T2 and FF have the same shape
    if aligned_t2.shape != aligned_ff.shape:
        raise ValueError(f"Shape mismatch after alignment! T2: {aligned_t2.shape}, FF: {aligned_ff.shape}")

    return aligned_t2, aligned_ff, aligned_seg, matches


def save_as_nifti(volume, output_path, spacing=(1.0, 1.0, 1.0)):
    """
    Save a numpy array as NIfTI file

    Args:
        volume: numpy array (D, H, W)
        output_path: path to save .nii.gz file
        spacing: (z_spacing, y_spacing, x_spacing) in mm
    """
    sitk_image = sitk.GetImageFromArray(volume)
    sitk_image.SetSpacing(spacing)
    sitk.WriteImage(sitk_image, str(output_path))
    print(f"  Saved: {output_path}")


def process_patient(patient_id, t2_dir, dixon_dir, output_dir, segmenter):
    """Process a single patient: load, segment, align, and save."""
    print(f"\n{'='*80}")
    print(f"Processing patient: {patient_id}")
    print(f"{'='*80}")

    patient_output_dir = Path(output_dir) / patient_id
    patient_output_dir.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading T2 volume...")
    t2_volume, t2_meta = load_dicom_volume(t2_dir)

    print("\n[2/6] Loading Fat Fraction volume from Dixon sequence...")
    ff_volume, ff_meta = load_dicom_volume(dixon_dir, select_image_type='FF')

    print("\n[3/6] Generating liver segmentation on original T2...")
    segmentation = segmenter.segment_volume(t2_volume)
    print(f"  Segmentation shape: {segmentation.shape}")
    print(f"  Unique labels: {np.unique(segmentation)}")

    print("\n[4/6] Post-processing complete (largest component already kept)")

    print("\n[5/6] Aligning T2 and segmentation to FF...")
    aligned_t2, aligned_ff, aligned_seg, matches = align_t2_to_ff(
        t2_volume, ff_volume, t2_meta, ff_meta, segmentation
    )

    # Calculate output spacing
    ff_spacing = ff_meta['pixel_spacing']
    if len(matches) > 1:
        z_distances = [matches[i+1]['ff_z'] - matches[i]['ff_z'] for i in range(len(matches)-1)]
        avg_z_spacing = np.mean(z_distances)
    else:
        avg_z_spacing = ff_meta['slice_thickness']

    spacing = (avg_z_spacing, ff_spacing[1], ff_spacing[0])  # (Z, Y, X)
    print(f"\n  Output NIfTI spacing: {spacing} mm")

    print("\n[6/6] Saving aligned volumes as NIfTI...")
    save_as_nifti(aligned_t2, patient_output_dir / f"{patient_id}_t2_aligned.nii.gz", spacing)
    save_as_nifti(aligned_ff, patient_output_dir / f"{patient_id}_ff_normalized.nii.gz", spacing)
    if aligned_seg is not None:
        save_as_nifti(aligned_seg.astype(np.uint8), patient_output_dir / f"{patient_id}_segmentation.nii.gz", spacing)

    print(f"\n✓ Patient {patient_id} processed successfully!")


def main():
    parser = argparse.ArgumentParser(
        description="CORRECTED: Process T2 and FF DICOM images with validation and debugging"
    )
    parser.add_argument('--input_dir', type=str, default='/home/homesOnMaster/dgeiger/repos/T2Liver/patient_data',
                       help='Input directory containing patient subdirectories')
    parser.add_argument('--output_dir', type=str, default='/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/preprocessed_images_fixed3',
                       help='Output directory for processed NIfTI files')
    parser.add_argument('--t2_sequence_name', type=str, default='MR HR_T2_AX_MVXD',
                       help='Name of T2 sequence subdirectory (default: MR HR_T2_AX_MVXD)')
    parser.add_argument('--dixon_sequence_name', type=str, default='MR mDIXONQuant_BH',
                       help='Name of Dixon sequence subdirectory (default: MR mDIXONQuant_BH)')

    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists():
        print(f"Error: Input directory does not exist: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize segmenter (once for all patients)
    print("Initializing liver segmenter...")
    segmenter = LiverSegmenter()
    segmenter.load_model()
    print("✓ Segmenter ready!\n")

    # Find all patient directories
    patient_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(patient_dirs)} patient directories\n")

    # Process each patient
    success_count = 0
    failed_patients = []

    for patient_dir in sorted(patient_dirs):
        patient_id = patient_dir.name

        # Construct paths to T2 and Dixon sequences
        t2_dir = patient_dir / args.t2_sequence_name
        dixon_dir = patient_dir / args.dixon_sequence_name

        # Check if both sequences exist
        if not t2_dir.exists():
            print(f"Warning: T2 sequence not found for {patient_id}: {t2_dir}")
            failed_patients.append((patient_id, "T2 directory not found"))
            continue

        if not dixon_dir.exists():
            print(f"Warning: Dixon sequence not found for {patient_id}: {dixon_dir}")
            failed_patients.append((patient_id, "Dixon directory not found"))
            continue

        try:
            process_patient(patient_id, t2_dir, dixon_dir, output_dir, segmenter)
            success_count += 1
        except Exception as e:
            print(f"\n✗ Error processing {patient_id}: {e}")
            import traceback
            traceback.print_exc()
            failed_patients.append((patient_id, str(e)))
            continue

    # Summary
    print(f"\n{'='*80}")
    print("PROCESSING SUMMARY")
    print(f"{'='*80}")
    print(f"Successfully processed: {success_count}/{len(patient_dirs)} patients")

    if failed_patients:
        print(f"\nFailed patients ({len(failed_patients)}):")
        for patient_id, reason in failed_patients:
            print(f"  - {patient_id}: {reason}")

    print(f"{'='*80}")


if __name__ == '__main__':
    main()
