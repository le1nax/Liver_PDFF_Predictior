# Fatty Liver Detection

This viewer automatically identifies patients with fatty liver based on mean fat fraction within the liver segmentation.

## How It Works

1. **Precomputation**: The system scans all patients and computes the mean fat fraction (FF) within the liver segmentation mask
2. **Threshold**: Patients with mean FF ≥ 10% are classified as having fatty liver
3. **YAML Storage**: Results are cached in `fatty_liver_patients.yaml` for fast loading

## Usage

### Automatic Detection

When you load a directory in the viewer:
1. The system checks for `fatty_liver_patients.yaml` in the directory
2. If found, it loads the cached data
3. If not found, it automatically runs the analysis (may take a few minutes)

### Filter Dropdown

Use the "Filter" dropdown to view:
- **All Patients**: Shows all patients in the directory
- **Fatty Liver Only**: Only shows patients with mean FF ≥ 10%
- **Normal Liver Only**: Only shows patients with mean FF < 10%

### Patient Information

The patient dropdown shows:
- Patient ID
- Number of slices
- Mean liver fat fraction percentage (e.g., "FF: 15.3%")

## Manual Precomputation

You can manually run the precomputation script:

```bash
cd /home/homesOnMaster/dgeiger/repos/T2Liver/preprocessed_viewer

# Basic usage (uses defaults)
python3 precompute_fatty_liver.py

# Custom input/output paths
python3 precompute_fatty_liver.py \
    --input-dir /path/to/preprocessed_images \
    --output /path/to/output.yaml

# Custom threshold (e.g., 5% instead of 10%)
python3 precompute_fatty_liver.py --threshold 0.05
```

## YAML File Format

The generated YAML file contains:

```yaml
threshold: 0.1  # Fatty liver threshold (10%)
patients:
  '0000016749':
    mean_ff: 0.152341
    mean_ff_percent: 15.23
    has_fatty_liver: true
  '0000020507':
    mean_ff: 0.045621
    mean_ff_percent: 4.56
    has_fatty_liver: false
```

## Fatty Liver Criteria

**Threshold**: Mean fat fraction ≥ 10% (0.1)

This threshold is based on clinical guidelines:
- **Normal liver**: < 5% fat content
- **Mild steatosis**: 5-10% fat content
- **Moderate steatosis**: 10-25% fat content
- **Severe steatosis**: > 25% fat content

The 10% threshold is commonly used to identify clinically significant hepatic steatosis.

## Performance

- **Initial precomputation**: ~1-3 seconds per patient (depends on image size)
- **Subsequent loads**: Instant (reads from cached YAML file)
- **Total time for 100 patients**: ~3-5 minutes

## Troubleshooting

### YAML File Not Generated

If the system fails to generate the YAML file:
1. Check that all patients have segmentation files (`*_segmentation.nii.gz`)
2. Verify that FF files are valid NIfTI format
3. Check console output for error messages

### Incorrect Classifications

If patients seem misclassified:
1. Verify segmentation quality (liver should be properly segmented)
2. Check that FF values are in 0-1 range (not 0-100)
3. Consider adjusting the threshold with `--threshold` parameter

### Regenerating YAML

To regenerate the YAML file:
1. Delete the existing `fatty_liver_patients.yaml` file
2. Reload the directory in the viewer
3. Or manually run `precompute_fatty_liver.py`

## Technical Details

### Mean FF Calculation

For each patient:
```python
# Load volumes
ff_volume = load_nifti(ff_file)
seg_volume = load_nifti(seg_file)

# Create liver mask (any segmentation value > 0)
liver_mask = seg_volume > 0

# Extract FF values in liver
ff_in_liver = ff_volume[liver_mask]

# Compute mean
mean_ff = np.mean(ff_in_liver)

# Classify
has_fatty_liver = (mean_ff >= 0.1)
```

### Color Mapping Enhancement

The viewer applies a square root transformation to enhance low FF values:
- 0-5% → More blue tones
- 5-10% → Cyan tones
- 10-15% → Green tones
- 15-20% → Yellow-green tones

This makes it easier to visually distinguish different levels of fat infiltration.
