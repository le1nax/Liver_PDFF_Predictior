# Preprocessed Data Viewer

A web-based viewer for inspecting preprocessed medical imaging data including aligned T2 images, normalized Fat Fraction images, and segmentation masks.

## Features

- **Automatic Directory Scanning**: Scans the preprocessed_images directory and loads all patients
- **Data Split Integration**: Automatically loads train/val/test splits from `data_splits.json`
  - Filter to show only test set patients for running inference
  - Prevents data leakage by identifying which patients were used for training
  - Displays data split (TRAIN/VAL/TEST) for each patient
- **Fatty Liver Detection**: Automatically identifies patients with fatty liver (≥10% mean FF in liver)
  - Filter dropdown to show: All patients, Test set only, Fatty liver only, or Normal liver only
  - Displays mean liver fat fraction percentage for each patient
  - Auto-generates and caches analysis results in YAML file
- **4-Panel View**: T2 Aligned, Fat Fraction (with Jet colormap), Segmentation, and T2+FF Overlay
- **Mask Overlays**: Toggle segmentation overlays on T2 and FF images
- **Adjustable Overlay Opacity**: Slider to control Fat Fraction overlay transparency on T2 image (0-100%)
- **Interactive Colorbar Legend**: Visual Jet colormap legend showing 0-100% Fat Fraction scale
- **Histogram Visualization**: Real-time histogram of Fat Fraction intensity distribution with statistics
- **Smart Preloading**: Caches ±5 slices for smooth navigation
- **Keyboard Navigation**: Arrow keys to navigate slices

## Installation

```bash
cd /home/homesOnMaster/dgeiger/repos/T2Liver/preprocessed_viewer

# Dependencies should already be installed from the main project
# If not, install:
pip install flask flask-cors numpy SimpleITK Pillow
```

## Usage

### 1. Start the Server

```bash
python3 backend.py
# OR
./start_viewer.sh
```

The server will start on `http://localhost:5040`

### 2. Open in Browser

Navigate to: `http://localhost:5040`

### 3. Load Preprocessed Data

The default path is already filled in:
```
/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/preprocessed_images
```

Click "Load Directory" to scan and load all patients.

**Progress Indicator**: When loading a directory for the first time, you'll see a progress bar showing:
- "Starting..." - Initial setup
- "Found X patients" - Directory scan complete
- "Computing mean fat fractions..." - Fatty liver analysis in progress
- "Loading patient data..." - Finalizing
- "Complete!" - Ready to use

Subsequent loads will be instant (uses cached YAML file).

## Directory Structure Expected

```
preprocessed_images/
├── 0000016749/
│   ├── 0000016749_t2_aligned.nii.gz
│   ├── 0000016749_ff_normalized.nii.gz
│   └── 0000016749_segmentation.nii.gz
├── 0000020507/
│   ├── 0000020507_t2_aligned.nii.gz
│   ├── 0000020507_ff_normalized.nii.gz
│   └── 0000020507_segmentation.nii.gz
└── ...
```

Each patient directory must contain exactly three files:
- `{patient_id}_t2_aligned.nii.gz` - Aligned T2 image
- `{patient_id}_ff_normalized.nii.gz` - Normalized Fat Fraction (0-1 range)
- `{patient_id}_segmentation.nii.gz` - Liver segmentation mask

## Features

### Visualization

- **T2 Images**: Grayscale with percentile normalization (1-99%)
- **Fat Fraction**: Jet colormap (Blue=0% → Cyan → Green → Yellow → Red=100%)
  - Matches the colormap used in the main T2Liver viewer
  - High contrast for easy visualization of fat distribution
- **Segmentation**: Cyan overlay with transparency
- **T2+FF Overlay**: T2 anatomical context with adjustable Fat Fraction overlay
  - Use the opacity slider to blend between pure T2 (0%) and equal T2+FF (100%)
  - Colorbar legend shows the Jet colormap scale for easy interpretation
- **Histogram**: Live intensity distribution plot for current slice
  - Shows Min, Max, Mean, Median, P25, P75 statistics
  - Histogram bars colored by the Jet colormap

### Navigation

- **Filter Dropdown**: Filter patients by various criteria
  - **All Patients**: Shows everyone
  - **Test Set Only (For Inference)**: Shows only patients in the test set (from `data_splits.json`)
    - These patients were NOT used for training and are suitable for model inference
    - Useful for running predictions without data leakage
  - **Fatty Liver Only (≥10%)**: Shows patients with ≥10% mean liver fat fraction
  - **Normal Liver Only (<10%)**: Shows patients with <10% mean liver fat fraction
- **Patient Dropdown**: Select any patient from the filtered list
  - Shows patient ID, number of slices, data split (TRAIN/VAL/TEST), and mean liver FF percentage
- **Manual Patient Input**: Type a patient ID directly and press Enter or click "Load"
  - Useful for quickly jumping to a specific patient without searching the dropdown
  - Example: Type "0000016749" and press Enter
- **Slice Slider**: Navigate through axial slices
- **Arrow Keys**: ← Previous, → Next slice
- **Checkboxes**: Toggle segmentation overlay on T2 and FF

### Performance

- **Lazy Loading**: Only loads individual slices on demand, not entire volumes
- **Fast Initialization**: Reads only NIfTI headers (not pixel data) when scanning directory
- **Smart Caching**: Caches loaded volumes and individual slices to avoid reloading
- **Preloading**: Frontend preloads ±5 slices for smooth navigation
- **Canvas Rendering**: Client-side colormap application for real-time updates

## Port Configuration

- **Port 5040**: Preprocessed Data Viewer
- **Port 5030**: Results Viewer (for model predictions)
- **Port 5020**: Main T2/FF Alignment Viewer

All three viewers can run simultaneously on different ports.

## Fatty Liver Detection

The viewer automatically identifies patients with fatty liver disease based on mean fat fraction within the liver segmentation.

### How it Works

1. **First Load**: When you load a directory for the first time, the system analyzes all patients
   - Computes mean fat fraction within liver segmentation mask for each patient
   - Classifies patients as fatty liver (≥10%) or normal (<10%)
   - Saves results to `fatty_liver_patients.yaml` in the directory

2. **Subsequent Loads**: The cached YAML file is instantly loaded

### Using the Filter

Use the "Filter" dropdown to view:
- **All Patients**: Shows all patients
- **Fatty Liver Only (≥10%)**: Shows only patients with clinically significant hepatic steatosis
- **Normal Liver Only (<10%)**: Shows patients with normal or minimal fat content

Each patient in the dropdown shows their mean liver FF percentage (e.g., "FF: 15.3%")

### Manual Precomputation

To manually run or regenerate the analysis:

```bash
cd /home/homesOnMaster/dgeiger/repos/T2Liver/preprocessed_viewer

# Run with default settings
python3 precompute_fatty_liver.py

# Custom threshold (e.g., 5% instead of 10%)
python3 precompute_fatty_liver.py --threshold 0.05

# Custom paths
python3 precompute_fatty_liver.py \
    --input-dir /path/to/preprocessed_images \
    --output /path/to/output.yaml
```

See [FATTY_LIVER_DETECTION.md](FATTY_LIVER_DETECTION.md) for detailed documentation.

## Troubleshooting

### Directory Not Found
- Verify the path exists and contains patient subdirectories
- Check that each patient folder has all three required files

### Images Not Displaying
- Verify NIfTI files are valid (not corrupted)
- Check browser console for errors
- Ensure backend server is running on port 5040

### Performance Issues
- Large datasets may take time to cache
- Reduce preload range if experiencing slowness
- Check available system memory

### Fatty Liver Analysis Not Running
- Ensure `precompute_fatty_liver.py` exists in the viewer directory
- Check that all patients have valid segmentation files
- Look for error messages in the backend console output
- Try manually running the precompute script

### Data Splits Not Loading
- Ensure `data_splits.json` exists in the preprocessed_images directory
- File should contain `train`, `val`, and `test` keys with patient ID lists
- Check backend console for "Loaded data splits" message
- If missing, the "Test Set Only" filter will show 0 patients

## Comparison with Other Viewers

| Viewer | Port | Purpose | Input |
|--------|------|---------|-------|
| Preprocessed Viewer | 5040 | Inspect preprocessed data | Directory of patient folders |
| Results Viewer | 5030 | Evaluate model predictions | YAML results file |
| Main Viewer | 5020 | Explore raw DICOM data | patient_data directory |
