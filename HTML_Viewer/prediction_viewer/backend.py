#!/usr/bin/env python3
"""
Preprocessed Data Viewer Backend
View preprocessed T2, Fat Fraction, and Segmentation images from a directory.
"""

import os
import sys
import numpy as np
import SimpleITK as sitk
from flask import Flask, jsonify, send_file, send_from_directory, request
from flask_cors import CORS
from pathlib import Path
from io import BytesIO
from PIL import Image
from scipy.ndimage import binary_erosion, generate_binary_structure
import matplotlib.cm as cm
from matplotlib.colors import LinearSegmentedColormap, to_rgb, rgb_to_hsv, hsv_to_rgb
import yaml
import subprocess
import json
import threading

app = Flask(__name__, static_folder='static', static_url_path='')
CORS(app)

# Global storage
PREPROCESSED_DIR = None
PATIENTS_DATA = []
VOLUME_CACHE = {}
PNG_CACHE = {}
FATTY_LIVER_DATA = None
DATA_SPLITS = None  # Store train/val/test splits
HEADER_CACHE = {}  # t2_file_path -> {'mtime': float, 'num_slices': int}
MASK_EROSION_ITERATIONS = 3


def _load_header_cache(base_dir):
    cache_path = Path(base_dir) / '.t2_header_cache.json'
    if not cache_path.exists():
        return cache_path, {}
    try:
        with open(cache_path, 'r') as f:
            data = json.load(f)
        if isinstance(data, dict):
            return cache_path, data
    except Exception as e:
        print(f"Warning: failed to read header cache: {e}")
    return cache_path, {}


def _save_header_cache(cache_path, cache_data):
    try:
        tmp_path = cache_path.with_suffix(cache_path.suffix + '.tmp')
        with open(tmp_path, 'w') as f:
            json.dump(cache_data, f)
        tmp_path.replace(cache_path)
    except Exception as e:
        print(f"Warning: failed to write header cache: {e}")

# Prediction / inference integration
PREDICTOR_CACHE = {}
PREDICTOR_LOCK = threading.Lock()
PREDICTION_JOBS = {}  # key -> {'status': 'idle'|'running'|'completed'|'error', 'output_path': str, 'error': str, 'experiment': str}
DEFAULT_PREDICTOR_CHECKPOINT = str(
    Path(__file__).resolve().parents[2]
    / 'outputs'
    / 'pixel_level_network'
    / 'experiment_001'
    / 'checkpoint_best.pth'
)

def load_or_generate_fatty_liver_yaml(base_dir):
    """
    Load or generate fatty liver patient data.
    Returns dict with fatty liver analysis results.
    """
    yaml_path = Path(base_dir) / 'fatty_liver_patients.yaml'

    # Check if YAML exists
    if yaml_path.exists():
        print(f"Loading existing fatty liver data from: {yaml_path}")
        try:
            with open(yaml_path, 'r') as f:
                data = yaml.safe_load(f)
            print(f"  Loaded data for {len(data.get('patients', {}))} patients")
            return data
        except Exception as e:
            print(f"Error loading YAML: {e}")
            print("Regenerating...")

    # Generate YAML if it doesn't exist
    print(f"Fatty liver YAML not found. Generating...")
    script_path = Path(__file__).parent / 'precompute_fatty_liver.py'

    if not script_path.exists():
        print(f"Error: precompute_fatty_liver.py not found at {script_path}")
        return {'patients': {}}

    try:
        # Run the precompute script
        result = subprocess.run(
            [sys.executable, str(script_path), '--input-dir', str(base_dir), '--output', str(yaml_path)],
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout
        )

        if result.returncode != 0:
            print(f"Error running precompute script:")
            print(result.stderr)
            return {'patients': {}}

        print(result.stdout)

        # Load the generated file
        with open(yaml_path, 'r') as f:
            data = yaml.safe_load(f)

        return data

    except subprocess.TimeoutExpired:
        print("Precompute script timed out")
        return {'patients': {}}
    except Exception as e:
        print(f"Error generating fatty liver data: {e}")
        return {'patients': {}}


def load_data_splits(base_dir):
    """Load train/val/test splits from the default split file."""
    splits_path = _default_split_path(base_dir)
    if not splits_path:
        print(f"Warning: no split file found under {base_dir}")
        return None
    return _load_splits_from_path(splits_path)


def _load_splits_from_path(splits_path: Path):
    if not splits_path or not splits_path.exists():
        return None
    try:
        with open(splits_path, 'r') as f:
            splits = json.load(f)
        print(f"Using split file: {splits_path}")
        print(
            "Loaded splits: "
            f"Train={len(splits.get('train', []))}, "
            f"Val={len(splits.get('val', []))}, "
            f"Test={len(splits.get('test', []))}"
        )
        return {
            'train': set(str(pid) for pid in splits.get('train', [])),
            'val': set(str(pid) for pid in splits.get('val', [])),
            'test': set(str(pid) for pid in splits.get('test', [])),
            'median_ff': {str(pid): val for pid, val in splits.get('median_ff', {}).items()},
        }
    except Exception as e:
        print(f"Error loading split file {splits_path}: {e}")
        return None


def _default_split_path(base_dir: str):
    base = Path(base_dir)
    fat_path = base / 'data_splits_fatassigned.json'
    if fat_path.exists():
        return fat_path
    std_path = base / 'data_splits.json'
    if std_path.exists():
        return std_path
    return None


def _experiment_split_path(experiment: str, base_dir: str):
    if not experiment:
        return None
    exp_path = Path(experiment)
    if exp_path.is_file():
        return exp_path
    if exp_path.is_dir():
        exp_dir = exp_path
    else:
        exp_dir = _outputs_dir() / experiment
    fat_path = exp_dir / 'data_splits_fatassigned.json'
    if fat_path.exists():
        return fat_path
    std_path = exp_dir / 'data_splits.json'
    if std_path.exists():
        return std_path
    return None


def _apply_splits_to_patients(splits):
    if not splits:
        return
    train = splits.get('train', set())
    val = splits.get('val', set())
    test = splits.get('test', set())
    median_ff = splits.get('median_ff', {})
    for patient in PATIENTS_DATA:
        pid = str(patient.get('patient_id'))
        if pid in train:
            patient['data_split'] = 'train'
        elif pid in val:
            patient['data_split'] = 'val'
        elif pid in test:
            patient['data_split'] = 'test'
        else:
            patient['data_split'] = 'unknown'
        if median_ff:
            patient['median_ff'] = median_ff.get(pid)




def scan_preprocessed_directory(base_dir):
    """Scan preprocessed directory and find all patients with their files"""
    patients = []

    base_path = Path(base_dir)
    if not base_path.exists():
        return patients

    # Load fatty liver data
    global FATTY_LIVER_DATA, DATA_SPLITS
    FATTY_LIVER_DATA = load_or_generate_fatty_liver_yaml(base_dir)
    DATA_SPLITS = load_data_splits(base_dir)

    # Load header cache from disk and merge into in-memory cache
    base_path = Path(base_dir)
    cache_path, disk_cache = _load_header_cache(base_dir)
    if disk_cache:
        HEADER_CACHE.update(disk_cache)
    cache_dirty = False

    # Iterate through patient directories
    for patient_dir in sorted(base_path.iterdir()):
        if not patient_dir.is_dir():
            continue

        patient_id = patient_dir.name

        # Look for the three expected files
        t2_file = patient_dir / f"{patient_id}_t2_aligned.nii.gz"
        ff_file = patient_dir / f"{patient_id}_ff_normalized.nii.gz"
        seg_file = patient_dir / f"{patient_id}_segmentation.nii.gz"

        if not t2_file.exists():
            t2_file = patient_dir / f"{patient_id}_t2_aligned.nii"
        if not ff_file.exists():
            ff_file = patient_dir / f"{patient_id}_ff_normalized.nii"
        if not seg_file.exists():
            seg_file = patient_dir / f"{patient_id}_segmentation.nii"

        # Only add patient if all three files exist
        if t2_file.exists() and ff_file.exists() and seg_file.exists():
            # Get number of slices from T2 (only read header, not pixel data)
            try:
                t2_path = str(t2_file)
                t2_mtime = t2_file.stat().st_mtime
                cached = HEADER_CACHE.get(t2_path)
                if cached and cached.get('mtime') == t2_mtime:
                    num_slices = cached['num_slices']
                else:
                    reader = sitk.ImageFileReader()
                    reader.SetFileName(t2_path)
                    reader.ReadImageInformation()
                    num_slices = reader.GetSize()[2]  # Z dimension
                    HEADER_CACHE[t2_path] = {'mtime': t2_mtime, 'num_slices': num_slices}
                    cache_dirty = True

                # Add fatty liver info if available
                patient_info = {
                    'patient_id': patient_id,
                    'num_slices': num_slices,
                    't2_file': str(t2_file),
                    'ff_file': str(ff_file),
                    'seg_file': str(seg_file)
                }

                # Add fatty liver metrics if available
                if patient_id in FATTY_LIVER_DATA.get('patients', {}):
                    fl_data = FATTY_LIVER_DATA['patients'][patient_id]
                    patient_info['mean_ff'] = fl_data.get('mean_ff')
                    patient_info['mean_ff_percent'] = fl_data.get('mean_ff_percent')
                    patient_info['has_fatty_liver'] = fl_data.get('has_fatty_liver', False)
                else:
                    patient_info['has_fatty_liver'] = None  # Unknown

                # Add data split information if available
                if DATA_SPLITS:
                    if patient_id in DATA_SPLITS['train']:
                        patient_info['data_split'] = 'train'
                    elif patient_id in DATA_SPLITS['val']:
                        patient_info['data_split'] = 'val'
                    elif patient_id in DATA_SPLITS['test']:
                        patient_info['data_split'] = 'test'
                    else:
                        patient_info['data_split'] = 'unknown'
                else:
                    patient_info['data_split'] = 'unknown'
                if DATA_SPLITS and 'median_ff' in DATA_SPLITS:
                    patient_info['median_ff'] = DATA_SPLITS['median_ff'].get(patient_id)

                # Check for existing prediction file (outputs/pixel_level_network/predictions)
                pred_file_default = _predictions_dir() / _default_experiment_name() / f"{patient_id}_prediction.nii.gz"
                pred_file_global = _predictions_dir() / f"{patient_id}_prediction.nii.gz"
                if pred_file_default.exists():
                    patient_info['prediction_file'] = str(pred_file_default)
                elif pred_file_global.exists():
                    patient_info['prediction_file'] = str(pred_file_global)
                else:
                    patient_info['prediction_file'] = None

                patients.append(patient_info)
            except Exception as e:
                print(f"Error loading {patient_id}: {e}")
                continue

    if cache_dirty:
        base_prefix = str(base_path) + os.sep
        cache_to_save = {
            path: info for path, info in HEADER_CACHE.items()
            if path.startswith(base_prefix)
        }
        _save_header_cache(cache_path, cache_to_save)

    return patients

def masked_histogram(values, mask, bins=50, value_range=(0.0, 1.0)):
    """
    values: 2D float array (FF or prediction)
    mask:   2D bool or {0,1} array
    """
    masked_vals = values[mask > 0]

    if masked_vals.size == 0:
        return None

    counts, bin_edges = np.histogram(
        masked_vals,
        bins=bins,
        range=value_range
    )

    stats = {
        "min": float(masked_vals.min()),
        "max": float(masked_vals.max()),
        "mean": float(masked_vals.mean()),
        "median": float(np.median(masked_vals)),
        "p25": float(np.percentile(masked_vals, 25)),
        "p75": float(np.percentile(masked_vals, 75)),
    }

    return {
        "counts": counts.tolist(),
        "bins": bin_edges.tolist(),
        "stats": stats
    }
    
def load_nifti_slice(nifti_path, slice_idx):
    """Load a single slice from NIfTI volume (more efficient than loading entire volume)"""
    cache_key = f"{nifti_path}_{slice_idx}"

    if cache_key in VOLUME_CACHE:
        return VOLUME_CACHE[cache_key]

    # Load entire volume (SimpleITK doesn't support lazy slice loading)
    # But we cache individual slices to avoid reloading
    volume_cache_key = f"{nifti_path}_volume"

    if volume_cache_key not in VOLUME_CACHE:
        sitk_image = sitk.ReadImage(str(nifti_path))
        volume = sitk.GetArrayFromImage(sitk_image)  # Returns (D, H, W)
        VOLUME_CACHE[volume_cache_key] = volume

    volume = VOLUME_CACHE[volume_cache_key]
    slice_data = volume[slice_idx]

    # Cache the slice
    VOLUME_CACHE[cache_key] = slice_data

    return slice_data


def normalize_for_display(image_slice, percentile_clip=True):
    """Normalize image slice to 0-255 range for display"""
    img = image_slice.copy().astype(np.float32)

    if percentile_clip:
        # Clip to 1st and 99th percentile for better contrast
        p1, p99 = np.percentile(img[img > 0], [1, 99]) if np.any(img > 0) else (img.min(), img.max())
        img = np.clip(img, p1, p99)

    # Normalize to 0-255
    img_min, img_max = img.min(), img.max()
    if img_max > img_min:
        img = ((img - img_min) / (img_max - img_min) * 255).astype(np.uint8)
    else:
        img = np.zeros_like(img, dtype=np.uint8)

    return img


def create_mask_overlay(mask_slice, color=(0, 255, 255, 128)):
    """Create colored overlay for segmentation mask"""
    # Create RGBA image
    h, w = mask_slice.shape
    overlay = np.zeros((h, w, 4), dtype=np.uint8)

    # Color the mask with specified RGBA
    mask_binary = mask_slice > 0
    overlay[mask_binary] = color

    return overlay


def get_eroded_mask_volume(seg_path):
    """Return eroded 3D mask volume using the same erosion as masked loss training."""
    volume_cache_key = f"{seg_path}_volume"
    eroded_cache_key = f"{seg_path}_eroded_{MASK_EROSION_ITERATIONS}"
    if eroded_cache_key in VOLUME_CACHE:
        return VOLUME_CACHE[eroded_cache_key]

    if volume_cache_key not in VOLUME_CACHE:
        sitk_image = sitk.ReadImage(str(seg_path))
        VOLUME_CACHE[volume_cache_key] = sitk.GetArrayFromImage(sitk_image)

    mask_volume = (VOLUME_CACHE[volume_cache_key] > 0).astype(np.float32)
    struct = generate_binary_structure(3, 1)
    eroded = binary_erosion(
        mask_volume,
        structure=struct,
        iterations=MASK_EROSION_ITERATIONS
    ).astype(np.float32)
    VOLUME_CACHE[eroded_cache_key] = eroded
    return eroded


CUSTOM_COLORMAP_VALUES = np.linspace(0, 1, 21)
CUSTOM_COLORMAP_COLORS = [
    "#00008B",  # 0.00 black
    "#00008B",  # 0.05 dark blue
    "#0000FF",  # 0.10 blue
    "#00CED1",  # 0.15 cyan
    "#008000",  # 0.20 green
    "#7CFC00",  # 0.25 light green
    "#FFFF00",  # 0.30 yellow
    "#FFD700",  # 0.35 gold
    "#FFA500",  # 0.40 orange
    "#F28C28",  # 0.45 dark orange
    "#E8742A",  # 0.50 orange-red
    "#DE5B2A",  # 0.55 red-orange
    "#D44227",  # 0.60 deep orange-red
    "#C92A22",  # 0.65 red
    "#B71B1B",  # 0.70 deep red
    "#A51214",  # 0.75 darker red
    "#930A0D",  # 0.80 very dark red
    "#8B0000",  # 0.85 dark red
    "#7A0000",  # 0.90 very dark red
    "#6A0000",  # 0.95 near-black red
    "#5A0000",  # 1.00 near-black red
]
HUE_COMPRESSION = 1.0
COLORMAP_VERSION = "distinct_0p05_hc1.0"


def _compress_hues(hex_colors, factor):
    rgb = np.array([to_rgb(c) for c in hex_colors], dtype=np.float32)
    hsv = rgb_to_hsv(rgb)

    angles = hsv[:, 0] * 2 * np.pi
    mean_angle = np.arctan2(np.sin(angles).mean(), np.cos(angles).mean())
    mean_hue = (mean_angle / (2 * np.pi)) % 1.0

    hue_delta = ((hsv[:, 0] - mean_hue + 0.5) % 1.0) - 0.5
    hsv[:, 0] = (mean_hue + hue_delta * factor) % 1.0

    return [tuple(hsv_to_rgb(h)) for h in hsv]


CUSTOM_COLORMAP = LinearSegmentedColormap.from_list(
    "distinct_0p05",
    list(zip(CUSTOM_COLORMAP_VALUES, _compress_hues(CUSTOM_COLORMAP_COLORS, HUE_COMPRESSION)))
)


def apply_continuous_colormap(normalized):
    """Map 0-1 values to the custom continuous colormap."""
    mapped = np.clip(normalized, 0, 1)
    colored = CUSTOM_COLORMAP(mapped)
    return (colored[:, :, :3] * 255).astype(np.uint8)


def create_colormap_legend(width=256, height=16):
    """Create a horizontal legend image for a colormap."""
    gradient = np.linspace(0, 1, width, dtype=np.float32)[None, :]
    rgb = apply_continuous_colormap(gradient)
    rgb = np.repeat(rgb, height, axis=0)
    img = Image.fromarray(rgb, mode='RGB')
    img_io = BytesIO()
    img.save(img_io, 'PNG')
    return img_io.getvalue()


@app.route('/')
def index():
    """Serve the frontend HTML"""
    # Serve index.html from the app's static folder so CWD doesn't matter.
    return send_from_directory(app.static_folder, 'index.html')


@app.route('/api/initialize', methods=['POST'])
def initialize():
    """Initialize the viewer with a preprocessed data directory"""
    global PREPROCESSED_DIR, PATIENTS_DATA, VOLUME_CACHE, PNG_CACHE

    data = request.json
    preprocessed_dir = data.get('preprocessed_dir')

    if not preprocessed_dir or not os.path.exists(preprocessed_dir):
        return jsonify({'error': 'Invalid preprocessed directory path'}), 400

    try:
        # Scan directory
        PREPROCESSED_DIR = preprocessed_dir
        PATIENTS_DATA = scan_preprocessed_directory(preprocessed_dir)

        # Clear caches
        VOLUME_CACHE = {}
        PNG_CACHE = {}

        print(f"Found {len(PATIENTS_DATA)} patients in {preprocessed_dir}")

        return jsonify({
            'success': True,
            'patients': PATIENTS_DATA,
            'total_patients': len(PATIENTS_DATA)
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/patients')
def get_patients():
    """Return current patient list (including split assignments)."""
    if PREPROCESSED_DIR is None:
        return jsonify({'error': 'Viewer not initialized with preprocessed_dir'}), 400
    return jsonify({
        'patients': PATIENTS_DATA,
        'total_patients': len(PATIENTS_DATA)
    })


@app.route('/api/initialize/progress')
def initialize_progress():
    """Get initialization progress (for fatty liver computation)"""
    import json
    from flask import Response
    import time

    def generate():
        """Stream progress updates"""
        preprocessed_dir = request.args.get('dir')

        if not preprocessed_dir or not os.path.exists(preprocessed_dir):
            yield f"data: {json.dumps({'error': 'Invalid directory'})}\n\n"
            return

        yaml_path = Path(preprocessed_dir) / 'fatty_liver_patients.yaml'

        # Check if YAML already exists
        if yaml_path.exists():
            yield f"data: {json.dumps({'stage': 'complete', 'message': 'Loading cached data...', 'progress': 100})}\n\n"
            return

        # Need to generate - send progress updates
        yield f"data: {json.dumps({'stage': 'init', 'message': 'Starting fatty liver analysis...', 'progress': 0})}\n\n"
        sys.stdout.flush()
        time.sleep(0.1)

        # Count patient directories
        base_path = Path(preprocessed_dir)
        patient_dirs = [d for d in base_path.iterdir() if d.is_dir()]
        total = len(patient_dirs)

        yield f"data: {json.dumps({'stage': 'scanning', 'message': f'Found {total} patients', 'progress': 10})}\n\n"
        sys.stdout.flush()
        time.sleep(0.1)

        # Start the subprocess to generate YAML
        yield f"data: {json.dumps({'stage': 'analyzing', 'message': 'Computing mean fat fractions...', 'progress': 20})}\n\n"
        sys.stdout.flush()

        script_path = Path(__file__).parent / 'precompute_fatty_liver.py'

        if not script_path.exists():
            yield f"data: {json.dumps({'error': 'Precompute script not found'})}\n\n"
            return

        # Start subprocess
        import subprocess
        proc = subprocess.Popen(
            [sys.executable, str(script_path), '--input-dir', str(preprocessed_dir), '--output', str(yaml_path)],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )

        # Monitor progress
        max_wait = 300  # 5 minutes max
        waited = 0
        while waited < max_wait:
            # Check if process finished
            retcode = proc.poll()
            if retcode is not None:
                # Process finished
                if retcode == 0 and yaml_path.exists():
                    yield f"data: {json.dumps({'stage': 'complete', 'message': 'Analysis complete!', 'progress': 100})}\n\n"
                    sys.stdout.flush()
                else:
                    stderr = proc.stderr.read() if proc.stderr else ""
                    yield f"data: {json.dumps({'stage': 'error', 'message': 'Analysis failed', 'progress': 50})}\n\n"
                    sys.stdout.flush()
                return

            # Still running - update progress
            time.sleep(1)
            waited += 1

            # Estimate progress (20-90% during analysis)
            progress = 20 + min(int((waited / max_wait) * 70), 70)

            # Show elapsed time
            if waited % 5 == 0:  # Update every 5 seconds
                yield f"data: {json.dumps({'stage': 'analyzing', 'message': f'Analyzing patients... ({waited}s elapsed)', 'progress': progress})}\n\n"
                sys.stdout.flush()

        # Timeout
        proc.kill()
        yield f"data: {json.dumps({'stage': 'timeout', 'message': 'Analysis timeout - continuing anyway', 'progress': 90})}\n\n"
        sys.stdout.flush()

    return Response(generate(), mimetype='text/event-stream')


@app.route('/api/patient/<patient_id>/info')
def get_patient_info(patient_id):
    """Get patient information"""
    for patient in PATIENTS_DATA:
        if patient['patient_id'] == patient_id:
            has_pred = 'prediction_file' in patient
            print(f"[INFO] Patient {patient_id} - has prediction_file: {has_pred}")
            if has_pred:
                print(f"[INFO] Prediction path: {patient['prediction_file']}")
            return jsonify(patient)

    return jsonify({'error': 'Patient not found'}), 404

@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/ff_histogram')
def get_ff_histogram(patient_id, slice_idx):
    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    # Load raw FF slice (values in 0–1)
    ff_slice = load_nifti_slice(patient['ff_file'], slice_idx)

    # Load segmentation mask (liver only)
    seg_slice = load_nifti_slice(patient['seg_file'], slice_idx)
    liver_mask = seg_slice > 0

    # Extract valid FF values
    values = ff_slice[liver_mask]
    values = np.clip(values, 0, 1)

    if values.size == 0:
        return jsonify({'bins': [], 'counts': [], 'stats': None})

    # Histogram (0–100%)
    bins = np.linspace(0, 1, 51)
    counts, _ = np.histogram(values, bins=bins)

    stats = {
        'min': float(values.min() * 100),
        'max': float(values.max() * 100),
        'mean': float(values.mean() * 100),
        'median': float(np.median(values) * 100),
        'p25': float(np.percentile(values, 25) * 100),
        'p75': float(np.percentile(values, 75) * 100),
        'num_pixels': int(values.size)
    }

    return jsonify({
        'bins': (bins * 100).tolist(),   # percent
        'counts': counts.tolist(),
        'stats': stats
    })

@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/t2')
def get_t2_slice(patient_id, slice_idx):
    """Get T2 image slice as PNG"""
    cache_key = f"{patient_id}_t2_{slice_idx}"
    if cache_key in PNG_CACHE:
        buf = BytesIO(PNG_CACHE[cache_key])
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    # Get and normalize slice
    slice_data = load_nifti_slice(patient['t2_file'], slice_idx)
    slice_normalized = normalize_for_display(slice_data)

    # Convert to PNG
    img = Image.fromarray(slice_normalized, mode='L')
    img_io = BytesIO()
    img.save(img_io, 'PNG')

    # Cache the bytes
    PNG_CACHE[cache_key] = img_io.getvalue()

    # Return a new BytesIO from the cached bytes
    buf = BytesIO(PNG_CACHE[cache_key])
    buf.seek(0)
    return send_file(buf, mimetype='image/png')

@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/prediction_histogram')
def get_prediction_histogram(patient_id, slice_idx):
    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    experiment = request.args.get('experiment')
    pred_path = _resolve_prediction_path(patient_id, experiment)

    if not pred_path.exists():
        return jsonify({'error': 'Prediction not available'}), 404

    # Load prediction slice
    pred_slice = load_nifti_slice(str(pred_path), slice_idx)

    # Load segmentation mask
    seg_slice = load_nifti_slice(patient['seg_file'], slice_idx)
    liver_mask = seg_slice > 0

    # Extract values inside liver
    values = pred_slice[liver_mask]
    values = np.clip(values, 0, 1)

    if values.size == 0:
        return jsonify({'bins': [], 'counts': [], 'stats': None})

    # Histogram (0–100%)
    bins = np.linspace(0, 1, 51)
    counts, _ = np.histogram(values, bins=bins)

    stats = {
        'min': float(values.min() * 100),
        'max': float(values.max() * 100),
        'mean': float(values.mean() * 100),
        'median': float(np.median(values) * 100),
        'p25': float(np.percentile(values, 25) * 100),
        'p75': float(np.percentile(values, 75) * 100),
        'num_pixels': int(values.size)
    }

    return jsonify({
        'bins': (bins * 100).tolist(),
        'counts': counts.tolist(),
        'stats': stats
    })
    
@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/ff')
def get_ff_slice(patient_id, slice_idx):
    """Get Fat Fraction slice as PNG with Jet colormap (enhanced low-value discrimination)"""
    cache_key = f"{patient_id}_ff_{slice_idx}_{COLORMAP_VERSION}"
    if cache_key in PNG_CACHE:
        buf = BytesIO(PNG_CACHE[cache_key])
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    # Get slice (FF values are typically 0-1)
    slice_data = load_nifti_slice(patient['ff_file'], slice_idx)

    # Normalize to 0-1 range (clamp to ensure valid range)
    normalized = np.clip(slice_data, 0, 1)

    # Apply custom continuous colormap
    rgb = apply_continuous_colormap(normalized)

    # Convert to PNG
    img = Image.fromarray(rgb, mode='RGB')
    img_io = BytesIO()
    img.save(img_io, 'PNG')

    # Cache the bytes
    PNG_CACHE[cache_key] = img_io.getvalue()

    # Return a new BytesIO from the cached bytes
    buf = BytesIO(PNG_CACHE[cache_key])
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/segmentation')
def get_segmentation_slice(patient_id, slice_idx):
    """Get segmentation mask overlay as PNG with transparency"""
    cache_key = f"{patient_id}_seg_{slice_idx}"
    if cache_key in PNG_CACHE:
        buf = BytesIO(PNG_CACHE[cache_key])
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    # Get mask slice and create overlay
    slice_data = load_nifti_slice(patient['seg_file'], slice_idx)
    overlay = create_mask_overlay(slice_data)

    # Convert to PNG with transparency
    img = Image.fromarray(overlay, mode='RGBA')
    img_io = BytesIO()
    img.save(img_io, 'PNG')

    # Cache the bytes
    PNG_CACHE[cache_key] = img_io.getvalue()

    # Return a new BytesIO from the cached bytes
    buf = BytesIO(PNG_CACHE[cache_key])
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/segmentation_eroded')
def get_eroded_segmentation_slice(patient_id, slice_idx):
    """Get eroded segmentation mask overlay as PNG with transparency."""
    cache_key = f"{patient_id}_seg_eroded_{slice_idx}_{MASK_EROSION_ITERATIONS}"
    if cache_key in PNG_CACHE:
        buf = BytesIO(PNG_CACHE[cache_key])
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    patient = next((p for p in PATIENTS_DATA if p['patient_id'] == patient_id), None)
    if not patient:
        return jsonify({'error': 'Patient not found'}), 404

    if slice_idx < 0 or slice_idx >= patient['num_slices']:
        return jsonify({'error': 'Invalid slice index'}), 400

    eroded_volume = get_eroded_mask_volume(patient['seg_file'])
    slice_data = eroded_volume[slice_idx]

    overlay = create_mask_overlay(slice_data, color=(255, 165, 0, 128))
    img = Image.fromarray(overlay, mode='RGBA')

    img_io = BytesIO()
    img.save(img_io, 'PNG')
    PNG_CACHE[cache_key] = img_io.getvalue()

    buf = BytesIO(PNG_CACHE[cache_key])
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/legend/rainbow')
def get_rainbow_legend():
    """Get a PNG legend for the rainbow colormap."""
    try:
        width = int(request.args.get('width', 256))
        height = int(request.args.get('height', 16))
    except ValueError:
        width, height = 256, 16

    width = max(32, min(width, 1024))
    height = max(8, min(height, 64))

    png_bytes = create_colormap_legend(width=width, height=height)
    buf = BytesIO(png_bytes)
    buf.seek(0)
    return send_file(buf, mimetype='image/png')


@app.route('/api/prediction/splits', methods=['POST'])
def prediction_splits():
    """Switch split mapping based on selected experiment."""
    global DATA_SPLITS
    data = request.json or {}
    experiment = data.get('experiment')
    if not PREPROCESSED_DIR:
        return jsonify({'error': 'Viewer not initialized with preprocessed_dir'}), 400

    split_path = _experiment_split_path(experiment, PREPROCESSED_DIR) if experiment else None
    used_path = split_path
    splits = _load_splits_from_path(split_path) if split_path else None
    if splits is None:
        default_path = _default_split_path(PREPROCESSED_DIR)
        used_path = default_path
        splits = _load_splits_from_path(default_path) if default_path else None

    if splits is None:
        return jsonify({
            'error': 'No valid split file found',
            'experiment': experiment,
            'split_path': str(used_path) if used_path else None,
        }), 404

    DATA_SPLITS = splits
    _apply_splits_to_patients(splits)

    return jsonify({
        'experiment': experiment,
        'split_path': str(used_path) if used_path else None,
        'counts': {
            'train': len(splits.get('train', [])),
            'val': len(splits.get('val', [])),
            'test': len(splits.get('test', [])),
        }
    })


@app.route('/api/prediction/experiments')
def prediction_experiments():
    outputs_dir = _outputs_dir()
    experiments = []
    if outputs_dir.exists():
        for exp_dir in sorted(outputs_dir.iterdir()):
            if not exp_dir.is_dir():
                continue
            if exp_dir.name == 'predictions':
                continue
            checkpoint = exp_dir / 'checkpoint_best.pth'
            if checkpoint.exists():
                experiments.append({
                    'name': exp_dir.name,
                    'checkpoint': str(checkpoint)
                })

    default_name = _default_experiment_name()
    return jsonify({
        'experiments': experiments,
        'default_experiment': default_name
    })


@app.route('/api/prediction/available')
def prediction_available():
    patient_id = request.args.get('patient_id')
    experiment = request.args.get('experiment')
    if not patient_id or not PREPROCESSED_DIR:
        return jsonify({'available': False}), 400

    pred_path = _resolve_prediction_path(patient_id, experiment)
    return jsonify({
        'available': pred_path.exists(),
        'prediction_path': str(pred_path),
        'experiment': experiment
    })


# ----------------------
# Prediction / Inference Endpoints
# ----------------------

def _import_predictor_module():
    """Dynamically import the predictor module and return FatFractionPredictor class."""
    predictor_root = Path(__file__).resolve().parents[2]
    if str(predictor_root) not in sys.path:
        sys.path.insert(0, str(predictor_root))

    try:
        from pixel_level_network.inference import FatFractionPredictor  # type: ignore
        return FatFractionPredictor
    except Exception as e:
        print(f"Error importing FatFractionPredictor: {e}")
        print(f"Attempted to import from: {predictor_root}")
        return None


def _outputs_dir():
    return Path(__file__).resolve().parents[2] / 'outputs' / 'pixel_level_network'


def _predictions_dir():
    return _outputs_dir() / 'predictions'


def _default_checkpoint_path():
    outputs_dir = _outputs_dir()
    candidates = []
    if outputs_dir.exists():
        for exp_dir in outputs_dir.iterdir():
            if not exp_dir.is_dir() or exp_dir.name == 'predictions':
                continue
            checkpoint = exp_dir / 'checkpoint_best.pth'
            if checkpoint.exists():
                candidates.append(checkpoint)
    if candidates:
        latest = max(candidates, key=lambda p: p.stat().st_mtime)
        return str(latest)
    return DEFAULT_PREDICTOR_CHECKPOINT


def _default_experiment_name():
    checkpoint = Path(_default_checkpoint_path())
    return checkpoint.parent.name


def _get_experiment_checkpoint(experiment):
    if not experiment:
        return _default_checkpoint_path()

    exp_path = Path(experiment)
    if exp_path.is_file():
        return str(exp_path)
    if exp_path.is_dir():
        checkpoint = exp_path / 'checkpoint_best.pth'
        return str(checkpoint)

    outputs_dir = _outputs_dir()
    checkpoint = outputs_dir / experiment / 'checkpoint_best.pth'
    return str(checkpoint)


def _resolve_prediction_path(patient_id, experiment):
    if experiment:
        pred_path = _predictions_dir() / experiment / f"{patient_id}_prediction.nii.gz"
        return pred_path

    pred_file = None
    for p in PATIENTS_DATA:
        if p.get('patient_id') == patient_id:
            pred_file = p.get('prediction_file')
            break

    if pred_file:
        return Path(pred_file)

    return _predictions_dir() / f"{patient_id}_prediction.nii.gz"


def _get_predictor(checkpoint_path=None, device=None):
    """Return a cached predictor instance per checkpoint path."""
    checkpoint = checkpoint_path or _default_checkpoint_path()
    with PREDICTOR_LOCK:
        predictor = PREDICTOR_CACHE.get(checkpoint)
        if predictor is None:
            FatFractionPredictor = _import_predictor_module()
            if FatFractionPredictor is None:
                raise ImportError("FatFractionPredictor not available")
            predictor = FatFractionPredictor(checkpoint_path=checkpoint, config_path=None, device=device or 'cuda')
            PREDICTOR_CACHE[checkpoint] = predictor
        return predictor


def _run_prediction_job(job_key, patient_id, output_path, checkpoint_path=None, experiment=None):
    """Run predictor.predict in a background thread and update job status."""
    global PREDICTION_JOBS
    try:
        predictor = _get_predictor(checkpoint_path=checkpoint_path)

        # Update job status (preserve t2_path from initialization)
        t2_path = PREDICTION_JOBS[job_key].get('t2_path', '')
        PREDICTION_JOBS[job_key]['status'] = 'running'
        PREDICTION_JOBS[job_key]['error'] = None

        predictor.predict(t2_path=t2_path, output_path=output_path, visualize=False)

        # After completion, mark completed
        PREDICTION_JOBS[job_key]['status'] = 'completed'
        PREDICTION_JOBS[job_key]['output_path'] = output_path

        # Update PATIENTS_DATA entry if present
        updated = False
        for p in PATIENTS_DATA:
            if p.get('patient_id') == patient_id:
                p['prediction_file'] = output_path
                print(f"[PREDICTION] Updated PATIENTS_DATA for {patient_id} with prediction_file: {output_path} (exp={experiment})")
                updated = True
                break

        if not updated:
            print(f"[PREDICTION] WARNING: Patient {patient_id} not found in PATIENTS_DATA to update")
    except Exception as e:
        PREDICTION_JOBS[job_key]['status'] = 'error'
        PREDICTION_JOBS[job_key]['error'] = str(e)
        print(f"Prediction error for {patient_id}: {e}")


@app.route('/api/prediction/status')
def prediction_status():
    """Return overall prediction capability and per-patient job status"""
    patient_id = request.args.get('patient_id')
    experiment = request.args.get('experiment')

    status = {
        'predictor_checkpoint': _default_checkpoint_path(),
        'predictor_available': False,
        'cuda_available': False,
        'jobs': {}
    }

    # Check checkpoint
    status['predictor_available'] = os.path.exists(status['predictor_checkpoint'])

    # Check CUDA
    try:
        import torch
        status['cuda_available'] = torch.cuda.is_available()
    except Exception:
        status['cuda_available'] = False

    # Add job info if requested
    if patient_id:
        if experiment:
            job_key = f"{patient_id}:{experiment}"
            job = PREDICTION_JOBS.get(job_key, None)
            status['jobs'][job_key] = job
        else:
            status['jobs'] = {
                key: value for key, value in PREDICTION_JOBS.items()
                if key.startswith(f"{patient_id}:")
            }
        return jsonify(status)

    # Otherwise include all jobs
    status['jobs'] = PREDICTION_JOBS
    return jsonify(status)


@app.route('/api/prediction/generate', methods=['POST'])
def generate_prediction():
    """Start prediction for a given patient (async)."""
    data = request.json or {}
    patient_id = data.get('patient_id')
    experiment = data.get('experiment')
    checkpoint = data.get('checkpoint') or _get_experiment_checkpoint(experiment)

    if not patient_id:
        return jsonify({'error': 'patient_id is required'}), 400

    if PREPROCESSED_DIR is None:
        return jsonify({'error': 'Viewer not initialized with preprocessed_dir'}), 400

    # Find patient directory
    patient_dir = Path(PREPROCESSED_DIR) / patient_id
    if not patient_dir.exists():
        return jsonify({'error': 'Patient not found'}), 404

    # Ensure t2 file exists
    t2_file = patient_dir / f"{patient_id}_t2_aligned.nii.gz"
    if not t2_file.exists():
        return jsonify({'error': 'T2 file not found for patient'}), 404

    if not os.path.exists(checkpoint):
        return jsonify({'error': f'Checkpoint not found: {checkpoint}'}), 400

    # Prepare output dir (per-experiment predictions folder)
    experiment_name = experiment or Path(checkpoint).resolve().parent.name
    out_dir = _predictions_dir() / experiment_name
    out_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(out_dir / f"{patient_id}_prediction.nii.gz")

    # Prevent double-running
    job_key = f"{patient_id}:{experiment_name}"
    job = PREDICTION_JOBS.get(job_key)
    if job and job.get('status') == 'running':
        return jsonify({'error': 'Prediction already running for this patient'}), 409

    # Initialize job entry
    PREDICTION_JOBS[job_key] = {
        'status': 'queued',
        'output_path': output_path,
        'error': None,
        't2_path': str(t2_file),
        'experiment': experiment_name
    }

    # Start background thread
    thread = threading.Thread(
        target=_run_prediction_job,
        args=(job_key, patient_id, output_path, checkpoint, experiment_name),
        daemon=True
    )
    thread.start()

    return jsonify({'success': True, 'patient_id': patient_id, 'status': 'queued', 'output_path': output_path, 'experiment': experiment_name}), 202


@app.route('/api/patient/<patient_id>/slice/<int:slice_idx>/prediction')
def get_prediction_slice(patient_id, slice_idx):
    """Get predicted fat fraction slice as PNG (Jet colormap with sqrt enhancement)"""
    print(f"[PREDICTION] Request for patient {patient_id}, slice {slice_idx}")

    experiment = request.args.get('experiment')
    cache_key = f"{patient_id}_prediction_{slice_idx}_{COLORMAP_VERSION}_{experiment or 'default'}"
    if cache_key in PNG_CACHE:
        buf = BytesIO(PNG_CACHE[cache_key])
        buf.seek(0)
        return send_file(buf, mimetype='image/png')

    # Find patient
    patient = None
    for p in PATIENTS_DATA:
        if p['patient_id'] == patient_id:
            patient = p
            break

    if not patient:
        print(f"[PREDICTION] ERROR: Patient {patient_id} not found in PATIENTS_DATA")
        return jsonify({'error': 'Patient not found'}), 404

    # Prediction file location
    pred_path = _resolve_prediction_path(patient_id, experiment)
    print(f"[PREDICTION] Using prediction path: {pred_path}")

    if not pred_path.exists():
        print(f"[PREDICTION] ERROR: Prediction file does not exist: {pred_path}")
        return jsonify({'error': 'Prediction not available'}), 404

    print(f"[PREDICTION] Loading prediction from: {pred_path}")

    # Load prediction slice
    slice_data = load_nifti_slice(str(pred_path), slice_idx)

    # Load corresponding segmentation mask to mask out predictions outside liver
    seg_slice = load_nifti_slice(patient['seg_file'], slice_idx)
    liver_mask = seg_slice > 0

    # Apply mask: only keep predictions inside liver
    masked_prediction = slice_data.copy()
    masked_prediction[~liver_mask] = 0

    # Normalize to 0-1 range (clamp to ensure valid range)
    normalized = np.clip(masked_prediction, 0, 1)

    # Apply custom continuous colormap
    rgb = apply_continuous_colormap(normalized)

    img = Image.fromarray(rgb, mode='RGB')
    img_io = BytesIO()
    img.save(img_io, 'PNG')

    # Cache and return
    PNG_CACHE[cache_key] = img_io.getvalue()
    buf = BytesIO(PNG_CACHE[cache_key])
    buf.seek(0)
    response = send_file(buf, mimetype='image/png')
    # Ensure CORS headers are set for image responses
    response.headers['Access-Control-Allow-Origin'] = '*'
    return response


if __name__ == '__main__':
    print("=" * 80)
    print("Preprocessed Data Viewer Backend")
    print("=" * 80)
    print("Starting server on http://localhost:5040")
    print("Open the frontend to load a preprocessed data directory")
    print("=" * 80)

    app.run(host='0.0.0.0', port=5040, debug=True, threaded=True)
