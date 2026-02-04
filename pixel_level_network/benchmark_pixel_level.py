"""
Benchmark script for pixel-level fat fraction prediction.
Compares experiments 5, 6, and 7 on the test set using median liver FF.
"""

import argparse
import json
import yaml
import sys
import subprocess
from pathlib import Path
from typing import Any, Tuple

import torch
import numpy as np
import nibabel as nib
from tqdm import tqdm
from scipy.ndimage import binary_erosion, generate_binary_structure

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from pixel_level_network.model import get_model


def load_model(checkpoint_path: str, device: torch.device) -> Tuple[torch.nn.Module, dict]:
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Get model config from checkpoint
    config = checkpoint.get('config', {})
    model_config = config.get('model', {})

    model = get_model(
        model_type=model_config.get('type', 'standard'),
        n_channels=model_config.get('n_channels', 1),
        n_outputs=model_config.get('n_outputs', 1),
        base_channels=model_config.get('base_channels', 32),
        trilinear=model_config.get('trilinear', True)
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    return model, checkpoint.get('config', {})


def normalize_t2(t2_data: np.ndarray) -> np.ndarray:
    """Percentile-based T2 normalization."""
    p1, p99 = np.percentile(t2_data, [1, 99])
    t2_norm = np.clip(t2_data, p1, p99)
    t2_norm = (t2_norm - p1) / (p99 - p1 + 1e-8)
    return t2_norm


def predict_volume(model: torch.nn.Module, t2_data: np.ndarray, device: torch.device) -> np.ndarray:
    """Run inference on a single volume."""
    # Normalize
    t2_norm = normalize_t2(t2_data)

    # To tensor
    t2_tensor = torch.from_numpy(t2_norm.astype(np.float32)).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
    t2_tensor = t2_tensor.to(device)

    # Predict
    with torch.no_grad():
        pred = model(t2_tensor)

    # To numpy
    pred_np = pred.squeeze().cpu().numpy()
    return pred_np


def get_eroded_mask(mask_data: np.ndarray, iterations: int = 3) -> np.ndarray:
    """Erode binary mask."""
    mask_binary = (mask_data > 0).astype(np.float32)
    if iterations > 0:
        struct = generate_binary_structure(3, 1)
        mask_binary = binary_erosion(mask_binary, structure=struct, iterations=iterations).astype(np.float32)
    return mask_binary


def compute_median_ff(ff_data: np.ndarray, mask: np.ndarray) -> float:
    """Compute median fat fraction within mask."""
    masked_values = ff_data[mask > 0]
    if len(masked_values) == 0:
        return np.nan
    return np.median(masked_values)

def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    return obj


def main(data_dir: Path, splits_file: Path, output_dir: Path, experiments: dict[str, Path]) -> None:
    # Configuration
    data_dir = Path(data_dir)
    splits_file = Path(splits_file)
    output_dir = Path(output_dir)

    erosion_iterations = 3

    # Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test set
    with open(splits_file, 'r') as f:
        splits = json.load(f)
    test_ids = splits['test']
    print(f"Test set: {len(test_ids)} patients")

    # Results storage
    results = {exp: {'pred_medians': [], 'gt_medians': [], 'patient_ids': []} for exp in experiments}

    # Process each experiment
    for exp_name, checkpoint_path in experiments.items():
        print(f"\n{'='*60}")
        print(f"Benchmarking {exp_name}")
        print(f"{'='*60}")

        # Load model
        model, exp_config = load_model(str(checkpoint_path), device)
        print(f"Loaded checkpoint: {checkpoint_path}")
        results[exp_name]['config'] = _jsonify(exp_config)

        # Process each test patient
        for patient_id in tqdm(test_ids, desc=f"Processing {exp_name}"):
            patient_dir = data_dir / patient_id

            # File paths
            t2_path = patient_dir / f"{patient_id}_t2_aligned.nii.gz"
            ff_path = patient_dir / f"{patient_id}_ff_normalized.nii.gz"
            mask_path = patient_dir / f"{patient_id}_segmentation.nii.gz"

            # Check files exist
            if not all(p.exists() for p in [t2_path, ff_path, mask_path]):
                print(f"  Skipping {patient_id}: missing files")
                continue

            # Load data
            t2_data = nib.load(str(t2_path)).get_fdata().astype(np.float32)
            ff_data = nib.load(str(ff_path)).get_fdata().astype(np.float32)
            mask_data = nib.load(str(mask_path)).get_fdata().astype(np.float32)

            # Get masks: original for prediction, eroded for ground truth
            original_mask = (mask_data > 0).astype(np.float32)
            eroded_mask = get_eroded_mask(mask_data, iterations=erosion_iterations)

            # Skip if masks are empty
            if original_mask.sum() == 0:
                print(f"  Skipping {patient_id}: empty mask")
                continue
            if eroded_mask.sum() == 0:
                print(f"  Skipping {patient_id}: empty eroded mask")
                continue

            # Predict
            pred_ff = predict_volume(model, t2_data, device)

            # Compute median FF values
            # Prediction uses original (non-eroded) mask
            # Ground truth uses eroded mask (more reliable, avoids edge effects)
            pred_median = compute_median_ff(pred_ff, original_mask)
            gt_median = compute_median_ff(ff_data, eroded_mask)

            # Store results
            results[exp_name]['pred_medians'].append(pred_median)
            results[exp_name]['gt_medians'].append(gt_median)
            results[exp_name]['patient_ids'].append(patient_id)

    # Calculate and display metrics
    print(f"\n{'='*60}")
    print("BENCHMARK RESULTS")
    print(f"{'='*60}")

    for exp_name in experiments:
        pred_medians = np.array(results[exp_name]['pred_medians'])
        gt_medians = np.array(results[exp_name]['gt_medians'])

        # Remove NaN values
        valid_mask = ~(np.isnan(pred_medians) | np.isnan(gt_medians))
        pred_medians = pred_medians[valid_mask]
        gt_medians = gt_medians[valid_mask]

        if len(pred_medians) == 0:
            print(f"\n{exp_name}: No valid predictions")
            continue

        # Calculate MAE (in percentage points)
        mae = np.mean(np.abs(pred_medians - gt_medians)) * 100
        std = np.std(np.abs(pred_medians - gt_medians)) * 100

        # Additional metrics
        correlation = np.corrcoef(pred_medians, gt_medians)[0, 1]
        rmse = np.sqrt(np.mean((pred_medians - gt_medians) ** 2)) * 100

        print(f"\n{exp_name}:")
        print(f"  Patients evaluated: {len(pred_medians)}")
        print(f"  MAE:  {mae:.2f} +/- {std:.2f} percentage points")
        print(f"  RMSE: {rmse:.2f} percentage points")
        print(f"  Correlation: {correlation:.4f}")
        print(f"  GT median range: [{gt_medians.min()*100:.1f}%, {gt_medians.max()*100:.1f}%]")
        print(f"  Pred median range: [{pred_medians.min()*100:.1f}%, {pred_medians.max()*100:.1f}%]")

    # Save detailed results
    output_dir.mkdir(parents=True, exist_ok=True)
    results_file = output_dir / "benchmark_results_newdist.yaml"
    save_results = {}
    for exp_name in experiments:
        rows = list(zip(
            results[exp_name]['patient_ids'],
            results[exp_name]['gt_medians'],
            results[exp_name]['pred_medians'],
        ))
        rows.sort(key=lambda r: str(r[0]))
        patients = [
            {
                'patient_id': str(pid),
                'gt': float(gt),
                'pred': float(pred),
            }
            for pid, gt, pred in rows
        ]
        save_results[exp_name] = {
            'config': results[exp_name].get('config'),
            'patients': patients,
        }

    with open(results_file, 'w') as f:
        yaml.safe_dump(save_results, f, sort_keys=False)
    print(f"Detailed results saved to: {results_file}")

    eval_script = ROOT_DIR / "evaluate_benchmark.py"
    if eval_script.exists():
        print(f"Running evaluation: {eval_script} --results {results_file}")
        subprocess.run(
            [sys.executable, str(eval_script), "--results", str(results_file)],
            check=False,
        )
    else:
        print(f"Warning: evaluate_benchmark.py not found at {eval_script}")


if __name__ == "__main__":
    default_data_dir = Path("/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/datasets/preprocessed_images_fixed3")
    default_output_dir_base = Path("/home/homesOnMaster/dgeiger/repos/Liver_FF_Predictor/outputs/pixel_level_network")
    default_experiments = {
        "experiment_006": default_output_dir_base / "experiment_006" / "checkpoint_best.pth"
    }
    default_suffix = "_".join(default_experiments.keys())
    default_output_dir = default_output_dir_base / "benchmark_evaluation" / "" / default_suffix
    splits_file = default_data_dir / "data_splits.json" #_fatassigned.json"

    parser = argparse.ArgumentParser(description="Benchmark pixel-level fat fraction prediction.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=default_data_dir,
        help="Path to the preprocessed_images directory.",
    )
    parser.add_argument(
        "--splits-file",
        type=Path,
        default=splits_file,
        help="Path to the data splits JSON (defaults to data_dir/data_splits_fatassigned.json).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=default_output_dir,
        help="Directory for experiment checkpoints and output results.",
    )
    parser.add_argument(
        "--experiment",
        action="append",
        default=[],
        help="Experiment in the form name=/path/to/checkpoint_best.pth (repeatable).",
    )

    args = parser.parse_args()

    experiments = {}
    for item in args.experiment:
        if "=" not in item:
            raise ValueError(f"Invalid --experiment '{item}', expected name=/path/to/checkpoint")
        name, path = item.split("=", 1)
        experiments[name] = Path(path)

    if not experiments:
        experiments = default_experiments

    main(
        data_dir=args.data_dir,
        splits_file=args.splits_file,
        output_dir=args.output_dir,
        experiments=experiments,
    )
