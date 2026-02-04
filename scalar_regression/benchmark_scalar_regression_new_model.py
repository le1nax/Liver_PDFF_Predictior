#!/usr/bin/env python3
"""
Benchmark script for scalar regression fat fraction prediction.
Generates output in the same format as benchmark_pixel_level.py.
"""

import argparse
import json
import yaml
import subprocess
import sys
from pathlib import Path
from typing import Any, List

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_data_splits, load_data_splits
from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar
from scalar_regression.model_scalar import get_scalar_model


def build_arg_parser(
    default_experiments: list[str],
    default_checkpoint_name: str,
    default_batch_size: int,
    default_num_workers: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark scalar regression experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=default_experiments,
        help="Experiment directories containing checkpoint_best.pth",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=default_checkpoint_name,
        help="Checkpoint filename inside each experiment directory",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save benchmark results JSON",
    )
    parser.add_argument("--batch-size", type=int, default=default_batch_size, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=default_num_workers, help="DataLoader workers")
    return parser


def resolve_data_dir(config: dict) -> Path:
    data_cfg = config.get("data", {})
    data_dir = data_cfg.get("data_dir") or config.get("data_dir")
    if not data_dir:
        raise KeyError("data_dir not found in checkpoint config")
    data_dir = Path(data_dir)
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()
    return data_dir


def resolve_experiment_dir(exp_arg: str) -> Path:
    candidates = [Path(exp_arg), REPO_ROOT / exp_arg]
    if not exp_arg.startswith("scalar_regression/"):
        candidates.append(REPO_ROOT / "scalar_regression" / exp_arg)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Experiment directory not found. Tried: {tried}")


def resolve_split_file(exp_dir: Path, data_dir: Path) -> Path | None:
    candidates = [
        exp_dir / 'data_splits_fatassigned.json',
        exp_dir / 'data_splits.json',
        data_dir / 'data_splits_fatassigned.json',
        data_dir / 'data_splits.json',
    ]
    for c in candidates:
        if c.exists():
            return c
    return None



def load_test_ids(data_dir: Path, data_cfg: dict, exp_dir: Path) -> List[str]:
    splits_file = resolve_split_file(exp_dir, data_dir)
    if splits_file is not None and splits_file.exists():
        # load using helper if stored in experiment dir
        if splits_file.name == 'data_splits_fatassigned.json':
            with open(splits_file, 'r') as f:
                splits = json.load(f)
            return splits.get('test', [])
        else:
            # fall back to standard loader for data_splits.json in data_dir
            _, _, test_ids = load_data_splits(str(data_dir))
            return test_ids

    return create_data_splits(
        str(data_dir),
        train_ratio=data_cfg.get("train_ratio", 0.7),
        val_ratio=data_cfg.get("val_ratio", 0.15),
        test_ratio=data_cfg.get("test_ratio", 0.15),
        random_seed=data_cfg.get("random_seed", 42),
        save_splits=True,
        use_subdirs=data_cfg.get("use_subdirs", False),
        use_patient_subdirs=data_cfg.get("use_patient_subdirs", False),
        t2_suffix=data_cfg.get("t2_suffix", "_t2_aligned"),
    )[2]


def build_dataset(data_dir: Path, test_ids: List[str], data_cfg: dict) -> LiverFatScalarDataset:
    return LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=test_ids,
        t2_suffix=data_cfg.get("t2_suffix", "_t2_aligned"),
        ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
        mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
        input_mask_suffix=data_cfg.get("input_mask_suffix", "_t2_original_segmentation"),
        use_subdirs=data_cfg.get("use_subdirs", False),
        use_patient_subdirs=data_cfg.get("use_patient_subdirs", True),
        t2_subdir=data_cfg.get("t2_subdir", "t2_images"),
        ff_subdir=data_cfg.get("ff_subdir", "fat_fraction_maps"),
        mask_subdir=data_cfg.get("mask_subdir", "liver_masks"),
        normalize_t2=data_cfg.get("normalize_t2", False),
        normalize_ff=data_cfg.get("normalize_ff", True),
        mask_erosion=data_cfg.get("mask_erosion", 3),
        augment=False,
        validate_files=True,
    )


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint.get("config", {}).get("model", {})
    base_channels = model_config.get("base_channels", 16)
    model = get_scalar_model(in_channels=model_config.get("in_channels", 2), base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    return obj


def main() -> None:
    default_checkpoint_name = "checkpoint_best.pth"
    outputs_dir = REPO_ROOT / "outputs" / "scalar_regression_run"
    default_experiments = [
        "outputs/scalar_regression_run/experiment_20260129_145820",
        "outputs/scalar_regression_run/experiment_20260129_150141",
        "outputs/scalar_regression_run/experiment_20260129_150431",
        "outputs/scalar_regression_run/experiment_20260129_150545"

    ]
    default_batch_size = 1
    default_num_workers = 2

    args = build_arg_parser(
        default_experiments=default_experiments,
        default_checkpoint_name=default_checkpoint_name,
        default_batch_size=default_batch_size,
        default_num_workers=default_num_workers,
    ).parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    experiments = [resolve_experiment_dir(p) for p in args.experiments]
    if args.output is None:
        suffix = "_".join(exp.name for exp in experiments)
        output_path = outputs_dir / "benchmark_evaluation" / suffix / "benchmark_results_scalar_regression.yaml"
    else:
        output_path = Path(args.output)

    for exp_dir in experiments:
        exp_name = exp_dir.name
        checkpoint_path = exp_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        data_cfg = config.get("data", {})
        data_dir = resolve_data_dir(config)

        test_ids = load_test_ids(data_dir, data_cfg, exp_dir)
        dataset = build_dataset(data_dir, test_ids, data_cfg)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate_scalar,
        )

        model = build_model(checkpoint, device)
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {exp_name}")
        print(f"{'=' * 60}")
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Test set: {len(dataset)} patients")

        pred_medians = []
        gt_medians = []
        patient_ids = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Processing {exp_name}"):
                t2 = batch["t2"].to(device)
                mask = batch.get("mask")
                if mask is None:
                    raise KeyError("mask not found in batch. Ensure dataset provides masks for new model.")
                output = model(t2, mask.to(device)).squeeze(1).cpu().numpy()
                targets = batch["target"].squeeze(1).cpu().numpy()
                for pid, pred, gt in zip(batch["patient_id"], output, targets):
                    patient_ids.append(pid)
                    pred_medians.append(float(pred))
                    gt_medians.append(float(gt))

        results[exp_name] = {
            "patient_ids": patient_ids,
            "pred_medians": pred_medians,
            "gt_medians": gt_medians,
            "config": _jsonify(config),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results = {}
    for exp_name, data in results.items():
        rows = list(zip(data['patient_ids'], data['gt_medians'], data['pred_medians']))
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
            'config': data.get('config'),
            'patients': patients,
        }
    with open(output_path, "w") as f:
        yaml.safe_dump(save_results, f, sort_keys=False)
    print(f"Detailed results saved to: {output_path}")

    eval_script = REPO_ROOT / "evaluate_benchmark.py"
    if eval_script.exists():
        print(f"Running evaluation: {eval_script} --results {output_path}")
        subprocess.run(
            [sys.executable, str(eval_script), "--results", str(output_path)],
            check=False,
        )
    else:
        print(f"Warning: evaluate_benchmark.py not found at {eval_script}")


if __name__ == "__main__":
    main()
