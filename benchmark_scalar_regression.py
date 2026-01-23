#!/usr/bin/env python3
"""
Benchmark script for scalar regression fat fraction prediction.
Generates output in the same format as benchmark_pixel_level.py.
"""

import argparse
import json
from pathlib import Path
from typing import List
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import create_data_splits, load_data_splits
from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar
from scalar_regression.model_scalar import get_scalar_model

REPO_ROOT = Path(__file__).resolve().parent


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark scalar regression experiments")
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=[
            "scalar_regression/outputs/scalar_regression_run/experiment_20260122_095324",
            "scalar_regression/outputs/scalar_regression_run/experiment_20260122_110236",
        ],
        help="Experiment directories containing checkpoint_best.pth",
    )
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_best.pth",
        help="Checkpoint filename inside each experiment directory",
    )
    parser.add_argument(
        "--output",
        default="outputs/benchmark_results_scalar_regression.json",
        help="Path to save benchmark results JSON",
    )
    parser.add_argument("--batch-size", type=int, default=2, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=2, help="DataLoader workers")
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


def load_test_ids(data_dir: Path, data_cfg: dict) -> List[str]:
    splits_file = data_dir / "data_splits.json"
    if splits_file.exists():
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
        use_patient_subdirs=data_cfg.get("use_patient_subdirs", False),
        t2_subdir=data_cfg.get("t2_subdir", "t2_images"),
        ff_subdir=data_cfg.get("ff_subdir", "fat_fraction_maps"),
        mask_subdir=data_cfg.get("mask_subdir", "liver_masks"),
        normalize_t2=data_cfg.get("normalize_t2", True),
        normalize_ff=data_cfg.get("normalize_ff", True),
        mask_erosion=data_cfg.get("mask_erosion", 3),
        augment=False,
        validate_files=True,
    )


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint.get("config", {}).get("model", {})
    base_channels = model_config.get("base_channels", 16)
    model = get_scalar_model(in_channels=1, base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def main() -> None:
    args = build_arg_parser().parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    results = {}
    experiments = [resolve_experiment_dir(p) for p in args.experiments]

    for exp_dir in experiments:
        exp_name = exp_dir.name
        checkpoint_path = exp_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        data_cfg = config.get("data", {})
        data_dir = resolve_data_dir(config)

        test_ids = load_test_ids(data_dir, data_cfg)
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
                output = model(t2).squeeze(1).cpu().numpy()
                targets = batch["target"].squeeze(1).cpu().numpy()
                for pid, pred, gt in zip(batch["patient_id"], output, targets):
                    patient_ids.append(pid)
                    pred_medians.append(float(pred))
                    gt_medians.append(float(gt))

        results[exp_name] = {
            "patient_ids": patient_ids,
            "pred_medians": pred_medians,
            "gt_medians": gt_medians,
        }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")


if __name__ == "__main__":
    main()
