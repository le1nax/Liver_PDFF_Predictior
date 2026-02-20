#!/usr/bin/env python3
"""
Benchmark script for liver fat fraction classification.

Produces two outputs:
  1. Regression-compatible YAML (maps predicted class -> bin center) so
     evaluate_benchmark.py can generate scatter/Bland-Altman plots.
  2. Classification-specific YAML with accuracy, F1, and confusion matrix.

Invokes evaluate_benchmark.py via subprocess for plot generation.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from dataset import create_data_splits, create_fat_stratified_splits, load_data_splits  # noqa: E402
from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar  # noqa: E402
from classification_network.dataset_classification import (  # noqa: E402
    LiverFatClassificationDataset,
    pad_collate_classification,
    CLASS_NAMES,
)
from classification_network.model_classification import get_classification_model  # noqa: E402

# Bin centers for mapping predicted class -> pseudo-continuous value
# Healthy 0-5%  -> 2.5%, Slight 5-15% -> 10%, Mild 15-25% -> 20%, Strong 25-100% -> 37.5%
CLASS_BIN_CENTERS = [0.025, 0.10, 0.20, 0.375]


def build_arg_parser(
    default_experiments: list,
    default_checkpoint_name: str,
    default_batch_size: int,
    default_num_workers: int,
) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark classification experiments")
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
        help="Path to save benchmark results YAML",
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
    if not exp_arg.startswith("classification_network/"):
        candidates.append(REPO_ROOT / "classification_network" / exp_arg)
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    tried = ", ".join(str(c) for c in candidates)
    raise FileNotFoundError(f"Experiment directory not found. Tried: {tried}")


def resolve_split_file(exp_dir: Path, data_dir: Path) -> Path | None:
    candidates = [
        exp_dir / "data_splits_fatassigned.json",
        exp_dir / "data_splits.json",
        data_dir / "data_splits_fatassigned.json",
        data_dir / "data_splits.json",
    ]
    for c in candidates:
        if c.exists():
            return c
    return None


def load_test_ids(data_dir: Path, data_cfg: dict, exp_dir: Path) -> List[str]:
    splits_file = resolve_split_file(exp_dir, data_dir)
    if splits_file is not None and splits_file.exists():
        if splits_file.name == "data_splits_fatassigned.json":
            with open(splits_file, "r") as f:
                splits = json.load(f)
            return splits.get("test", [])
        else:
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


def build_dataset(
    data_dir: Path, test_ids: List[str], data_cfg: dict, thresholds: List[float],
) -> LiverFatClassificationDataset:
    scalar_ds = LiverFatScalarDataset(
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
    return LiverFatClassificationDataset(scalar_ds, thresholds=thresholds)


def build_model(checkpoint: dict, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint.get("config", {}).get("model", {})
    model = get_classification_model(
        in_channels=model_config.get("in_channels", 2),
        base_channels=model_config.get("base_channels", 16),
        num_classes=model_config.get("num_classes", 4),
        ordinal=model_config.get("ordinal", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    return model


def compute_classification_report(
    gt_classes: np.ndarray,
    pred_classes: np.ndarray,
    num_classes: int = 4,
) -> Dict[str, Any]:
    """Compute accuracy, per-class precision/recall/F1, and confusion matrix."""
    accuracy = float((gt_classes == pred_classes).mean())

    conf_matrix = np.zeros((num_classes, num_classes), dtype=int)
    for g, p in zip(gt_classes, pred_classes):
        conf_matrix[g, p] += 1

    per_class = {}
    f1_scores = []
    for c in range(num_classes):
        tp = int(conf_matrix[c, c])
        n_gt = int((gt_classes == c).sum())
        n_pred = int((pred_classes == c).sum())
        precision = tp / n_pred if n_pred > 0 else 0.0
        recall = tp / n_gt if n_gt > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        per_class[CLASS_NAMES[c]] = {
            "n_gt": n_gt,
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_scores.append(f1)

    return {
        "accuracy": round(accuracy, 4),
        "macro_f1": round(float(np.mean(f1_scores)), 4),
        "confusion_matrix": conf_matrix.tolist(),
        "per_class": per_class,
    }


def _jsonify(obj: Any) -> Any:
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, dict):
        return {str(k): _jsonify(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonify(v) for v in obj]
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def main() -> None:
    default_checkpoint_name = "checkpoint_best.pth"
    outputs_dir = REPO_ROOT / "outputs" / "classification_run"
    default_experiments: list = []
    default_batch_size = 1
    default_num_workers = 2

    args = build_arg_parser(
        default_experiments=default_experiments,
        default_checkpoint_name=default_checkpoint_name,
        default_batch_size=default_batch_size,
        default_num_workers=default_num_workers,
    ).parse_args()

    if not args.experiments:
        print("Error: no experiments specified. Use --experiments <dir1> [<dir2> ...]")
        sys.exit(1)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    regression_results = {}
    classification_results = {}
    experiments = [resolve_experiment_dir(p) for p in args.experiments]

    if args.output is None:
        suffix = "_".join(exp.name for exp in experiments)
        output_dir = outputs_dir / "benchmark_evaluation" / suffix
    else:
        output_dir = Path(args.output).parent

    output_dir.mkdir(parents=True, exist_ok=True)
    regression_yaml_path = output_dir / "benchmark_results_classification_regression_compat.yaml"
    classification_yaml_path = output_dir / "benchmark_results_classification.yaml"

    for exp_dir in experiments:
        exp_name = exp_dir.name
        checkpoint_path = exp_dir / args.checkpoint_name
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        config = checkpoint.get("config", {})
        data_cfg = config.get("data", {})
        cls_cfg = config.get("classification", {})
        thresholds = cls_cfg.get("thresholds", [0.05, 0.15, 0.25])
        num_classes = config.get("model", {}).get("num_classes", 4)
        ordinal = config.get("model", {}).get("ordinal", False)
        data_dir = resolve_data_dir(config)

        test_ids = load_test_ids(data_dir, data_cfg, exp_dir)
        dataset = build_dataset(data_dir, test_ids, data_cfg, thresholds)
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=pad_collate_classification,
        )

        model = build_model(checkpoint, device)
        print(f"\n{'=' * 60}")
        print(f"Benchmarking {exp_name}")
        print(f"{'=' * 60}")
        print(f"Loaded checkpoint: {checkpoint_path}")
        print(f"Test set: {len(dataset)} patients")

        patient_ids = []
        gt_classes_list = []
        pred_classes_list = []
        gt_ff_list = []
        pred_ff_list = []

        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Processing {exp_name}"):
                t2 = batch["t2"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"]
                ff_value = batch["ff_value"]

                logits = model(t2, mask)

                if ordinal:
                    probs = torch.sigmoid(logits)
                    pred_cls = (probs > 0.5).sum(dim=1).long().cpu().numpy()
                else:
                    pred_cls = logits.argmax(dim=1).cpu().numpy()

                gt_cls = target.cpu().numpy()
                gt_ff = ff_value.cpu().numpy()

                for pid, gc, pc, ff in zip(batch["patient_id"], gt_cls, pred_cls, gt_ff):
                    patient_ids.append(pid)
                    gt_classes_list.append(int(gc))
                    pred_classes_list.append(int(pc))
                    gt_ff_list.append(float(ff))
                    pred_ff_list.append(CLASS_BIN_CENTERS[int(pc)])

        gt_classes_arr = np.array(gt_classes_list)
        pred_classes_arr = np.array(pred_classes_list)

        # Classification report
        report = compute_classification_report(gt_classes_arr, pred_classes_arr, num_classes)
        print(f"Accuracy: {report['accuracy']:.4f}")
        print(f"Macro F1: {report['macro_f1']:.4f}")

        # Build regression-compatible output (pred = bin center, gt = actual FF)
        rows = list(zip(patient_ids, gt_ff_list, pred_ff_list))
        rows.sort(key=lambda r: str(r[0]))
        patients_regression = [
            {"patient_id": str(pid), "gt": float(gt), "pred": float(pred)}
            for pid, gt, pred in rows
        ]
        regression_results[exp_name] = {
            "config": _jsonify(config),
            "patients": patients_regression,
        }

        # Build classification-specific output
        rows_cls = list(zip(patient_ids, gt_classes_list, pred_classes_list, gt_ff_list))
        rows_cls.sort(key=lambda r: str(r[0]))
        patients_cls = [
            {
                "patient_id": str(pid),
                "gt_class": int(gc),
                "gt_class_name": CLASS_NAMES[int(gc)],
                "pred_class": int(pc),
                "pred_class_name": CLASS_NAMES[int(pc)],
                "gt_ff": float(ff),
            }
            for pid, gc, pc, ff in rows_cls
        ]
        classification_results[exp_name] = {
            "config": _jsonify(config),
            "report": report,
            "patients": patients_cls,
        }

    # Save regression-compatible YAML
    with open(regression_yaml_path, "w") as f:
        yaml.safe_dump(regression_results, f, sort_keys=False)
    print(f"\nRegression-compatible results saved to: {regression_yaml_path}")

    # Save classification-specific YAML
    with open(classification_yaml_path, "w") as f:
        yaml.safe_dump(_jsonify(classification_results), f, sort_keys=False)
    print(f"Classification results saved to: {classification_yaml_path}")

    # Invoke evaluate_benchmark.py for plots/report
    eval_script = REPO_ROOT / "evaluate_benchmark.py"
    if eval_script.exists():
        print(f"\nRunning evaluation: {eval_script} --results {regression_yaml_path}")
        subprocess.run(
            [sys.executable, str(eval_script), "--results", str(regression_yaml_path)],
            check=False,
        )
    else:
        print(f"Warning: evaluate_benchmark.py not found at {eval_script}")


if __name__ == "__main__":
    main()
