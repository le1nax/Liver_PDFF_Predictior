#!/usr/bin/env python3
"""
Benchmark script for liver fat fraction classification.

Produces two outputs:
  1. Regression-compatible YAML (maps predicted class -> bin center) so
     evaluate_benchmark.py can generate scatter/Bland-Altman plots.
  2. Classification-specific YAML with accuracy, F1, and confusion matrix.

Supports YAML config file (--config) and/or CLI arguments.
CLI arguments override config file values.
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

from dataset import create_data_splits, load_data_splits  # noqa: E402
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

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config_classification" / "benchmark_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark classification experiments. "
        "Uses config/config_classification/benchmark_config.yaml by default. "
        "CLI arguments override config file values."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to benchmark config YAML (default: config/config_classification/benchmark_config.yaml)",
    )
    parser.add_argument(
        "--experiment",
        default=None,
        help="Experiment directory containing checkpoint_best.pth",
    )
    parser.add_argument(
        "--checkpoint-name",
        default=None,
        help="Checkpoint filename inside experiment directory",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help="Path to data splits JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Output directory or YAML file path for benchmark results",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI arguments. CLI takes precedence."""
    config_path = Path(args.config) if args.config else DEFAULT_CONFIG_PATH
    if config_path.exists():
        cfg = load_config(config_path)
        print(f"Loaded config: {config_path}")
    else:
        if args.config:
            raise FileNotFoundError(f"Config file not found: {config_path}")
        cfg = {}

    # CLI overrides
    if args.experiment is not None:
        cfg["experiment"] = args.experiment
    if args.checkpoint_name is not None:
        cfg["checkpoint_name"] = args.checkpoint_name
    if args.splits_file is not None:
        cfg["splits_file"] = args.splits_file
    if args.output is not None:
        cfg["output"] = args.output
    if args.batch_size is not None:
        cfg["batch_size"] = args.batch_size
    if args.device is not None:
        cfg["device"] = args.device
    if args.num_workers is not None:
        cfg["num_workers"] = args.num_workers

    # Defaults
    cfg.setdefault("checkpoint_name", "checkpoint_best.pth")
    cfg.setdefault("batch_size", 1)
    cfg.setdefault("num_workers", 2)
    cfg.setdefault("device", "cuda")

    if "experiment" not in cfg or not cfg["experiment"]:
        raise ValueError(
            "No experiment specified. Set 'experiment' in the config YAML or pass --experiment."
        )

    return cfg


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


def resolve_splits_file(splits_file_cfg, exp_dir: Path, data_dir: Path) -> Path | None:
    """Resolve the splits file path.

    Priority:
    1. Explicit path from config/CLI (splits_file)
    2. Split file inside the experiment directory
    3. data_splits.json in data_dir (handled by load_test_ids fallback)
    """
    if splits_file_cfg:
        p = Path(splits_file_cfg)
        if not p.is_absolute():
            p = (REPO_ROOT / p).resolve()
        if p.exists():
            return p
        raise FileNotFoundError(f"Splits file not found: {p}")

    for name in [
        "data_splits_fatassigned_filtered.json",
        "data_splits_fatassigned.json",
        "data_splits.json",
    ]:
        candidate = exp_dir / name
        if candidate.exists():
            return candidate

    return None


def load_test_ids_from_file(splits_file: Path) -> List[str]:
    """Load test IDs from a specific splits JSON file."""
    with open(splits_file, "r") as f:
        splits = json.load(f)
    return splits["test"]


def load_test_ids(data_dir: Path, data_cfg: dict) -> List[str]:
    """Fallback: load or create test IDs from data_dir."""
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


def resolve_output_dir(cfg: dict, exp_name: str) -> Path:
    """Resolve the output directory for benchmark results."""
    if cfg.get("output"):
        output_dir = Path(cfg["output"])
        if not output_dir.is_absolute():
            output_dir = (REPO_ROOT / output_dir).resolve()
        # If it looks like a file path, use its parent
        if output_dir.suffix in (".yaml", ".yml"):
            output_dir = output_dir.parent
    else:
        outputs_dir = REPO_ROOT / "outputs" / "classification_run"
        output_dir = outputs_dir / "benchmark_evaluation" / exp_name

    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


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


def build_eval_cmd(eval_script: Path, results_path: Path, eval_cfg: dict) -> List[str]:
    """Build the evaluate_benchmark.py subprocess command from evaluation config."""
    cmd = [sys.executable, str(eval_script), "--results", str(results_path)]

    min_gt_ff = eval_cfg.get("min_gt_ff", 0.0)
    if min_gt_ff > 0:
        cmd += ["--min-gt-ff", str(min_gt_ff)]

    outlier_threshold = eval_cfg.get("outlier_threshold", 10.0)
    if outlier_threshold != 10.0:
        cmd += ["--outlier-threshold", str(outlier_threshold)]

    toggle_map = {
        "scatter": "--no-scatter",
        "bland_altman": "--no-bland-altman",
        "error_distribution": "--no-error-distribution",
        "error_vs_gt": "--no-error-vs-gt",
        "classification": "--no-classification",
        "ranking": "--no-ranking",
        "outliers": "--no-outliers",
        "pdf_report": "--no-pdf",
        "metrics_table": "--no-metrics",
    }
    for key, flag in toggle_map.items():
        if not eval_cfg.get(key, True):
            cmd.append(flag)

    return cmd


def main() -> None:
    args = build_arg_parser().parse_args()
    cfg = resolve_config(args)

    device_str = cfg["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        device_str = "cpu"
    device = torch.device(device_str)
    print(f"Using device: {device}")

    exp_dir = resolve_experiment_dir(cfg["experiment"])
    checkpoint_name = cfg["checkpoint_name"]
    checkpoint_path = exp_dir / checkpoint_name
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    train_config = checkpoint.get("config", {})
    data_cfg = train_config.get("data", {})
    cls_cfg = train_config.get("classification", {})
    thresholds = cls_cfg.get("thresholds", [0.05, 0.15, 0.25])
    num_classes = train_config.get("model", {}).get("num_classes", 4)
    ordinal = train_config.get("model", {}).get("ordinal", False)
    data_dir = resolve_data_dir(train_config)

    # Resolve splits file
    splits_file = resolve_splits_file(cfg.get("splits_file"), exp_dir, data_dir)
    if splits_file:
        print(f"Using splits file: {splits_file}")
        test_ids = load_test_ids_from_file(splits_file)
    else:
        print(f"No explicit splits file found, falling back to data_dir: {data_dir}")
        test_ids = load_test_ids(data_dir, data_cfg)

    dataset = build_dataset(data_dir, test_ids, data_cfg, thresholds)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=pad_collate_classification,
    )

    model = build_model(checkpoint, device)
    exp_name = exp_dir.name
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

    # Resolve output directory
    output_dir = resolve_output_dir(cfg, exp_name)
    regression_yaml_path = output_dir / "benchmark_results_classification_regression_compat.yaml"
    classification_yaml_path = output_dir / "benchmark_results_classification.yaml"

    # Build regression-compatible output (pred = bin center, gt = actual FF)
    rows = list(zip(patient_ids, gt_ff_list, pred_ff_list))
    rows.sort(key=lambda r: str(r[0]))
    patients_regression = [
        {"patient_id": str(pid), "gt": float(gt), "pred": float(pred)}
        for pid, gt, pred in rows
    ]
    regression_results = {
        exp_name: {
            "config": _jsonify(train_config),
            "patients": patients_regression,
        }
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
    classification_results = {
        exp_name: {
            "config": _jsonify(train_config),
            "report": report,
            "patients": patients_cls,
        }
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
        eval_cfg = cfg.get("evaluation", {})
        eval_cmd = build_eval_cmd(eval_script, regression_yaml_path, eval_cfg)
        print(f"Running evaluation: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=False)
    else:
        print(f"Warning: evaluate_benchmark.py not found at {eval_script}")


if __name__ == "__main__":
    main()
