#!/usr/bin/env python3
"""
Benchmark script for scalar regression fat fraction prediction.
Generates output in the same format as benchmark_pixel_level.py.

Supports YAML config file (--config) and/or CLI arguments.
CLI arguments override config file values.
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

MODEL_VERSIONS = {
    "v2": "scalar_regression.model_scalar_v2",
    "v1": "scalar_regression.model_scalar",
    "old": "scalar_regression.old_model",
}


def get_model_factory(model_version: str):
    """Import and return get_scalar_model from the requested model module."""
    if model_version not in MODEL_VERSIONS:
        raise ValueError(
            f"Unknown model_version '{model_version}'. Choose from: {', '.join(MODEL_VERSIONS)}"
        )
    import importlib
    module = importlib.import_module(MODEL_VERSIONS[model_version])
    return module.get_scalar_model

DEFAULT_CONFIG_PATH = REPO_ROOT / "config" / "config_scalar_regression" / "benchmark_config.yaml"


def load_config(config_path: Path) -> dict:
    """Load YAML config file."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f) or {}


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Benchmark scalar regression experiments. "
        "Uses config/config_scalar_regression/benchmark_config.yaml by default. "
        "CLI arguments override config file values."
    )
    parser.add_argument(
        "--config",
        default=None,
        help="Path to benchmark config YAML (default: config/config_scalar_regression/benchmark_config.yaml)",
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
        "--model-version",
        default=None,
        choices=list(MODEL_VERSIONS),
        help="Model version: v2 (model_scalar_v2), v1 (model_scalar), old (old_model)",
    )
    parser.add_argument(
        "--splits-file",
        default=None,
        help="Path to data splits JSON file",
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to save benchmark results YAML",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Inference batch size")
    parser.add_argument("--num-workers", type=int, default=None, help="DataLoader workers")
    parser.add_argument("--device", default=None, help="Device (cuda or cpu)")
    return parser


def resolve_config(args: argparse.Namespace) -> dict:
    """Merge YAML config with CLI arguments. CLI takes precedence."""
    # Load YAML config
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
    if args.model_version is not None:
        cfg["model_version"] = args.model_version
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

    # Defaults for anything still unset
    cfg.setdefault("checkpoint_name", "checkpoint_best.pth")
    cfg.setdefault("model_version", "v2")
    cfg.setdefault("batch_size", 2)
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
    if not exp_arg.startswith("scalar_regression/"):
        candidates.append(REPO_ROOT / "scalar_regression" / exp_arg)
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

    # Look for split files inside the experiment directory
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


def build_model(checkpoint: dict, model_version: str, device: torch.device) -> torch.nn.Module:
    model_config = checkpoint.get("config", {}).get("model", {})
    base_channels = model_config.get("base_channels", 16)
    factory = get_model_factory(model_version)
    model = factory(in_channels=model_config.get("in_channels", 2), base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    print(f"Using model version: {model_version} ({MODEL_VERSIONS[model_version]})")
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
    data_dir = resolve_data_dir(train_config)

    # Resolve splits file
    splits_file = resolve_splits_file(cfg.get("splits_file"), exp_dir, data_dir)
    if splits_file:
        print(f"Using splits file: {splits_file}")
        test_ids = load_test_ids_from_file(splits_file)
    else:
        print(f"No explicit splits file found, falling back to data_dir: {data_dir}")
        test_ids = load_test_ids(data_dir, data_cfg)

    dataset = build_dataset(data_dir, test_ids, data_cfg)
    loader = DataLoader(
        dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
        collate_fn=pad_collate_scalar,
    )

    model = build_model(checkpoint, cfg["model_version"], device)
    exp_name = exp_dir.name
    print(f"\n{'=' * 60}")
    print(f"Benchmarking {exp_name}")
    print(f"{'=' * 60}")
    print(f"Loaded checkpoint: {checkpoint_path}")
    print(f"Test set: {len(dataset)} patients")

    pred_medians = []
    gt_medians = []
    patient_ids = []

    # Old model takes single input; v1/v2 take (t2, mask)
    needs_mask = cfg["model_version"] != "old"

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Processing {exp_name}"):
            t2 = batch["t2"].to(device)
            if needs_mask:
                mask = batch["mask"].to(device)
                output = model(t2, mask).squeeze(1).cpu().numpy()
            else:
                output = model(t2).squeeze(1).cpu().numpy()
            targets = batch["target"].squeeze(1).cpu().numpy()
            for pid, pred, gt in zip(batch["patient_id"], output, targets):
                patient_ids.append(pid)
                pred_medians.append(float(pred))
                gt_medians.append(float(gt))

    results = {
        exp_name: {
            "patient_ids": patient_ids,
            "pred_medians": pred_medians,
            "gt_medians": gt_medians,
            "config": _jsonify(train_config),
        }
    }

    # Determine output path
    default_filename = "benchmark_results_scalar_regression.yaml"
    if cfg.get("output"):
        output_path = Path(cfg["output"])
        if not output_path.is_absolute():
            output_path = (REPO_ROOT / output_path).resolve()
        # If output is a directory or has no yaml extension, treat as directory
        if output_path.is_dir() or output_path.suffix not in (".yaml", ".yml"):
            output_path = output_path / default_filename
    else:
        outputs_dir = REPO_ROOT / "outputs" / "scalar_regression_run"
        output_path = outputs_dir / "benchmark_evaluation" / exp_name / default_filename

    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_results = {}
    for name, data in results.items():
        rows = list(zip(data['patient_ids'], data['gt_medians'], data['pred_medians']))
        rows.sort(key=lambda r: str(r[0]))
        patients = [
            {
                'patient_id': str(pid),
                'gt': float(gt_val),
                'pred': float(pred_val),
            }
            for pid, gt_val, pred_val in rows
        ]
        save_results[name] = {
            'config': data.get('config'),
            'patients': patients,
        }
    with open(output_path, "w") as f:
        yaml.safe_dump(save_results, f, sort_keys=False)
    print(f"Detailed results saved to: {output_path}")

    eval_script = REPO_ROOT / "evaluate_benchmark.py"
    if eval_script.exists():
        eval_cfg = cfg.get("evaluation", {})
        eval_cmd = [sys.executable, str(eval_script), "--results", str(output_path)]

        min_gt_ff = eval_cfg.get("min_gt_ff", 0.0)
        if min_gt_ff > 0:
            eval_cmd += ["--min-gt-ff", str(min_gt_ff)]

        outlier_threshold = eval_cfg.get("outlier_threshold", 10.0)
        if outlier_threshold != 10.0:
            eval_cmd += ["--outlier-threshold", str(outlier_threshold)]

        # Toggle flags: config uses true/false, CLI uses --no-* to disable
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
                eval_cmd.append(flag)

        print(f"Running evaluation: {' '.join(eval_cmd)}")
        subprocess.run(eval_cmd, check=False)
    else:
        print(f"Warning: evaluate_benchmark.py not found at {eval_script}")


if __name__ == "__main__":
    main()
