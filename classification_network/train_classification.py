"""
Training script for liver fat fraction classification (steatosis grading).

Follows the structure of scalar_regression/train_scalar.py but uses
CrossEntropyLoss and classification metrics (accuracy, per-class accuracy,
macro F1).
"""

import argparse
import json
import os
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import create_data_splits, create_fat_stratified_splits, load_data_splits  # noqa: E402
from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar  # noqa: E402
from classification_network.dataset_classification import (  # noqa: E402
    LiverFatClassificationDataset,
    pad_collate_classification,
    compute_class_weights,
    CLASS_NAMES,
)
from classification_network.model_classification import get_classification_model  # noqa: E402
from utils import save_checkpoint, load_checkpoint, EarlyStopping  # noqa: E402


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_classification_metrics(
    pred_logits: torch.Tensor,
    target: torch.Tensor,
    num_classes: int = 4,
    ordinal: bool = False,
) -> Dict[str, float]:
    """Compute accuracy, per-class accuracy, and macro F1."""
    if ordinal:
        probs = torch.sigmoid(pred_logits)
        pred_classes = (probs > 0.5).sum(dim=1).long()
    else:
        pred_classes = pred_logits.argmax(dim=1)

    target_np = target.cpu().numpy()
    pred_np = pred_classes.cpu().numpy()

    correct = (pred_np == target_np).sum()
    accuracy = float(correct) / len(target_np) if len(target_np) > 0 else 0.0

    per_class_acc = {}
    precisions = []
    recalls = []
    for c in range(num_classes):
        gt_mask = target_np == c
        pred_mask = pred_np == c
        n_gt = gt_mask.sum()
        n_pred = pred_mask.sum()
        tp = (gt_mask & pred_mask).sum()

        recall = float(tp) / n_gt if n_gt > 0 else 0.0
        precision = float(tp) / n_pred if n_pred > 0 else 0.0
        per_class_acc[CLASS_NAMES[c]] = recall
        precisions.append(precision)
        recalls.append(recall)

    macro_f1 = 0.0
    for p, r in zip(precisions, recalls):
        if p + r > 0:
            macro_f1 += 2 * p * r / (p + r)
    macro_f1 /= num_classes

    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        **{f"acc_{name}": acc for name, acc in per_class_acc.items()},
    }


# ---------------------------------------------------------------------------
# Ordinal loss helpers
# ---------------------------------------------------------------------------

def ordinal_targets(class_labels: torch.Tensor, num_classes: int) -> torch.Tensor:
    """Convert class labels (B,) to cumulative binary targets (B, num_classes-1).

    E.g. class 2 with 4 classes -> [1, 1, 0].
    """
    batch_size = class_labels.size(0)
    targets = torch.zeros(batch_size, num_classes - 1, device=class_labels.device)
    for k in range(num_classes - 1):
        targets[:, k] = (class_labels > k).float()
    return targets


# ---------------------------------------------------------------------------
# Stratified sampling weights (reused from train_scalar.py)
# ---------------------------------------------------------------------------

def compute_stratified_weights(
    targets: List[float], bin_edges: List[float], bin_probs: List[float]
) -> List[float]:
    if len(bin_edges) != len(bin_probs) + 1:
        raise ValueError("bin_edges must be one longer than bin_probs")
    n_bins = len(bin_probs)
    counts = [0] * n_bins
    bins = []
    for t in targets:
        idx = 0
        for i in range(n_bins):
            if bin_edges[i] <= t < bin_edges[i + 1]:
                idx = i
                break
        bins.append(idx)
        counts[idx] += 1

    prob_sum = float(np.sum(bin_probs))
    if prob_sum <= 0:
        raise ValueError("bin_probs must sum to > 0")
    bin_probs = [p / prob_sum for p in bin_probs]

    bin_weights = []
    for i in range(n_bins):
        if counts[i] == 0:
            bin_weights.append(0.0)
        else:
            bin_weights.append(bin_probs[i] / counts[i])

    weights = [bin_weights[idx] for idx in bins]
    mean_weight = float(np.mean([w for w in weights if w > 0])) if weights else 1.0
    if mean_weight > 0:
        weights = [w / mean_weight for w in weights]
    return weights


# ---------------------------------------------------------------------------
# CLI & main
# ---------------------------------------------------------------------------

def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train liver fat fraction classifier")
    default_config = REPO_ROOT / "config" / "config_classification" / "train_config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to config file (default: config/config_classification/train_config.yaml)",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser


def _build_scalar_dataset(data_dir, patient_ids, data_cfg, augment=False, validate=True):
    """Helper to instantiate a LiverFatScalarDataset with common config."""
    return LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=patient_ids,
        t2_suffix=data_cfg.get("t2_suffix", "_t2_original"),
        ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
        mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
        use_subdirs=data_cfg.get("use_subdirs", False),
        use_patient_subdirs=data_cfg.get("use_patient_subdirs", True),
        t2_subdir=data_cfg.get("t2_subdir", "t2_images"),
        ff_subdir=data_cfg.get("ff_subdir", "fat_fraction_maps"),
        mask_subdir=data_cfg.get("mask_subdir", "liver_masks"),
        mask_erosion=data_cfg.get("mask_erosion", 3),
        input_mask_suffix=data_cfg.get("input_mask_suffix", "_t2_original_segmentation"),
        normalize_t2=data_cfg.get("normalize_t2", True),
        normalize_ff=data_cfg.get("normalize_ff", True),
        log_t2=data_cfg.get("log_t2", False),
        augment=augment,
        flip_prob=data_cfg.get("flip_prob", 0.5),
        rotate_prob=data_cfg.get("rotate_prob", 0.2),
        rotate_angle_min=data_cfg.get("rotate_angle_min", 1.0),
        rotate_angle_max=data_cfg.get("rotate_angle_max", 15.0),
        validate_files=validate,
    )


def main() -> None:
    args = build_arg_parser().parse_args()

    # ---- Config ----
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    data_cfg = config.get("data", {})
    train_cfg = config.get("training", {})
    model_cfg = config.get("model", {})
    optim_cfg = config.get("optimizer", {})
    loss_cfg = config.get("loss", {})
    cls_cfg = config.get("classification", {})

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

    num_classes = model_cfg.get("num_classes", 4)
    ordinal = model_cfg.get("ordinal", False)
    thresholds = cls_cfg.get("thresholds", [0.05, 0.15, 0.25])

    # ---- DDP ----
    ddp_cfg = config.get("ddp", {})
    ddp_enabled = bool(ddp_cfg.get("enabled", False))
    backend = ddp_cfg.get("backend", "nccl")

    rank = 0
    world_size = 1
    local_rank = 0

    if ddp_enabled:
        rank = int(os.environ.get("RANK", ddp_cfg.get("rank", 0)))
        world_size = int(os.environ.get("WORLD_SIZE", ddp_cfg.get("world_size", 1)))
        local_rank = int(os.environ.get("LOCAL_RANK", ddp_cfg.get("local_rank", 0)))
        dist.init_process_group(backend=backend, init_method="env://")

    # ---- Device ----
    device_name = config.get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        if ddp_enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # ---- Output dir ----
    output_root = Path(config["output_dir"])
    if not output_root.is_absolute():
        output_root = (REPO_ROOT / output_root).resolve()
    timestamp = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs")) if rank == 0 else None

    # ---- Data splits ----
    data_dir = Path(data_cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    fat_split_cfg = data_cfg.get("fat_split", {})
    use_fat_split = bool(fat_split_cfg.get("enabled", False))
    split_filename = fat_split_cfg.get("split_file", "data_splits_fatassigned.json")
    splits_file = data_dir / (split_filename if use_fat_split else "data_splits.json")

    if splits_file.exists():
        train_ids, val_ids, test_ids = load_data_splits(str(data_dir))
        if use_fat_split and splits_file.name != "data_splits.json":
            with open(splits_file, "r") as f:
                splits = json.load(f)
            train_ids, val_ids, test_ids = splits["train"], splits["val"], splits["test"]
    else:
        if use_fat_split:
            train_ids, val_ids, test_ids = create_fat_stratified_splits(
                str(data_dir),
                bin_edges=fat_split_cfg.get("bin_edges", [0.0, 0.05, 0.15, 0.25, 1.0]),
                train_ratio=data_cfg.get("train_ratio", 0.7),
                val_ratio=data_cfg.get("val_ratio", 0.15),
                test_ratio=data_cfg.get("test_ratio", 0.15),
                random_seed=fat_split_cfg.get("seed", data_cfg.get("random_seed", seed)),
                save_splits=True,
                split_filename=split_filename,
                use_subdirs=data_cfg.get("use_subdirs", False),
                use_patient_subdirs=data_cfg.get("use_patient_subdirs", True),
                t2_suffix=data_cfg.get("t2_suffix", "_t2_original"),
                ff_suffix=data_cfg.get("ff_suffix", "_ff_normalized"),
                mask_suffix=data_cfg.get("mask_suffix", "_segmentation"),
                mask_erosion=data_cfg.get("mask_erosion", 3),
            )
        else:
            train_ids, val_ids, test_ids = create_data_splits(
                str(data_dir),
                train_ratio=data_cfg.get("train_ratio", 0.7),
                val_ratio=data_cfg.get("val_ratio", 0.15),
                test_ratio=data_cfg.get("test_ratio", 0.15),
                random_seed=data_cfg.get("random_seed", seed),
                save_splits=True,
                use_subdirs=data_cfg.get("use_subdirs", False),
                use_patient_subdirs=data_cfg.get("use_patient_subdirs", True),
                t2_suffix=data_cfg.get("t2_suffix", "_t2_original"),
            )

    # Persist split file alongside experiment outputs
    try:
        split_copy_name = split_filename if use_fat_split else "data_splits.json"
        if splits_file.exists():
            shutil.copy2(splits_file, output_dir / split_copy_name)
            print(f"Saved split file copy to: {output_dir / split_copy_name}")
        else:
            print(f"Warning: split file not found for copy: {splits_file}")
    except Exception as exc:
        print(f"Warning: failed to copy split file to experiment folder: {exc}")

    # Validate all patient IDs
    all_ids = train_ids + val_ids + test_ids
    temp_ds = _build_scalar_dataset(data_dir, all_ids, data_cfg, augment=False, validate=True)
    valid_ids = set(temp_ds.patient_ids)
    train_ids = [pid for pid in train_ids if pid in valid_ids]
    val_ids = [pid for pid in val_ids if pid in valid_ids]
    test_ids = [pid for pid in test_ids if pid in valid_ids]

    # ---- Datasets ----
    train_scalar_ds = _build_scalar_dataset(
        data_dir, train_ids, data_cfg,
        augment=data_cfg.get("augment", False), validate=False,
    )
    val_scalar_ds = _build_scalar_dataset(
        data_dir, val_ids, data_cfg, augment=False, validate=False,
    )

    train_ds = LiverFatClassificationDataset(train_scalar_ds, thresholds=thresholds)
    val_ds = LiverFatClassificationDataset(val_scalar_ds, thresholds=thresholds)

    # ---- Samplers ----
    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        fat_sampling_cfg = data_cfg.get("fat_sampling", {})
        if fat_sampling_cfg.get("enabled", False):
            print("Fat-aware sampling enabled. Computing per-volume medians...")
            t0 = time.time()
            split_path = data_dir / fat_split_cfg.get("split_file", "data_splits_fatassigned.json")
            median_map = {}
            if split_path.exists():
                try:
                    with open(split_path, "r") as f:
                        split_data = json.load(f)
                    median_map = split_data.get("median_ff", {}) or {}
                    print(f"Loaded {len(median_map)} precomputed medians from {split_path}")
                except Exception as exc:
                    print(f"Failed to read {split_path}: {exc}")

            weight_ds = _build_scalar_dataset(
                data_dir, train_ids, data_cfg, augment=False, validate=False,
            )
            targets = []
            computed = 0
            for i in tqdm(range(len(weight_ds)), desc="Computing medians"):
                pid = weight_ds.patient_ids[i]
                if pid in median_map:
                    targets.append(float(median_map[pid]))
                else:
                    targets.append(float(weight_ds[i]["target"].item()))
                    computed += 1

            bin_edges = fat_sampling_cfg.get("bin_edges", [0.0, 0.05, 0.15, 0.25, 1.0])
            bin_probs = fat_sampling_cfg.get("bin_probs", [0.25, 0.25, 0.25, 0.25])
            print(f"Fat sampling bins: {bin_edges} probs: {bin_probs}")
            sample_weights = compute_stratified_weights(targets, bin_edges, bin_probs)
            print(
                f"Computed {len(sample_weights)} weights in {time.time() - t0:.1f}s "
                f"(computed {computed} medians, reused {len(median_map)})"
            )
            train_sampler = WeightedRandomSampler(
                sample_weights, num_samples=len(sample_weights), replacement=True,
            )

    # ---- Data loaders ----
    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate_classification,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 1),
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate_classification,
    )

    # ---- Model ----
    model = get_classification_model(
        in_channels=model_cfg.get("in_channels", 2),
        base_channels=model_cfg.get("base_channels", 16),
        num_classes=num_classes,
        ordinal=ordinal,
    ).to(device)
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    # ---- Loss ----
    class_weights_cfg = loss_cfg.get("class_weights", "auto")
    label_smoothing = loss_cfg.get("label_smoothing", 0.0)

    if ordinal:
        criterion = nn.BCEWithLogitsLoss()
    else:
        if class_weights_cfg == "auto":
            print("Computing class weights from training set...")
            cw = compute_class_weights(train_ds, num_classes=num_classes).to(device)
            print(f"Class weights: {cw.tolist()}")
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)
        elif class_weights_cfg == "none" or class_weights_cfg is None:
            criterion = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        else:
            cw = torch.tensor(class_weights_cfg, dtype=torch.float32).to(device)
            criterion = nn.CrossEntropyLoss(weight=cw, label_smoothing=label_smoothing)

    # ---- Optimizer ----
    opt_type = optim_cfg.get("type", "adamw").lower()
    lr = optim_cfg.get("lr", 1e-4)
    weight_decay = optim_cfg.get("weight_decay", 1e-5)
    if opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    # ---- LR Scheduler ----
    scheduler_cfg = config.get("scheduler", {})
    scheduler_type = scheduler_cfg.get("type", "none").lower()
    scheduler = None
    if scheduler_type == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=train_cfg.get("num_epochs", 120),
            eta_min=scheduler_cfg.get("eta_min", 1e-6),
        )
    elif scheduler_type == "plateau":
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=scheduler_cfg.get("factor", 0.5),
            patience=scheduler_cfg.get("patience", 10),
        )

    early_stopping = EarlyStopping(patience=train_cfg.get("early_stopping_patience", 20), min_delta=1e-4, mode="max")

    # ---- Resume ----
    start_epoch = 0
    best_val_f1 = 0.0

    if args.resume and rank == 0:
        checkpoint = load_checkpoint(args.resume, model.module if ddp_enabled else model, optimizer)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_f1 = checkpoint.get("best_val_f1", best_val_f1)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if rank == 0:
        print(f"Using device: {device}")
        print(f"Train: {len(train_ids)}, Val: {len(val_ids)}, Test: {len(test_ids)}")
        print(f"Num classes: {num_classes}, Ordinal: {ordinal}")
        print(f"Thresholds: {thresholds}")

    # ---- Training loop ----
    num_epochs = train_cfg.get("num_epochs", 120)
    grad_clip = train_cfg.get("gradient_clip", 1.0)
    if train_cfg.get("grad_accum_enabled", False):
        grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    else:
        grad_accum_steps = 1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []
        all_train_preds = []
        all_train_targets = []

        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()
        step_now = False
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")

        for batch_idx, batch in enumerate(pbar):
            t2 = batch["t2"].to(device)
            mask = batch["mask"].to(device)
            target = batch["target"].to(device)

            logits = model(t2, mask)

            if ordinal:
                ord_targets = ordinal_targets(target, num_classes)
                loss = criterion(logits, ord_targets) / grad_accum_steps
            else:
                loss = criterion(logits, target) / grad_accum_steps

            loss.backward()

            step_now = ((batch_idx + 1) % grad_accum_steps == 0)
            if step_now:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * grad_accum_steps)
            all_train_preds.append(logits.detach())
            all_train_targets.append(target.detach())

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}"})
            global_step = epoch * len(train_loader) + batch_idx
            if writer:
                writer.add_scalar("train/loss_step", loss.item() * grad_accum_steps, global_step)

        # Flush remaining gradients
        if not step_now:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        # ---- Validation ----
        model.eval()
        val_losses = []
        all_val_preds = []
        all_val_targets = []

        with torch.no_grad():
            for batch in val_loader:
                t2 = batch["t2"].to(device)
                mask = batch["mask"].to(device)
                target = batch["target"].to(device)

                logits = model(t2, mask)

                if ordinal:
                    ord_targets = ordinal_targets(target, num_classes)
                    loss = criterion(logits, ord_targets)
                else:
                    loss = criterion(logits, target)

                val_losses.append(loss.item())
                all_val_preds.append(logits.detach())
                all_val_targets.append(target.detach())

        # ---- Epoch metrics ----
        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        train_preds_cat = torch.cat(all_train_preds, dim=0)
        train_targets_cat = torch.cat(all_train_targets, dim=0)
        val_preds_cat = torch.cat(all_val_preds, dim=0)
        val_targets_cat = torch.cat(all_val_targets, dim=0)

        train_metrics = compute_classification_metrics(
            train_preds_cat, train_targets_cat, num_classes, ordinal,
        )
        val_metrics = compute_classification_metrics(
            val_preds_cat, val_targets_cat, num_classes, ordinal,
        )

        # ---- LR Scheduler step ----
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(val_loss)
            else:
                scheduler.step()

        # ---- Logging ----
        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("train/accuracy", train_metrics["accuracy"], epoch)
            writer.add_scalar("val/accuracy", val_metrics["accuracy"], epoch)
            writer.add_scalar("train/macro_f1", train_metrics["macro_f1"], epoch)
            writer.add_scalar("val/macro_f1", val_metrics["macro_f1"], epoch)
            for name in CLASS_NAMES:
                key = f"acc_{name}"
                if key in val_metrics:
                    writer.add_scalar(f"val/{key}", val_metrics[key], epoch)
            current_lr = optimizer.param_groups[0]["lr"]
            writer.add_scalar("train/lr", current_lr, epoch)

        # ---- Checkpoint (based on macro F1) ----
        val_f1 = val_metrics["macro_f1"]
        is_best = val_f1 > best_val_f1
        if is_best:
            best_val_f1 = val_f1

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_f1": best_val_f1,
            "config": config,
        }

        if rank == 0:
            save_checkpoint(checkpoint_state, is_best=is_best, checkpoint_dir=output_dir)

        if rank == 0:
            print(
                f"Epoch {epoch} - train loss {train_loss:.4f} - val loss {val_loss:.4f} "
                f"- val acc {val_metrics['accuracy']:.4f} - val F1 {val_metrics['macro_f1']:.4f}"
            )

        if early_stopping(val_f1) and rank == 0:
            print("Early stopping triggered.")
            break

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
