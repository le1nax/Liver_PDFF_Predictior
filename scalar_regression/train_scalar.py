"""
Training script for scalar fat fraction regression from raw 3D T2 volumes.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import os
from torch.nn.parallel import DistributedDataParallel as DDP
import yaml
from torch.utils.data import DataLoader, DistributedSampler, WeightedRandomSampler
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from dataset import create_data_splits, load_data_splits  # noqa: E402
from scalar_regression.dataset_scalar import LiverFatScalarDataset, pad_collate_scalar  # noqa: E402
from scalar_regression.model_scalar import get_scalar_model  # noqa: E402
from utils import save_checkpoint, load_checkpoint, EarlyStopping  # noqa: E402


def compute_scalar_metrics(pred: torch.Tensor, target: torch.Tensor) -> Dict[str, float]:
    pred_np = pred.detach().cpu().numpy().reshape(-1)
    target_np = target.detach().cpu().numpy().reshape(-1)
    mae = float(np.mean(np.abs(pred_np - target_np)))
    rmse = float(np.sqrt(np.mean((pred_np - target_np) ** 2)))
    if len(pred_np) > 1 and np.std(pred_np) > 0 and np.std(target_np) > 0:
        correlation = float(np.corrcoef(pred_np, target_np)[0, 1])
    else:
        correlation = 0.0
    return {"mae": mae, "rmse": rmse, "correlation": correlation}


def compute_weighted_scalar_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    loss_type: str,
    beta: float,
    weight_cfg: dict
) -> torch.Tensor:
    diff = pred - target
    if loss_type == "l1":
        base_loss = diff.abs()
    elif loss_type == "mse":
        base_loss = diff ** 2
    else:
        abs_diff = diff.abs()
        base_loss = torch.where(
            abs_diff < beta,
            0.5 * abs_diff ** 2 / beta,
            abs_diff - 0.5 * beta
        )

    alpha = float(weight_cfg.get("alpha", 4.0))
    gamma = float(weight_cfg.get("gamma", 2.0))
    max_weight = weight_cfg.get("max_weight")
    weight = 1.0 + alpha * torch.pow(target.clamp_min(0.0), gamma)
    if max_weight is not None:
        weight = torch.clamp(weight, max=float(max_weight))

    return (base_loss * weight).mean()


def compute_stratified_weights(targets: List[float], bin_edges: List[float], bin_probs: List[float]) -> List[float]:
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


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train scalar fat fraction regressor")
    default_config = REPO_ROOT / "config" / "config_scalar_regression" / "train_config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to config file (default: config/config_scalar_regression/train_config.yaml)",
    )
    parser.add_argument("--resume", default=None, help="Path to checkpoint to resume from")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

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

    seed = train_cfg.get("seed", 42)
    torch.manual_seed(seed)
    np.random.seed(seed)

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

    device_name = config.get("device", "cuda")
    if device_name == "cuda" and torch.cuda.is_available():
        if ddp_enabled:
            torch.cuda.set_device(local_rank)
            device = torch.device(f"cuda:{local_rank}")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    output_root = Path(config["output_dir"])
    timestamp = datetime.now().strftime("experiment_%Y%m%d_%H%M%S")
    output_dir = output_root / timestamp
    output_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(output_dir / "logs")) if rank == 0 else None

    data_dir = Path(data_cfg["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    splits_file = data_dir / "data_splits.json"
    if splits_file.exists():
        train_ids, val_ids, test_ids = load_data_splits(str(data_dir))
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

    all_ids = train_ids + val_ids + test_ids
    temp_dataset = LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=all_ids,
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
        log_t2=data_cfg.get("log_t2", False),
        augment=data_cfg.get("augment", False),
        flip_prob=data_cfg.get("flip_prob", 0.5),
        rotate_prob=data_cfg.get("rotate_prob", 0.2),
        rotate_angle_min=data_cfg.get("rotate_angle_min", 1.0),
        rotate_angle_max=data_cfg.get("rotate_angle_max", 15.0),
        validate_files=True,
    )
    valid_ids = set(temp_dataset.patient_ids)
    train_ids = [pid for pid in train_ids if pid in valid_ids]
    val_ids = [pid for pid in val_ids if pid in valid_ids]
    test_ids = [pid for pid in test_ids if pid in valid_ids]

    train_ds = LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=train_ids,
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
        log_t2=data_cfg.get("log_t2", False),
        augment=data_cfg.get("augment", False),
        flip_prob=data_cfg.get("flip_prob", 0.5),
        rotate_prob=data_cfg.get("rotate_prob", 0.2),
        rotate_angle_min=data_cfg.get("rotate_angle_min", 1.0),
        rotate_angle_max=data_cfg.get("rotate_angle_max", 15.0),
        validate_files=False,
    )
    val_ds = LiverFatScalarDataset(
        data_dir=str(data_dir),
        patient_ids=val_ids,
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
        log_t2=data_cfg.get("log_t2", False),
        augment=False,
        validate_files=False,
    )

    train_sampler = None
    val_sampler = None
    if ddp_enabled:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        fat_sampling_cfg = data_cfg.get("fat_sampling", {})
        if fat_sampling_cfg.get("enabled", False):
            weight_ds = LiverFatScalarDataset(
                data_dir=str(data_dir),
                patient_ids=train_ids,
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
                log_t2=data_cfg.get("log_t2", False),
                augment=False,
                validate_files=False,
            )
            targets = [float(weight_ds[i]["target"].item()) for i in range(len(weight_ds))]
            bin_edges = fat_sampling_cfg.get("bin_edges", [0.0, 0.1, 0.2, 1.0])
            bin_probs = fat_sampling_cfg.get("bin_probs", [0.2, 0.3, 0.5])
            sample_weights = compute_stratified_weights(targets, bin_edges, bin_probs)
            train_sampler = WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate_scalar,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=train_cfg.get("batch_size", 2),
        shuffle=False,
        sampler=val_sampler,
        num_workers=train_cfg.get("num_workers", 2),
        collate_fn=pad_collate_scalar,
    )

    model = get_scalar_model(in_channels=1, base_channels=model_cfg.get("base_channels", 16)).to(device)
    if ddp_enabled:
        model = DDP(model, device_ids=[local_rank] if device.type == "cuda" else None)

    loss_type = loss_cfg.get("type", "smooth_l1").lower()
    weight_cfg = loss_cfg.get("fat_weighting", {})
    weight_enabled = weight_cfg.get("enabled", False)
    beta = float(loss_cfg.get("beta", 0.02))
    if loss_type == "l1":
        criterion = nn.L1Loss()
    elif loss_type == "mse":
        criterion = nn.MSELoss()
    else:
        criterion = nn.SmoothL1Loss(beta=beta)

    opt_type = optim_cfg.get("type", "adam").lower()
    lr = optim_cfg.get("lr", 1e-4)
    weight_decay = optim_cfg.get("weight_decay", 1e-5)
    if opt_type == "adamw":
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    elif opt_type == "sgd":
        optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.9)
    else:
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    early_stopping = EarlyStopping(patience=20, min_delta=1e-4)

    start_epoch = 0
    best_val_loss = float("inf")

    if args.resume and rank == 0:
        checkpoint = load_checkpoint(args.resume, model.module if ddp_enabled else model, optimizer)
        start_epoch = checkpoint.get("epoch", 0) + 1
        best_val_loss = checkpoint.get("best_val_loss", best_val_loss)
        print(f"Resumed from {args.resume} at epoch {start_epoch}")

    if rank == 0:
        print(f"Using device: {device}")
        print(f"Train size: {len(train_ids)}, Val size: {len(val_ids)}, Test size: {len(test_ids)}")

    num_epochs = train_cfg.get("num_epochs", 100)
    grad_clip = train_cfg.get("gradient_clip", 0.0)
    if train_cfg.get("grad_accum_enabled", True):
        grad_accum_steps = max(1, int(train_cfg.get("grad_accum_steps", 1)))
    else:
        grad_accum_steps = 1

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_losses = []
        train_metrics = {"mae": [], "rmse": [], "correlation": []}

        if train_sampler is not None:
            train_sampler.set_epoch(epoch)

        optimizer.zero_grad()
        step_now = False
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}")
        for batch_idx, batch in enumerate(pbar):
            t2 = batch["t2"].to(device)
            target = batch["target"].to(device)

            output = model(t2)
            if weight_enabled:
                loss = compute_weighted_scalar_loss(output, target, loss_type, beta, weight_cfg) / grad_accum_steps
            else:
                loss = criterion(output, target) / grad_accum_steps
            loss.backward()

            step_now = ((batch_idx + 1) % grad_accum_steps == 0)
            if step_now:
                if grad_clip and grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * grad_accum_steps)
            metrics = compute_scalar_metrics(output, target)
            for k, v in metrics.items():
                train_metrics[k].append(v)

            pbar.set_postfix({"loss": f"{loss.item() * grad_accum_steps:.4f}", "mae": f"{metrics['mae']:.4f}"})
            global_step = epoch * len(train_loader) + batch_idx
            if writer:
                writer.add_scalar("train/loss_step", loss.item() * grad_accum_steps, global_step)

        if not step_now:
            if grad_clip and grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()
            optimizer.zero_grad()

        model.eval()
        val_losses = []
        val_metrics = {"mae": [], "rmse": [], "correlation": []}
        with torch.no_grad():
            for batch in val_loader:
                t2 = batch["t2"].to(device)
                target = batch["target"].to(device)
                output = model(t2)
                if weight_enabled:
                    loss = compute_weighted_scalar_loss(output, target, loss_type, beta, weight_cfg)
                else:
                    loss = criterion(output, target)
                val_losses.append(loss.item())
                metrics = compute_scalar_metrics(output, target)
                for k, v in metrics.items():
                    val_metrics[k].append(v)

        train_loss = float(np.mean(train_losses)) if train_losses else 0.0
        val_loss = float(np.mean(val_losses)) if val_losses else 0.0
        train_mae = float(np.mean(train_metrics["mae"])) if train_metrics["mae"] else 0.0
        val_mae = float(np.mean(val_metrics["mae"])) if val_metrics["mae"] else 0.0

        if writer:
            writer.add_scalar("train/loss_epoch", train_loss, epoch)
            writer.add_scalar("val/loss_epoch", val_loss, epoch)
            writer.add_scalar("train/mae_epoch", train_mae, epoch)
            writer.add_scalar("val/mae_epoch", val_mae, epoch)

        is_best = val_loss < best_val_loss
        if is_best:
            best_val_loss = val_loss

        checkpoint_state = {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "best_val_loss": best_val_loss,
            "config": config,
        }

        if rank == 0:
            save_checkpoint(checkpoint_state, is_best=is_best, checkpoint_dir=output_dir)

        if rank == 0:
            print(
                f"Epoch {epoch} - train loss {train_loss:.4f} - val loss {val_loss:.4f} "
                f"- val mae {val_mae:.4f}"
            )

        if early_stopping(val_loss) and rank == 0:
            print("Early stopping triggered.")
            break

    if ddp_enabled:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
