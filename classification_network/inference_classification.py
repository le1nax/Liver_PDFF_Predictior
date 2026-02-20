"""
Inference script for liver fat fraction classification.

Loads a trained checkpoint, runs inference on a dataset, and saves
per-patient predictions (class, class name, probabilities) as YAML.
"""

import argparse
import sys
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scalar_regression.dataset_scalar import ScalarInferenceDataset, pad_collate_scalar  # noqa: E402
from classification_network.model_classification import get_classification_model  # noqa: E402
from classification_network.dataset_classification import CLASS_NAMES  # noqa: E402


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Liver fat fraction classification inference")
    default_config = REPO_ROOT / "config" / "config_classification" / "inference_config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to inference config YAML",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device = torch.device(config.get("device", "cuda") if torch.cuda.is_available() else "cpu")

    checkpoint_path = Path(config["checkpoint_path"])
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    ckpt_config = checkpoint.get("config", {})
    model_cfg = ckpt_config.get("model", {})

    model = get_classification_model(
        in_channels=model_cfg.get("in_channels", 2),
        base_channels=model_cfg.get("base_channels", 16),
        num_classes=model_cfg.get("num_classes", 4),
        ordinal=model_cfg.get("ordinal", False),
    ).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    ordinal = model_cfg.get("ordinal", False)
    num_classes = model_cfg.get("num_classes", 4)

    data_dir = Path(config.get("data_dir", ckpt_config.get("data", {}).get("data_dir", "")))
    if not data_dir.is_absolute():
        data_dir = (REPO_ROOT / data_dir).resolve()

    data_cfg = config.get("data", {})
    dataset = ScalarInferenceDataset(
        data_dir=str(data_dir),
        t2_suffix=data_cfg.get("t2_suffix", "_t2_original"),
        use_subdirs=data_cfg.get("use_subdirs", False),
        use_patient_subdirs=data_cfg.get("use_patient_subdirs", True),
        t2_subdir=data_cfg.get("t2_subdir", "t2_images"),
        log_t2=data_cfg.get("log_t2", False),
        normalize_t2=data_cfg.get("normalize_t2", False),
    )

    loader = DataLoader(
        dataset,
        batch_size=config.get("batch_size", 2),
        shuffle=False,
        num_workers=config.get("num_workers", 2),
        collate_fn=pad_collate_scalar,
    )

    predictions = {}
    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            t2 = batch["t2"].to(device)
            # Use ones mask for inference dataset (no mask available)
            mask = torch.ones_like(t2)
            logits = model(t2, mask)

            if ordinal:
                probs = torch.sigmoid(logits)
                pred_classes = (probs > 0.5).sum(dim=1).long()
                # Build full class probabilities from ordinal cumulative probs
                full_probs = torch.zeros(logits.size(0), num_classes, device=device)
                cum_probs = torch.cat([
                    torch.ones(logits.size(0), 1, device=device),
                    probs,
                    torch.zeros(logits.size(0), 1, device=device),
                ], dim=1)
                for c in range(num_classes):
                    full_probs[:, c] = cum_probs[:, c] - cum_probs[:, c + 1]
                full_probs = full_probs.clamp(min=0.0)
            else:
                probs = torch.softmax(logits, dim=1)
                pred_classes = logits.argmax(dim=1)
                full_probs = probs

            for pid, cls_idx, prob_vec in zip(
                batch["patient_id"],
                pred_classes.cpu().numpy(),
                full_probs.cpu().numpy(),
            ):
                predictions[pid] = {
                    "predicted_class": int(cls_idx),
                    "class_name": CLASS_NAMES[int(cls_idx)],
                    "probabilities": {
                        CLASS_NAMES[i]: float(prob_vec[i])
                        for i in range(num_classes)
                    },
                }

    output_path = Path(config["output_path"])
    if not output_path.is_absolute():
        output_path = (REPO_ROOT / output_path).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        yaml.safe_dump(predictions, f, sort_keys=True)

    print(f"Saved predictions for {len(predictions)} patients to {output_path}")


if __name__ == "__main__":
    main()
