"""
Inference script for scalar fat fraction prediction from raw 3D T2 volumes.
"""

import argparse
from pathlib import Path

import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

from scalar_regression.dataset_scalar import ScalarInferenceDataset, pad_collate_scalar
from scalar_regression.model_scalar import get_scalar_model


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run scalar fat fraction inference")
    default_config = Path(__file__).resolve().parents[1] / "config" / "config_scalar_regression" / "inference_config.yaml"
    parser.add_argument(
        "--config",
        type=str,
        default=str(default_config),
        help="Path to config file (default: config/config_scalar_regression/inference_config.yaml)",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (Path(__file__).resolve().parents[1] / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    device_name = config.get("device", "cuda")
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")

    checkpoint = torch.load(config["checkpoint_path"], map_location=device, weights_only=False)
    model_config = checkpoint.get("config", {}).get("model", {})
    base_channels = model_config.get("base_channels", 16)

    model = get_scalar_model(in_channels=model_cfg.get("in_channels", 2), base_channels=base_channels).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    data_dir = Path(config["data_dir"])
    if not data_dir.is_absolute():
        data_dir = (Path(__file__).resolve().parents[1] / data_dir).resolve()

    dataset = ScalarInferenceDataset(
        data_dir=str(data_dir),
        t2_suffix=config.get("data", {}).get("t2_suffix", "_t2_original"),
        use_subdirs=config.get("data", {}).get("use_subdirs", False),
        use_patient_subdirs=config.get("data", {}).get("use_patient_subdirs", True),
        t2_subdir=config.get("data", {}).get("t2_subdir", "t2_images"),
        log_t2=config.get("data", {}).get("log_t2", False),
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
            output = model(t2).squeeze(1).cpu().numpy()
            for patient_id, pred in zip(batch["patient_id"], output):
                predictions[patient_id] = float(pred)

    output_path = Path(config["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        yaml.safe_dump(predictions, f, sort_keys=True)

    print(f"Saved predictions to {output_path}")


if __name__ == "__main__":
    main()
