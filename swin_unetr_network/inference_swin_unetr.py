"""
Inference script for Swin-UNETR fat fraction prediction on new T2 MRI images.

Based on pixel_level_network/inference.py but uses SwinUNETRWrapper.
No cropping — runs on full volumes (model handles padding internally).
"""

import argparse
import sys
import yaml
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from swin_unetr_network.model_swin_unetr import get_model
from utils import save_prediction_as_nifti, visualize_prediction, visualize_error_map


class FatFractionPredictor:
    """Predictor class for Swin-UNETR fat fraction estimation from T2 MRI."""

    def __init__(self, checkpoint_path: str, config_path: str = None, device: str = 'cuda'):
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and no config_path provided")

        # Build model — forward full model config dict
        model_config = self.config['model']
        kwargs = {k: v for k, v in model_config.items() if k != 'type'}
        self.model = get_model(**kwargs)

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")

    def preprocess_t2(self, t2_data: np.ndarray) -> np.ndarray:
        p1, p99 = np.percentile(t2_data, [1, 99])
        t2_norm = np.clip(t2_data, p1, p99)
        t2_norm = (t2_norm - p1) / (p99 - p1 + 1e-8)
        return t2_norm.astype(np.float32)

    @torch.no_grad()
    def predict(self, t2_path: str, output_path: str = None,
                visualize: bool = True, slice_idx: int = None) -> np.ndarray:
        print(f"Loading T2 image from {t2_path}")
        t2_nii = nib.load(t2_path)
        t2_data = t2_nii.get_fdata().astype(np.float32)

        t2_norm = self.preprocess_t2(t2_data)
        t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        print("Running prediction...")
        prediction = self.model(t2_tensor)
        pred_np = prediction.squeeze().cpu().numpy()

        if output_path:
            print(f"Saving prediction to {output_path}")
            save_prediction_as_nifti(pred_np, t2_path, output_path)

        if visualize:
            vis_path = None
            if output_path:
                vis_path = str(Path(output_path).parent / f"{Path(output_path).stem}_visualization.png")
            visualize_prediction(
                t2=t2_norm, pred=pred_np, target=None, mask=None,
                slice_idx=slice_idx, save_path=vis_path
            )

        return pred_np

    @torch.no_grad()
    def predict_with_ground_truth(self, t2_path: str, gt_path: str,
                                   output_path: str = None, visualize: bool = True,
                                   slice_idx: int = None) -> tuple:
        from utils import calculate_metrics

        print(f"Loading T2 image from {t2_path}")
        t2_nii = nib.load(t2_path)
        t2_data = t2_nii.get_fdata().astype(np.float32)

        print(f"Loading ground truth from {gt_path}")
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata().astype(np.float32)

        if gt_data.max() > 1.5:
            gt_data = gt_data / 100.0
        gt_data = np.clip(gt_data, 0, 1)

        t2_norm = self.preprocess_t2(t2_data)
        t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).unsqueeze(0).to(self.device)

        print("Running prediction...")
        prediction = self.model(t2_tensor)
        pred_np = prediction.squeeze().cpu().numpy()

        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
        metrics = calculate_metrics(pred_tensor, gt_tensor)

        print(f"\nMetrics: MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}, "
              f"Correlation={metrics['correlation']:.4f}, Bias={metrics['bias']:.4f}")

        if output_path:
            save_prediction_as_nifti(pred_np, t2_path, output_path)

        if visualize and output_path:
            vis_dir = Path(output_path).parent
            vis_prefix = Path(output_path).stem
            visualize_prediction(
                t2=t2_norm, pred=pred_np, target=gt_data, mask=None,
                slice_idx=slice_idx,
                save_path=str(vis_dir / f"{vis_prefix}_comparison.png")
            )
            visualize_error_map(
                pred=pred_np, target=gt_data, mask=None,
                slice_idx=slice_idx,
                save_path=str(vis_dir / f"{vis_prefix}_error.png")
            )

        return pred_np, metrics

    @torch.no_grad()
    def predict_batch(self, input_dir: str, output_dir: str, pattern: str = "*.nii.gz"):
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        t2_files = sorted(input_dir.glob(pattern))
        print(f"Found {len(t2_files)} T2 images")

        for t2_file in tqdm(t2_files, desc="Processing"):
            output_path = output_dir / f"{t2_file.stem}_fat_fraction.nii.gz"
            self.predict(t2_path=str(t2_file), output_path=str(output_path), visualize=False)


def main():
    parser = argparse.ArgumentParser(description='Swin-UNETR fat fraction inference')
    parser.add_argument('--config', type=str,
                        default='config/config_swin_unetr/inference_config.yaml',
                        help='Path to inference config file')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path from config')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['single', 'batch', 'validation', 'yaml'],
                        help='Override mode from config')
    args = parser.parse_args()

    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = (REPO_ROOT / config_path).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    checkpoint_path = args.checkpoint if args.checkpoint else config['checkpoint_path']
    if not Path(checkpoint_path).is_absolute():
        checkpoint_path = str((REPO_ROOT / checkpoint_path).resolve())
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    mode = args.mode if args.mode else config.get('mode', 'single')
    device = config.get('device', 'cuda')

    predictor = FatFractionPredictor(checkpoint_path=checkpoint_path, device=device)

    print(f"\nRunning in '{mode}' mode...")

    if mode == 'single':
        single_config = config['single']
        t2_path = single_config['t2_path']
        output_path = single_config['output_path']
        gt_path = single_config.get('gt_path', None)
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if gt_path and Path(gt_path).exists():
            predictor.predict_with_ground_truth(
                t2_path=t2_path, gt_path=gt_path, output_path=output_path)
        else:
            predictor.predict(t2_path=t2_path, output_path=output_path)

    elif mode == 'batch':
        batch_config = config['batch']
        predictor.predict_batch(
            input_dir=batch_config['input_dir'],
            output_dir=batch_config['output_dir'],
            pattern=f"*{batch_config.get('t2_suffix', '_t2_aligned')}.nii.gz"
        )

    elif mode == 'yaml':
        yaml_config = config['yaml']
        inference_yaml = yaml_config['inference_yaml']
        output_dir = Path(yaml_config['output_dir'])

        with open(inference_yaml, 'r') as f:
            test_data = yaml.safe_load(f)

        test_cases = test_data['test_cases']
        output_dir.mkdir(parents=True, exist_ok=True)
        all_metrics = []

        for case in tqdm(test_cases, desc="Processing test cases"):
            patient_id = case['patient_id']
            output_file = output_dir / f"{patient_id}_prediction.nii.gz"

            if Path(case['ground_truth']).exists():
                _, metrics = predictor.predict_with_ground_truth(
                    t2_path=case['t2_image'], gt_path=case['ground_truth'],
                    output_path=str(output_file), visualize=False)
                all_metrics.append(metrics)
            else:
                predictor.predict(
                    t2_path=case['t2_image'], output_path=str(output_file), visualize=False)

        if all_metrics:
            print(f"\nMean MAE: {np.mean([m['mae'] for m in all_metrics]):.4f}")
            print(f"Mean RMSE: {np.mean([m['rmse'] for m in all_metrics]):.4f}")

    print("\nInference complete!")


if __name__ == "__main__":
    main()
