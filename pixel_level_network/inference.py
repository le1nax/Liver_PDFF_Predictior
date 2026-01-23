"""
Inference script for fat fraction prediction on new T2 MRI images.
"""

import argparse
import yaml
import torch
import numpy as np
import nibabel as nib
from pathlib import Path
from tqdm import tqdm

from pixel_level_network.model import get_model
from utils import save_prediction_as_nifti, visualize_prediction, visualize_error_map


class FatFractionPredictor:
    """Predictor class for fat fraction estimation from T2 MRI."""

    def __init__(
        self,
        checkpoint_path: str,
        config_path: str = None,
        device: str = 'cuda'
    ):
        """
        Args:
            checkpoint_path: Path to model checkpoint
            config_path: Path to config file (optional, extracted from checkpoint)
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        # Get config
        if config_path:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
        elif 'config' in checkpoint:
            self.config = checkpoint['config']
        else:
            raise ValueError("Config not found in checkpoint and no config_path provided")

        # Build model
        model_config = self.config['model']
        self.model = get_model(
            model_type=model_config.get('type', 'standard'),
            n_channels=model_config.get('n_channels', 1),
            n_outputs=model_config.get('n_outputs', 1),
            base_channels=model_config.get('base_channels', 32),
            trilinear=model_config.get('trilinear', True)
        )

        # Load weights
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        print(f"Loaded model from {checkpoint_path}")
        print(f"Using device: {self.device}")

    def preprocess_t2(self, t2_data: np.ndarray) -> np.ndarray:
        """Preprocess T2 image (same as training)"""
        # Percentile normalization
        p1, p99 = np.percentile(t2_data, [1, 99])
        t2_norm = np.clip(t2_data, p1, p99)
        t2_norm = (t2_norm - p1) / (p99 - p1 + 1e-8)
        return t2_norm.astype(np.float32)

    @torch.no_grad()
    def predict(
        self,
        t2_path: str,
        output_path: str = None,
        visualize: bool = True,
        slice_idx: int = None
    ) -> np.ndarray:
        """
        Predict fat fraction from T2 MRI.

        Args:
            t2_path: Path to T2 NIfTI file
            output_path: Path to save prediction (optional)
            visualize: Whether to create visualization
            slice_idx: Slice to visualize (middle slice if None)

        Returns:
            Predicted fat fraction as numpy array
        """
        # Load T2 image
        print(f"Loading T2 image from {t2_path}")
        t2_nii = nib.load(t2_path)
        t2_data = t2_nii.get_fdata().astype(np.float32)

        # Preprocess
        t2_norm = self.preprocess_t2(t2_data)

        # Convert to tensor and add batch/channel dimensions
        t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).unsqueeze(0)  # [1, 1, D, H, W]
        t2_tensor = t2_tensor.to(self.device)

        # Predict
        print("Running prediction...")
        prediction = self.model(t2_tensor)

        # Convert to numpy
        pred_np = prediction.squeeze().cpu().numpy()  # [D, H, W]

        # Save if output path provided
        if output_path:
            print(f"Saving prediction to {output_path}")
            save_prediction_as_nifti(pred_np, t2_path, output_path)

        # Visualize if requested
        if visualize:
            vis_path = None
            if output_path:
                vis_path = str(Path(output_path).parent / f"{Path(output_path).stem}_visualization.png")

            visualize_prediction(
                t2=t2_norm,
                pred=pred_np,
                target=None,  # No ground truth
                mask=None,
                slice_idx=slice_idx,
                save_path=vis_path
            )

        return pred_np

    @torch.no_grad()
    def predict_with_ground_truth(
        self,
        t2_path: str,
        gt_path: str,
        output_path: str = None,
        visualize: bool = True,
        slice_idx: int = None
    ) -> tuple[np.ndarray, dict]:
        """
        Predict fat fraction and compare with ground truth.

        Args:
            t2_path: Path to T2 NIfTI file
            gt_path: Path to ground truth fat fraction NIfTI file
            output_path: Path to save prediction (optional)
            visualize: Whether to create visualizations
            slice_idx: Slice to visualize

        Returns:
            Tuple of (prediction, metrics_dict)
        """
        from utils import calculate_metrics

        # Load T2 and ground truth
        print(f"Loading T2 image from {t2_path}")
        t2_nii = nib.load(t2_path)
        t2_data = t2_nii.get_fdata().astype(np.float32)

        print(f"Loading ground truth from {gt_path}")
        gt_nii = nib.load(gt_path)
        gt_data = gt_nii.get_fdata().astype(np.float32)

        # Normalize ground truth if needed
        if gt_data.max() > 1.5:
            gt_data = gt_data / 100.0
        gt_data = np.clip(gt_data, 0, 1)

        # Preprocess T2
        t2_norm = self.preprocess_t2(t2_data)

        # Convert to tensors
        t2_tensor = torch.from_numpy(t2_norm).unsqueeze(0).unsqueeze(0)
        t2_tensor = t2_tensor.to(self.device)

        # Predict
        print("Running prediction...")
        prediction = self.model(t2_tensor)

        # Convert to numpy
        pred_np = prediction.squeeze().cpu().numpy()

        # Calculate metrics
        pred_tensor = torch.from_numpy(pred_np).unsqueeze(0).unsqueeze(0)
        gt_tensor = torch.from_numpy(gt_data).unsqueeze(0).unsqueeze(0)
        metrics = calculate_metrics(pred_tensor, gt_tensor)

        print("\nMetrics:")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  Correlation: {metrics['correlation']:.4f}")
        print(f"  Bias: {metrics['bias']:.4f}")

        # Save prediction
        if output_path:
            print(f"Saving prediction to {output_path}")
            save_prediction_as_nifti(pred_np, t2_path, output_path)

        # Visualize
        if visualize and output_path:
            vis_dir = Path(output_path).parent
            vis_prefix = Path(output_path).stem

            # Prediction vs ground truth
            visualize_prediction(
                t2=t2_norm,
                pred=pred_np,
                target=gt_data,
                mask=None,
                slice_idx=slice_idx,
                save_path=str(vis_dir / f"{vis_prefix}_comparison.png")
            )

            # Error map
            visualize_error_map(
                pred=pred_np,
                target=gt_data,
                mask=None,
                slice_idx=slice_idx,
                save_path=str(vis_dir / f"{vis_prefix}_error.png")
            )

        return pred_np, metrics

    @torch.no_grad()
    def predict_batch(
        self,
        input_dir: str,
        output_dir: str,
        pattern: str = "*.nii.gz"
    ):
        """
        Predict fat fraction for all T2 images in a directory.

        Args:
            input_dir: Directory containing T2 images
            output_dir: Directory to save predictions
            pattern: File pattern to match
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all T2 images
        t2_files = sorted(input_dir.glob(pattern))
        print(f"Found {len(t2_files)} T2 images")

        # Process each file
        for t2_file in tqdm(t2_files, desc="Processing"):
            output_path = output_dir / f"{t2_file.stem}_fat_fraction.nii.gz"
            self.predict(
                t2_path=str(t2_file),
                output_path=str(output_path),
                visualize=False
            )


def visualize_no_gt(t2: np.ndarray, pred: np.ndarray, slice_idx: int, save_path: str):
    """Modified visualization without ground truth"""
    import matplotlib.pyplot as plt

    if slice_idx is None:
        slice_idx = t2.shape[0] // 2

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # T2 image
    axes[0].imshow(t2[slice_idx], cmap='gray')
    axes[0].set_title('T2 MRI')
    axes[0].axis('off')

    # Predicted fat fraction
    im = axes[1].imshow(pred[slice_idx], cmap='hot', vmin=0, vmax=1)
    axes[1].set_title('Predicted Fat Fraction')
    axes[1].axis('off')
    plt.colorbar(im, ax=axes[1], fraction=0.046)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def main():
    parser = argparse.ArgumentParser(description='Predict fat fraction from T2 MRI')
    parser.add_argument('--config', type=str, default='config/config_pixel_level/inference_config.yaml',
                        help='Path to inference config file (default: config/config_pixel_level/inference_config.yaml)')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Override checkpoint path from config')
    parser.add_argument('--mode', type=str, default=None,
                        choices=['single', 'batch', 'validation', 'yaml'],
                        help='Override mode from config')
    args = parser.parse_args()

    # Load inference config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override checkpoint if provided
    checkpoint_path = args.checkpoint if args.checkpoint else config['checkpoint_path']
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Get mode
    mode = args.mode if args.mode else config.get('mode', 'single')

    # Get device
    device = config.get('device', 'cuda')

    # Initialize predictor
    print(f"Loading model from: {checkpoint_path}")
    predictor = FatFractionPredictor(
        checkpoint_path=checkpoint_path,
        config_path=None,  # Use config from checkpoint
        device=device
    )

    print(f"\nRunning in '{mode}' mode...")

    if mode == 'single':
        # Single file inference
        single_config = config['single']
        t2_path = single_config['t2_path']
        output_path = single_config['output_path']
        gt_path = single_config.get('gt_path', None)
        visualize = single_config.get('visualize', True)

        if not Path(t2_path).exists():
            raise FileNotFoundError(f"T2 file not found: {t2_path}")

        # Create output directory
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)

        if gt_path and Path(gt_path).exists():
            # With ground truth
            predictor.predict_with_ground_truth(
                t2_path=t2_path,
                gt_path=gt_path,
                output_path=output_path,
                visualize=visualize
            )
        else:
            # Without ground truth
            predictor.predict(
                t2_path=t2_path,
                output_path=output_path,
                visualize=visualize
            )

    elif mode == 'batch':
        # Batch inference
        batch_config = config['batch']
        input_dir = batch_config['input_dir']
        output_dir = batch_config['output_dir']
        t2_suffix = batch_config.get('t2_suffix', '_t2_aligned')

        if not Path(input_dir).exists():
            raise FileNotFoundError(f"Input directory not found: {input_dir}")

        predictor.predict_batch(
            input_dir=input_dir,
            output_dir=output_dir,
            pattern=f"*{t2_suffix}.nii.gz"
        )

    elif mode == 'validation':
        # Validation inference (with ground truth)
        val_config = config['validation']
        data_dir = Path(val_config['data_dir'])
        output_dir = Path(val_config['output_dir'])
        use_patient_subdirs = val_config.get('use_patient_subdirs', True)
        t2_suffix = val_config.get('t2_suffix', '_t2_aligned')
        ff_suffix = val_config.get('ff_suffix', '_ff_normalized')

        if not data_dir.exists():
            raise FileNotFoundError(f"Data directory not found: {data_dir}")

        output_dir.mkdir(parents=True, exist_ok=True)

        # Find all patient files
        if use_patient_subdirs:
            patient_dirs = [d for d in data_dir.iterdir() if d.is_dir()]
            print(f"Found {len(patient_dirs)} patient directories")

            for patient_dir in tqdm(patient_dirs, desc="Processing patients"):
                patient_id = patient_dir.name
                t2_file = patient_dir / f"{patient_id}{t2_suffix}.nii.gz"
                ff_file = patient_dir / f"{patient_id}{ff_suffix}.nii.gz"
                output_file = output_dir / f"{patient_id}_prediction.nii.gz"

                if t2_file.exists() and ff_file.exists():
                    predictor.predict_with_ground_truth(
                        t2_path=str(t2_file),
                        gt_path=str(ff_file),
                        output_path=str(output_file),
                        visualize=True
                    )
        else:
            # Flat directory structure
            t2_files = sorted(data_dir.glob(f"*{t2_suffix}.nii.gz"))
            print(f"Found {len(t2_files)} T2 files")

            for t2_file in tqdm(t2_files, desc="Processing files"):
                patient_id = t2_file.name.replace(f"{t2_suffix}.nii.gz", "")
                ff_file = data_dir / f"{patient_id}{ff_suffix}.nii.gz"
                output_file = output_dir / f"{patient_id}_prediction.nii.gz"

                if ff_file.exists():
                    predictor.predict_with_ground_truth(
                        t2_path=str(t2_file),
                        gt_path=str(ff_file),
                        output_path=str(output_file),
                        visualize=True
                    )

    elif mode == 'yaml':
        # YAML-based inference (reads test cases from YAML file)
        yaml_config = config['yaml']
        inference_yaml = yaml_config['inference_yaml']
        output_dir = Path(yaml_config['output_dir'])

        if not Path(inference_yaml).exists():
            raise FileNotFoundError(f"Inference YAML not found: {inference_yaml}")

        # Load test cases
        with open(inference_yaml, 'r') as f:
            test_data = yaml.safe_load(f)

        test_cases = test_data['test_cases']
        print(f"Loaded {len(test_cases)} test cases from {inference_yaml}")

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Store results for output YAML
        results = []
        all_metrics = []

        # Process each test case
        for case in tqdm(test_cases, desc="Processing test cases"):
            patient_id = case['patient_id']
            t2_path = case['t2_image']
            gt_path = case['ground_truth']
            mask_path = case.get('mask', None)

            # Define output path
            output_file = output_dir / f"{patient_id}_prediction.nii.gz"

            # Run prediction with ground truth
            if Path(gt_path).exists():
                pred_np, metrics = predictor.predict_with_ground_truth(
                    t2_path=t2_path,
                    gt_path=gt_path,
                    output_path=str(output_file),
                    visualize=False  # Skip visualization to speed up
                )
                all_metrics.append(metrics)
            else:
                # No ground truth available
                pred_np = predictor.predict(
                    t2_path=t2_path,
                    output_path=str(output_file),
                    visualize=False
                )
                metrics = None

            # Build result entry
            result_entry = {
                'patient_id': str(patient_id),
                't2_image': t2_path,
                'ground_truth': gt_path,
                'prediction': str(output_file.absolute())
            }

            if mask_path:
                result_entry['mask'] = mask_path

            if metrics:
                result_entry['metrics'] = {
                    'mae': float(metrics['mae']),
                    'rmse': float(metrics['rmse']),
                    'correlation': float(metrics['correlation']),
                    'bias': float(metrics['bias'])
                }

            results.append(result_entry)

        # Save results YAML
        results_yaml_path = output_dir / 'results.yaml'
        output_data = {
            'inference_config': {
                'checkpoint': checkpoint_path,
                'input_yaml': inference_yaml,
                'output_dir': str(output_dir.absolute())
            },
            'results': results
        }

        # Add summary statistics if metrics available
        if all_metrics:
            avg_metrics = {
                'mean_mae': float(np.mean([m['mae'] for m in all_metrics])),
                'mean_rmse': float(np.mean([m['rmse'] for m in all_metrics])),
                'mean_correlation': float(np.mean([m['correlation'] for m in all_metrics])),
                'mean_bias': float(np.mean([m['bias'] for m in all_metrics])),
                'std_mae': float(np.std([m['mae'] for m in all_metrics])),
                'std_rmse': float(np.std([m['rmse'] for m in all_metrics]))
            }
            output_data['summary'] = avg_metrics

            print("\n" + "="*50)
            print("Summary Statistics:")
            print("="*50)
            print(f"Mean MAE:         {avg_metrics['mean_mae']:.4f} ± {avg_metrics['std_mae']:.4f}")
            print(f"Mean RMSE:        {avg_metrics['mean_rmse']:.4f} ± {avg_metrics['std_rmse']:.4f}")
            print(f"Mean Correlation: {avg_metrics['mean_correlation']:.4f}")
            print(f"Mean Bias:        {avg_metrics['mean_bias']:.4f}")
            print("="*50)

        # Use custom representer to force patient IDs as quoted strings
        def str_representer(dumper, data):
            if data.isdigit():
                return dumper.represent_scalar('tag:yaml.org,2002:str', data, style="'")
            return dumper.represent_scalar('tag:yaml.org,2002:str', data)

        yaml.add_representer(str, str_representer)

        with open(results_yaml_path, 'w') as f:
            yaml.dump(output_data, f, default_flow_style=False, sort_keys=False)

        print(f"\n✓ Saved results to: {results_yaml_path}")

    print("\n✓ Inference complete!")


if __name__ == "__main__":
    main()
