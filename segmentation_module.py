#!/usr/bin/env python3
"""
Segmentation module for T2 liver images using trained nnU-Net model

NOTE: This requires PyTorch to be installed:
    pip install torch

For nnU-Net v1: pip install nnunet
For nnU-Net v2: pip install nnunetv2

This module attempts to auto-detect which version you have.
"""

import os
import numpy as np
from scipy import ndimage

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("PyTorch not available - segmentation will not work")

# Try to import nnU-Net (v1 or v2)
NNUNET_VERSION = None
try:
    import nnunetv2
    NNUNET_VERSION = 2
    print("Detected nnU-Net v2")
except ImportError:
    try:
        import nnunet
        NNUNET_VERSION = 1
        print("Detected nnU-Net v1")
    except ImportError:
        print("nnU-Net not installed - you'll need to implement model loading manually")

from pathlib import Path

try:
    from segmentation_config import get_checkpoint_path, get_active_config
    DEFAULT_CHECKPOINT = get_checkpoint_path()
    print(f"Using model from config: {get_active_config()['name']}")
except ImportError:
    # Fallback if config not available
    DEFAULT_CHECKPOINT = '/home/homesOnMaster/dgeiger/datasets/nnUNet_results/Dataset001_CirrMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth'
    print("segmentation_config.py not found, using default checkpoint path")

class LiverSegmenter:
    """Wrapper for nnU-Net liver segmentation"""

    def __init__(self, checkpoint_path=None):
        """
        Initialize the segmenter with model weights

        Args:
            checkpoint_path: Path to the trained model checkpoint (if None, uses config default)
        """
        self.checkpoint_path = checkpoint_path or DEFAULT_CHECKPOINT
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        print(f"Segmentation device: {self.device}")
        print(f"Model checkpoint: {self.checkpoint_path}")

    def load_model(self):
        """Load the trained nnU-Net v2 model"""
        if self.model is not None:
            return

        if NNUNET_VERSION != 2:
            raise ImportError("nnU-Net v2 not installed. Run: pip install nnunetv2")

        try:
            from nnunetv2.utilities.plans_handling.plans_handler import PlansManager
            from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer

            print(f"Loading nnU-Net v2 model from {self.checkpoint_path}")
            checkpoint = torch.load(self.checkpoint_path, map_location=self.device, weights_only=False)

            print(f"✓ Checkpoint loaded: Epoch {checkpoint['current_epoch']}, EMA {checkpoint['_best_ema']:.4f}")

            # Extract initialization arguments
            init_args = checkpoint['init_args']
            plans = init_args['plans']
            configuration = init_args['configuration']

            print(f"  Configuration: {configuration}")
            print(f"  Trainer: {checkpoint['trainer_name']}")

            # Build the network using PlansManager
            plans_manager = PlansManager(plans)
            configuration_manager = plans_manager.get_configuration(configuration)
            dataset_json = init_args['dataset_json']

            # Get architecture information
            network_class_name = configuration_manager.network_arch_class_name
            arch_init_kwargs = configuration_manager.network_arch_init_kwargs
            arch_init_kwargs_req_import = configuration_manager.network_arch_init_kwargs_req_import

            # Get number of channels from dataset_json
            num_input_channels = len(dataset_json.get('channel_names', {}))
            num_output_channels = len(dataset_json.get('labels', {}))

            print(f"  Network: {network_class_name}")
            print(f"  Input channels: {num_input_channels}, Output channels: {num_output_channels}")

            # Import and build the network
            from nnunetv2.utilities.get_network_from_plans import get_network_from_plans

            self.model = get_network_from_plans(
                arch_class_name=network_class_name,
                arch_kwargs=arch_init_kwargs,
                arch_kwargs_req_import=arch_init_kwargs_req_import,
                input_channels=num_input_channels,
                output_channels=num_output_channels,
                allow_init=True,
                deep_supervision=True
            )

            # Load weights
            self.model.load_state_dict(checkpoint['network_weights'])
            self.model.to(self.device)
            self.model.eval()

            # Store configuration for inference
            self.plans_manager = plans_manager
            self.configuration = configuration
            self.configuration_manager = configuration_manager
            self.plans = plans
            self.dataset_json = dataset_json  # Store for later use

            print("✓ Model loaded and ready for inference!")

        except Exception as e:
            print(f"Error loading model: {e}")
            import traceback
            traceback.print_exc()
            raise

    def preprocess_volume(self, volume):
        """
        Preprocess T2 volume using nnU-Net v2 preprocessing

        Args:
            volume: numpy array of shape (D, H, W) - T2 volume

        Returns:
            Preprocessed tensor ready for model input
        """
        from nnunetv2.preprocessing.preprocessors.default_preprocessor import DefaultPreprocessor

        # nnU-Net expects (C, D, H, W) format - add channel dimension
        volume = volume.astype(np.float32)
        volume_with_channel = volume[np.newaxis, ...]  # (1, D, H, W)

        # Get normalization schemes from plans
        normalization_schemes = self.plans_manager.plans['foreground_intensity_properties_per_channel']

        # Apply nnU-Net preprocessing (z-score normalization based on training data)
        for c in range(volume_with_channel.shape[0]):
            scheme = normalization_schemes[str(c)]
            mean = scheme['mean']
            std = scheme['std']

            # Clip to percentiles (optional but recommended)
            lower = scheme.get('percentile_00_5', volume_with_channel[c].min())
            upper = scheme.get('percentile_99_5', volume_with_channel[c].max())
            volume_with_channel[c] = np.clip(volume_with_channel[c], lower, upper)

            # Z-score normalization
            if std > 0:
                volume_with_channel[c] = (volume_with_channel[c] - mean) / std

        # Add batch dimension: (1, C, D, H, W)
        volume_tensor = torch.from_numpy(volume_with_channel[np.newaxis, ...])

        return volume_tensor.to(self.device)

    def segment_volume(self, volume):
        """
        Segment a T2 volume using nnU-Net v2 with sliding window inference

        Args:
            volume: numpy array of shape (D, H, W) - T2 volume

        Returns:
            Segmentation mask as numpy array of shape (D, H, W) with class labels
        """
        if self.model is None:
            self.load_model()

        print(f"Running inference on volume with shape: {volume.shape}")

        # Preprocess - get normalized volume in (C, D, H, W) format
        volume = volume.astype(np.float32)
        volume_with_channel = volume[np.newaxis, ...]  # (1, D, H, W)

        # Get normalization schemes from plans
        normalization_schemes = self.plans_manager.plans['foreground_intensity_properties_per_channel']

        # Apply nnU-Net preprocessing (z-score normalization based on training data)
        for c in range(volume_with_channel.shape[0]):
            scheme = normalization_schemes[str(c)]
            mean = scheme['mean']
            std = scheme['std']

            # Clip to percentiles (optional but recommended)
            lower = scheme.get('percentile_00_5', volume_with_channel[c].min())
            upper = scheme.get('percentile_99_5', volume_with_channel[c].max())
            volume_with_channel[c] = np.clip(volume_with_channel[c], lower, upper)

            # Z-score normalization
            if std > 0:
                volume_with_channel[c] = (volume_with_channel[c] - mean) / std

        print(f"Preprocessed volume shape: {volume_with_channel.shape}")

        # Get patch size and other inference parameters from configuration
        patch_size = self.configuration_manager.patch_size
        print(f"Using patch size: {patch_size}")

        # Convert to tuple if it's a list
        if isinstance(patch_size, list):
            patch_size = tuple(patch_size)

        # Perform sliding window inference
        # The predictor handles arbitrary input sizes by using sliding windows
        with torch.no_grad():
            # Manually implement sliding window since we already have the model loaded
            # Use the configuration's patch size
            from nnunetv2.inference.sliding_window_prediction import compute_gaussian, compute_steps_for_sliding_window

            # Get Gaussian importance map for blending overlapping patches
            gaussian = compute_gaussian(patch_size, sigma_scale=1./8, dtype=torch.float16 if self.device.type == 'cuda' else torch.float32, device=self.device)

            # Compute sliding window steps
            steps = compute_steps_for_sliding_window(
                volume_with_channel.shape[1:],  # image shape (D, H, W)
                patch_size,
                0.5  # step_size: 0.5 = 50% overlap
            )

            print(f"Sliding window steps: {steps}")

            # Initialize aggregation arrays
            # Get number of classes from dataset_json (stored during model loading)
            num_classes = len(self.dataset_json['labels'])
            aggregated_results = torch.zeros(
                [num_classes] + list(volume_with_channel.shape[1:]),
                dtype=torch.float32,
                device=self.device
            )
            aggregated_nb_of_predictions = torch.zeros(
                [num_classes] + list(volume_with_channel.shape[1:]),
                dtype=torch.float32,
                device=self.device
            )

            # Convert volume to tensor
            volume_tensor = torch.from_numpy(volume_with_channel).float().to(self.device)

            # Iterate over sliding windows
            for d_start in steps[0]:
                d_end = min(d_start + patch_size[0], volume_with_channel.shape[1])
                for h_start in steps[1]:
                    h_end = min(h_start + patch_size[1], volume_with_channel.shape[2])
                    for w_start in steps[2]:
                        w_end = min(w_start + patch_size[2], volume_with_channel.shape[3])

                        # Extract patch
                        patch = volume_tensor[:, d_start:d_end, h_start:h_end, w_start:w_end]

                        # Pad if needed (at boundaries)
                        if patch.shape[1:] != tuple(patch_size):
                            # Pad to patch size
                            pad_d = patch_size[0] - patch.shape[1]
                            pad_h = patch_size[1] - patch.shape[2]
                            pad_w = patch_size[2] - patch.shape[3]
                            patch = torch.nn.functional.pad(patch, (0, pad_w, 0, pad_h, 0, pad_d), mode='constant', value=0)

                        # Add batch dimension
                        patch = patch[None, ...]  # (1, C, D, H, W)

                        # Forward pass
                        output = self.model(patch)

                        # Handle deep supervision
                        if isinstance(output, (list, tuple)):
                            output = output[0]

                        # Remove batch dimension and trim padding if any
                        output = output[0]  # (C, D, H, W)
                        actual_d = d_end - d_start
                        actual_h = h_end - h_start
                        actual_w = w_end - w_start
                        output = output[:, :actual_d, :actual_h, :actual_w]

                        # Get corresponding gaussian weights (trim if needed)
                        # Gaussian has shape (D, H, W), so we trim it to match the patch size
                        gauss_weights = gaussian[:actual_d, :actual_h, :actual_w]

                        # Accumulate predictions with Gaussian weighting
                        # Broadcast gaussian across the class dimension
                        aggregated_results[:, d_start:d_end, h_start:h_end, w_start:w_end] += output * gauss_weights[None, ...]
                        aggregated_nb_of_predictions[:, d_start:d_end, h_start:h_end, w_start:w_end] += gauss_weights[None, ...]

            # Average predictions
            aggregated_results /= aggregated_nb_of_predictions

            # Get class predictions (argmax over class dimension)
            predictions = torch.argmax(aggregated_results, dim=0)

            # Convert to numpy
            segmentation = predictions.cpu().numpy()

        print(f"Segmentation complete! Output shape: {segmentation.shape}")

        # Post-process: Keep only largest connected component
        print("Post-processing: Keeping only largest connected component per class...")
        segmentation = self.keep_largest_component(segmentation)

        return segmentation.astype(np.uint8)

    def keep_largest_component(self, mask):
        """
        Keep only the largest connected component for each class

        Args:
            mask: Segmentation mask with class labels (D, H, W)

        Returns:
            Filtered mask with only largest component per class
        """
        # Process each class separately (skip background class 0)
        filtered_mask = np.zeros_like(mask)

        unique_labels = np.unique(mask)
        unique_labels = unique_labels[unique_labels > 0]  # Skip background

        for label in unique_labels:
            # Get binary mask for this class
            binary_mask = (mask == label).astype(np.uint8)

            # Find connected components
            labeled_array, num_features = ndimage.label(binary_mask)

            if num_features == 0:
                continue

            # Find the largest component
            component_sizes = ndimage.sum(binary_mask, labeled_array, range(1, num_features + 1))
            largest_component_label = np.argmax(component_sizes) + 1

            # Keep only the largest component
            largest_component_mask = (labeled_array == largest_component_label)
            filtered_mask[largest_component_mask] = label

            print(f"  Class {label}: Kept largest component ({int(component_sizes[largest_component_label - 1])} voxels) out of {num_features} components")

        return filtered_mask

    def postprocess_mask(self, mask, original_shape):
        """
        Postprocess segmentation mask

        Args:
            mask: Segmentation mask array
            original_shape: Target shape to resize to

        Returns:
            Postprocessed mask
        """
        # Add any postprocessing (e.g., removing small components)
        return mask


def segment_t2_volume_simple(volume, checkpoint_path):
    """
    Simple function to segment a T2 volume

    Args:
        volume: numpy array (D, H, W)
        checkpoint_path: Path to model checkpoint

    Returns:
        Segmentation mask as numpy array
    """
    segmenter = LiverSegmenter(checkpoint_path)
    return segmenter.segment_volume(volume)


if __name__ == '__main__':
    # Test the segmenter
    print("Testing segmentation module...")

    # Create dummy volume
    dummy_volume = np.random.randn(20, 256, 256).astype(np.float32)

    try:
        segmenter = LiverSegmenter()
        segmenter.load_model()
        print("Model loaded successfully!")
    except Exception as e:
        print(f"Error: {e}")
        print("Note: You need to implement the actual model architecture to run inference.")
        print("The checkpoint contains weights, but you need the model structure.")
