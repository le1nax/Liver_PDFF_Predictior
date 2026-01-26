# Liver Fat Fraction Predictor

A PyTorch-based 3D U-Net model for predicting fat fraction maps from T2 MRI liver images. The model uses paired T2 MRI and Dixon quantitative fat fraction data for training, with liver segmentation masks to focus on relevant regions.

## Featuresv

- **3D U-Net Architecture**: Spatial encoder-decoder for volumetric fat fraction regression
- **Automatic Preprocessing**: DICOM loading, liver segmentation, spatial alignment, and normalization
- **Masked Loss Functions**: Use liver segmentation to focus training on liver regions
- **Flexible Loss Options**: MSE, L1, Smooth L1, SSIM, and combined losses
- **Comprehensive Metrics**: MAE, RMSE, Pearson correlation, Bland-Altman analysis
- **Full Pipeline**: Preprocessing, training, validation, inference, and visualization

## Quick Links

- **[CONFIG_GUIDE.md](CONFIG_GUIDE.md)** - Configuration guide (start here!)
- **[DATA_STRUCTURE.md](DATA_STRUCTURE.md)** - How to organize your data
- **[PREPROCESSING_NOTES.md](PREPROCESSING_NOTES.md)** - Details on fat fraction normalization
- **[config/config_pixel_level/train_config.yaml](config/config_pixel_level/train_config.yaml)** - Training configuration
- **[config/config_pixel_level/inference_config.yaml](config/config_pixel_level/inference_config.yaml)** - Inference configuration
- **[config/config_scalar_regression/train_config.yaml](config/config_scalar_regression/train_config.yaml)** - Scalar regression training configuration
- **[config/config_scalar_regression/inference_config.yaml](config/config_scalar_regression/inference_config.yaml)** - Scalar regression inference configuration

## Installation

```bash
# Clone the repository
cd Liver_FF_Predictor

# Install dependencies
pip install -r requirements.txt
```

**Note:** You'll also need a trained nnU-Net model for liver segmentation. See `segmentation_module.py` for setup instructions.

## Pipeline Overview

### 1. Preprocessing (DICOM → NIfTI)

Process raw DICOM files to create aligned, normalized training data:

```bash
python process_and_align_patients.py \
    --input_dir ./raw_patient_data \
    --output_dir ./processed_data \
    --t2_sequence_name "MR HR_T2_AX_MVXD" \
    --dixon_sequence_name "MR mDIXONQuant_BH"
```

**What it does:**
1. Loads T2 and Fat Fraction volumes from DICOM
2. Generates liver segmentation on T2 using nnU-Net
3. Normalizes fat fraction to [0, 1] range
4. Spatially aligns T2 to FF resolution (XY crop + Z-matching)
5. Saves as NIfTI files

**Output per patient:**
```
processed_data/
└── patient_001/
    ├── patient_001_t2_aligned.nii.gz
    ├── patient_001_ff_normalized.nii.gz
    └── patient_001_segmentation.nii.gz
```

### 2. Data Organization

**No file movement needed!** Just point to your preprocessed output:

```yaml
# config/config_pixel_level/train_config.yaml
data:
  data_dir: "./processed_data"
  use_patient_subdirs: true
```

That's it! The dataset will read directly from:
```
processed_data/
├── patient_001/
│   ├── patient_001_t2_aligned.nii.gz
│   ├── patient_001_ff_normalized.nii.gz
│   └── patient_001_segmentation.nii.gz
├── patient_002/
│   └── ...
```

<details>
<summary>Alternative: Organize into flat directory (optional)</summary>

If you prefer a flat directory structure, use symbolic links (no data duplication):

```bash
python organize_data.py \
    --input_dir ./processed_data \
    --output_dir ./training_data \
    --method symlink
```

Then update config:
```yaml
data:
  data_dir: "./training_data"
  use_patient_subdirs: false
```

See [DATA_STRUCTURE.md](DATA_STRUCTURE.md) for more options.
</details>

**Data Requirements:**
- T2, FF, and segmentation must be spatially aligned (handled by preprocessing)
- Same dimensions and voxel spacing (handled by preprocessing)
- Fat fraction values in [0, 1] range where 0=0%, 1=100% (handled by preprocessing)
- Liver masks are binary (0 or 1)

### 3. Training

Edit [config/config_pixel_level/train_config.yaml](config/config_pixel_level/train_config.yaml) with your settings:

```yaml
data:
  data_dir: "./training_data"
  use_subdirs: false  # Single directory mode
  t2_suffix: "_t2_aligned"
  ff_suffix: "_ff_normalized"
  mask_suffix: "_segmentation"

  train_ratio: 0.70
  val_ratio: 0.15
  test_ratio: 0.15

model:
  type: "standard"  # or "lightweight"
  base_channels: 32

loss:
  type: "combined"
  mse_weight: 1.0
  ssim_weight: 0.1

training:
  num_epochs: 100
  batch_size: 2
  lr: 0.0001
```

**Train the model:**

```bash
# Uses config/config_pixel_level/train_config.yaml by default
python pixel_level_network/train.py

# Or specify a different config
python pixel_level_network/train.py --config config/my_custom_config.yaml
```

To resume training from a checkpoint:

```bash
python pixel_level_network/train.py --resume outputs/experiment_001/checkpoint_best.pth
```

### 4. Inference

Edit [config/config_pixel_level/inference_config.yaml](config/config_pixel_level/inference_config.yaml) with your settings:

```yaml
# Model checkpoint
checkpoint_path: "./outputs/experiment_001/checkpoint_best.pth"

# Choose mode: "single", "batch", or "validation"
mode: "single"

# Single file inference
single:
  t2_path: "/path/to/t2_image.nii.gz"
  output_path: "./predictions/prediction.nii.gz"
  gt_path: null  # Optional: add ground truth path
  visualize: true
```

**Run inference:**

```bash
# Uses config/config_pixel_level/inference_config.yaml by default
python pixel_level_network/inference.py

# Override mode from command line
python pixel_level_network/inference.py --mode batch

# Override checkpoint from command line
python pixel_level_network/inference.py --checkpoint outputs/experiment_002/checkpoint_best.pth
```

**Three inference modes:**

1. **Single file**: Predict one T2 image
   ```yaml
   mode: "single"
   single:
     t2_path: "/path/to/t2_image.nii.gz"
     output_path: "./predictions/prediction.nii.gz"
   ```

2. **Batch**: Predict all T2 images in a directory
   ```yaml
   mode: "batch"
   batch:
     input_dir: "/path/to/t2_images/"
     output_dir: "./predictions/"
   ```

3. **Validation**: Predict with ground truth comparison
   ```yaml
   mode: "validation"
   validation:
     data_dir: "/path/to/validation_data/"
     output_dir: "./validation_predictions/"
     use_patient_subdirs: true
   ```

## Model Architecture

### Standard U-Net3D
- 5 encoder levels with max pooling
- 4 decoder levels with skip connections
- Base channels: 32 (configurable)
- ~30M parameters

### Lightweight U-Net3D
- 4 encoder levels
- 3 decoder levels
- Base channels: 16 (configurable)
- ~4M parameters

## Loss Functions

Available loss functions in [losses.py](losses.py):

1. **MaskedMSELoss**: Mean squared error within liver mask
2. **MaskedL1Loss**: Mean absolute error within liver mask
3. **MaskedSmoothL1Loss**: Huber loss within liver mask
4. **SSIMLoss**: Structural similarity loss
5. **CombinedLoss**: Weighted combination of multiple losses
6. **WeightedMaskedLoss**: Additional weighting for high fat fraction regions

## Evaluation Metrics

Computed in [utils.py](utils.py):

- **MAE**: Mean Absolute Error
- **RMSE**: Root Mean Squared Error
- **Pearson Correlation**: Linear correlation coefficient
- **Bias**: Mean prediction error
- **Bland-Altman Plot**: Clinical validation

## Monitoring Training

View training progress with TensorBoard:

```bash
tensorboard --logdir outputs/experiment_001/logs
```

Metrics logged:
- Training/validation loss per epoch and step
- MAE, RMSE, correlation
- Learning rate
- Loss components (if using combined loss)

## Tips for Training

### Memory Issues

If you encounter out-of-memory errors:

1. Reduce batch size in config.yaml:
   ```yaml
   training:
     batch_size: 1
   ```

2. Use lightweight model:
   ```yaml
   model:
     type: "lightweight"
     base_channels: 16
   ```

3. Enable fat-aware volume sampling:
   ```yaml
   data:
     fat_sampling:
       enabled: true
       bin_edges: [0.0, 0.1, 0.2, 1.0]
       bin_probs: [0.2, 0.3, 0.5]
   ```

### Improving Performance

1. **Data augmentation**: Add augmentation in dataset.py
2. **Loss weighting**: Emphasize high fat fraction regions:
   ```yaml
   loss:
     type: "weighted"
     high_ff_threshold: 0.15
     high_ff_weight: 2.0
   ```

3. **Learning rate scheduling**: Use reduce_on_plateau or cosine annealing
4. **Gradient clipping**: Helps training stability

## File Structure

```
Liver_FF_Predictor/
├── config/
│   ├── config_pixel_level/
│   │   ├── train_config.yaml       # Pixel-level training configuration
│   │   └── inference_config.yaml   # Pixel-level inference configuration
│   └── config_scalar_regression/
│       ├── train_config.yaml       # Scalar regression training configuration
│       └── inference_config.yaml   # Scalar regression inference configuration
├── process_and_align_patients.py  # DICOM preprocessing pipeline
├── segmentation_module.py          # Liver segmentation (nnU-Net wrapper)
├── pixel_level_network/            # Pixel-level model and scripts
│   ├── model.py                    # 3D U-Net architectures
│   ├── train.py                    # Training script
│   └── inference.py                # Prediction script
├── scalar_regression/              # Scalar regression model and scripts
│   ├── model_scalar.py             # Scalar regressor architecture
│   ├── dataset_scalar.py           # Scalar dataset utilities
│   ├── train_scalar.py             # Scalar training script
│   └── inference_scalar.py         # Scalar inference script
├── dataset.py                      # Data loading (supports 2 directory modes)
├── losses.py                       # Masked loss functions
├── utils.py                        # Metrics, visualization, checkpointing
├── organize_data.py                # Optional data organization helper
├── requirements.txt                # Python dependencies
├── README.md                       # Main documentation (this file)
├── DATA_STRUCTURE.md               # Data organization guide
└── PREPROCESSING_NOTES.md          # Preprocessing details
```

## Complete Workflow Example

```bash
# 1. Preprocess DICOM data
python process_and_align_patients.py \
    --input_dir ./raw_dicoms \
    --output_dir ./processed_data

# 2. Update training config
# Edit config/config_pixel_level/train_config.yaml:
#   data_dir: "./processed_data"
#   use_patient_subdirs: true

# 3. Train
python pixel_level_network/train.py

# 4. Monitor training (in separate terminal)
tensorboard --logdir outputs/experiment_001/logs

# 5. Update inference config
# Edit config/config_pixel_level/inference_config.yaml:
#   checkpoint_path: "./outputs/experiment_001/checkpoint_best.pth"
#   mode: "single"
#   single:
#     t2_path: "/path/to/new_patient_t2.nii.gz"
#     output_path: "./prediction.nii.gz"

# 6. Run inference
python pixel_level_network/inference.py
```

## Citation

If you use this code, please cite:

- nnU-Net (for architecture inspiration): https://github.com/MIC-DKFZ/nnUNet
- Your fat fraction prediction paper (when published)

## License

[Specify your license here]

## Contact

[Your contact information]
