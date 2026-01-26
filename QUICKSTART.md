# Quick Start Guide

Get up and running with the Liver Fat Fraction Predictor in 4 steps.

## Prerequisites

- Python 3.8+
- CUDA-capable GPU (recommended)
- Raw DICOM files: T2 MRI + Dixon Fat Fraction sequences
- Trained nnU-Net liver segmentation model

---

## Step 1: Install Dependencies

```bash
cd Liver_FF_Predictor
pip install -r requirements.txt
```

---

## Step 2: Preprocess DICOM Data

Convert raw DICOM to aligned, normalized NIfTI files:

```bash
python process_and_align_patients.py \
    --input_dir /path/to/raw_dicoms \
    --output_dir ./processed_data
```

**Input directory structure:**
```
raw_dicoms/
├── patient_001/
│   ├── MR HR_T2_AX_MVXD/        # T2 sequence
│   │   └── [DICOM files]
│   └── MR mDIXONQuant_BH/       # Dixon sequence
│       └── [DICOM files]
├── patient_002/
│   └── ...
```

**Output:**
```
processed_data/
├── patient_001/
│   ├── patient_001_t2_aligned.nii.gz
│   ├── patient_001_ff_normalized.nii.gz  # [0,1] range
│   └── patient_001_segmentation.nii.gz
├── patient_002/
│   └── ...
```

---

## Step 3: Configure Training

Edit `config/config_pixel_level/train_config.yaml` to point to your preprocessed data:

```yaml
data:
  data_dir: "./processed_data"
  use_patient_subdirs: true  # Read directly from preprocessing output

output_dir: "./outputs/experiment_001"
```

**That's it!** No file copying or reorganization needed.

---

## Step 4: Train

```bash
# Train (uses config/config_pixel_level/train_config.yaml automatically)
python pixel_level_network/train.py

# Monitor (in separate terminal)
tensorboard --logdir outputs/experiment_001/logs
```

**Training will:**
- Automatically create train/val/test splits
- Save checkpoints every 10 epochs + best model
- Log metrics to TensorBoard
- Use early stopping if validation loss plateaus

---

## Run Inference

After training completes, edit `config/config_pixel_level/inference_config.yaml`:

```yaml
checkpoint_path: "./outputs/experiment_001/checkpoint_best.pth"
mode: "single"

single:
  t2_path: "/path/to/new_t2.nii.gz"
  output_path: "./prediction_ff.nii.gz"
  visualize: true
```

Then run:

```bash
# Uses config/config_pixel_level/inference_config.yaml automatically
python pixel_level_network/inference.py
```

Output will be a fat fraction map in [0, 1] range (multiply by 100 for percentage).

---

## Troubleshooting

### Out of Memory?

**Option 1:** Reduce batch size
```yaml
training:
  batch_size: 1
```

**Option 2:** Use lightweight model
```yaml
model:
  type: "lightweight"
  base_channels: 16
```

**Option 3:** Enable fat-aware volume sampling
```yaml
data:
  fat_sampling:
    enabled: true
    bin_edges: [0.0, 0.1, 0.2, 1.0]
    bin_probs: [0.2, 0.3, 0.5]
```

### Preprocessing Failed?

Check that:
- DICOM directories match sequence names in arguments
- T2 and Dixon sequences exist for each patient
- nnU-Net segmentation model is properly installed

### Poor Performance?

- Increase training epochs (default: 100)
- Try combined loss: `mse_weight: 1.0, ssim_weight: 0.1`
- Ensure preprocessing normalized FF correctly (check logs for "0.0000 to 1.0000")
- Verify liver masks look correct using a NIfTI viewer

---

## Next Steps

- **Monitor training:** Open TensorBoard to track loss curves
- **Validate results:** Use inference with ground truth to check metrics
- **Tune hyperparameters:** Adjust learning rate, loss weights, model size
- **Read documentation:** See README.md and DATA_STRUCTURE.md for details

---

## File Checklist

After each step, verify:

**After Step 2 (Preprocessing):**
- [ ] Each patient has 3 files: t2_aligned, ff_normalized, segmentation
- [ ] Fat fraction values in [0, 1] range (check preprocessing logs)
- [ ] All volumes have same dimensions per patient

**After Step 3 (Configuration):**
- [ ] config/config_pixel_level/train_config.yaml points to processed_data/
- [ ] use_patient_subdirs set to true in config
- [ ] output_dir configured

**After Step 4 (Training):**
- [ ] checkpoint_best.pth exists in output directory
- [ ] TensorBoard shows decreasing loss
- [ ] Validation metrics improve over time
