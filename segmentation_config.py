#!/usr/bin/env python3
"""
Configuration for liver segmentation models

You can switch between different trained models by modifying the active configuration.
"""

# Available model configurations
MODELS = {
    # Your newly trained Dataset001_CirrMRI model (2 classes: background, liver)
    'cirrmri_v1': {
        'name': 'CirrMRI Dataset v1',
        'checkpoint': '/home/homesOnMaster/dgeiger/datasets/nnUNet_results/Dataset001_CirrMRI/nnUNetTrainer__nnUNetPlans__3d_fullres/fold_0/checkpoint_best.pth',
        'description': 'nnU-Net trained on Dataset001_CirrMRI with 336 training cases',
        'classes': {
            0: 'background',
            1: 'liver'
        },
        'trainer': 'nnUNetTrainer',
        'plans': 'nnUNetPlans',
        'configuration': '3d_fullres',
        'fold': 0
    },

    # Old model (4 classes: background, liver, liver_no_parenchyma, spleen)
    'old_t2liver': {
        'name': 'Old T2 Liver Model',
        'checkpoint': '/home/homesOnMaster/dgeiger/WB/nnUnet_t2liver/checkpoint_best.pth',
        'description': 'Previous nnU-Net model with 4 classes',
        'classes': {
            0: 'background',
            1: 'liver',
            2: 'liver_no_parenchyma',
            3: 'spleen'
        },
        'trainer': 'nnUNetTrainer',
        'plans': 'nnUNetPlans',
        'configuration': '3d_fullres',
        'fold': None
    },
}

# Active model - change this to switch models
ACTIVE_MODEL = 'cirrmri_v1'

# Ensemble configuration (if using multiple folds)
USE_ENSEMBLE = False
ENSEMBLE_FOLDS = [0, 1, 2, 3, 4]  # Use all 5 folds for ensemble

def get_active_config():
    """Get the active model configuration"""
    return MODELS[ACTIVE_MODEL]

def get_checkpoint_path():
    """Get the checkpoint path for the active model"""
    return MODELS[ACTIVE_MODEL]['checkpoint']

def get_ensemble_checkpoints():
    """Get checkpoint paths for ensemble inference (all folds)"""
    if not USE_ENSEMBLE:
        return [get_checkpoint_path()]

    config = get_active_config()
    base_path = config['checkpoint'].rsplit('/fold_', 1)[0]

    checkpoints = []
    for fold in ENSEMBLE_FOLDS:
        ckpt_path = f"{base_path}/fold_{fold}/checkpoint_best.pth"
        checkpoints.append(ckpt_path)

    return checkpoints

def get_class_names():
    """Get class names for the active model"""
    return MODELS[ACTIVE_MODEL]['classes']

if __name__ == '__main__':
    print("Current Segmentation Model Configuration")
    print("=" * 80)
    config = get_active_config()
    print(f"Model: {config['name']}")
    print(f"Description: {config['description']}")
    print(f"Checkpoint: {config['checkpoint']}")
    print(f"\nClasses:")
    for class_id, class_name in config['classes'].items():
        print(f"  {class_id}: {class_name}")

    if USE_ENSEMBLE:
        print(f"\nEnsemble mode enabled with {len(ENSEMBLE_FOLDS)} folds")
        print("Checkpoints:")
        for ckpt in get_ensemble_checkpoints():
            print(f"  - {ckpt}")
