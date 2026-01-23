"""
Training script for fat fraction prediction model.
"""

import os
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import numpy as np
from datetime import datetime

from pixel_level_network.model import get_model
from dataset import get_dataloaders, create_data_splits, load_data_splits
from losses import get_loss_function, CombinedLoss
from utils import calculate_metrics, save_checkpoint, load_checkpoint, EarlyStopping


class Trainer:
    """Training manager for fat fraction prediction."""

    def __init__(self, config: dict):
        """
        Args:
            config: Configuration dictionary
        """
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Create output directory
        self.output_dir = Path(config['output_dir'])
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self.writer = SummaryWriter(log_dir=str(self.output_dir / 'logs'))

        # Initialize model
        self.model = self._build_model()
        self.model = self.model.to(self.device)

        # Initialize loss function
        self.criterion = get_loss_function(config['loss'])

        # Initialize optimizer
        self.optimizer = self._build_optimizer()

        # Initialize scheduler
        self.scheduler = self._build_scheduler()

        # Initialize early stopping
        self.early_stopping = EarlyStopping(
            patience=config['training'].get('early_stopping_patience', 20),
            min_delta=config['training'].get('early_stopping_delta', 1e-4)
        )

        # Tracking variables
        self.current_epoch = 0
        self.best_val_loss = float('inf')
        self.train_losses = []
        self.val_losses = []

        print(f"Using device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"Output directory: {self.output_dir}")

    def _build_model(self) -> nn.Module:
        """Build model from config"""
        model_config = self.config['model']
        model = get_model(
            model_type=model_config.get('type', 'standard'),
            n_channels=model_config.get('n_channels', 1),
            n_outputs=model_config.get('n_outputs', 1),
            base_channels=model_config.get('base_channels', 32),
            trilinear=model_config.get('trilinear', True)
        )
        return model

    def _build_optimizer(self) -> optim.Optimizer:
        """Build optimizer from config"""
        opt_config = self.config['optimizer']
        opt_type = opt_config.get('type', 'adam').lower()

        if opt_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 1e-5),
                betas=opt_config.get('betas', (0.9, 0.999))
            )
        elif opt_type == 'adamw':
            optimizer = optim.AdamW(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-4),
                weight_decay=opt_config.get('weight_decay', 1e-2)
            )
        elif opt_type == 'sgd':
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=opt_config.get('lr', 1e-3),
                momentum=opt_config.get('momentum', 0.9),
                weight_decay=opt_config.get('weight_decay', 1e-4)
            )
        else:
            raise ValueError(f"Unknown optimizer type: {opt_type}")

        return optimizer

    def _build_scheduler(self):
        """Build learning rate scheduler from config"""
        if 'scheduler' not in self.config:
            return None

        sched_config = self.config['scheduler']
        sched_type = sched_config.get('type', 'none').lower()

        if sched_type == 'none':
            return None
        elif sched_type == 'step':
            return optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=sched_config.get('step_size', 30),
                gamma=sched_config.get('gamma', 0.1)
            )
        elif sched_type == 'cosine':
            return optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config['training']['num_epochs'],
                eta_min=sched_config.get('eta_min', 1e-6)
            )
        elif sched_type == 'reduce_on_plateau':
            return optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=sched_config.get('factor', 0.5),
                patience=sched_config.get('patience', 10)
            )
        else:
            raise ValueError(f"Unknown scheduler type: {sched_type}")

    def train_epoch(self, train_loader) -> dict:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = []
        epoch_metrics = {'mae': [], 'rmse': []}

        pbar = tqdm(train_loader, desc=f"Epoch {self.current_epoch}")
        for batch_idx, batch in enumerate(pbar):
            # Move data to device
            t2 = batch['t2'].to(self.device)
            target = batch['fat_fraction'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            output = self.model(t2)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, loss_dict = self.criterion(output, target, mask)
            else:
                loss = self.criterion(output, target, mask)
                loss_dict = {'total': loss.item()}

            # Backward pass
            loss.backward()

            # Gradient clipping
            if self.config['training'].get('gradient_clip', 0) > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config['training']['gradient_clip']
                )

            self.optimizer.step()

            # Calculate metrics
            with torch.no_grad():
                metrics = calculate_metrics(output, target, mask)

            # Track losses and metrics
            epoch_losses.append(loss.item())
            epoch_metrics['mae'].append(metrics['mae'])
            epoch_metrics['rmse'].append(metrics['rmse'])

            # Update progress bar
            pbar.set_postfix({
                'loss': f"{loss.item():.4f}",
                'mae': f"{metrics['mae']:.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.6f}"
            })

            # Log to tensorboard
            global_step = self.current_epoch * len(train_loader) + batch_idx
            self.writer.add_scalar('train/loss_step', loss.item(), global_step)

        # Average metrics over epoch
        avg_loss = np.mean(epoch_losses)
        avg_mae = np.mean(epoch_metrics['mae'])
        avg_rmse = np.mean(epoch_metrics['rmse'])

        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse
        }

    @torch.no_grad()
    def validate(self, val_loader) -> dict:
        """Validate model"""
        self.model.eval()
        val_losses = []
        val_metrics = {'mae': [], 'rmse': [], 'correlation': []}

        for batch in tqdm(val_loader, desc="Validation"):
            # Move data to device
            t2 = batch['t2'].to(self.device)
            target = batch['fat_fraction'].to(self.device)
            mask = batch['mask'].to(self.device)

            # Forward pass
            output = self.model(t2)

            # Compute loss
            if isinstance(self.criterion, CombinedLoss):
                loss, _ = self.criterion(output, target, mask)
            else:
                loss = self.criterion(output, target, mask)

            # Calculate metrics
            metrics = calculate_metrics(output, target, mask)

            val_losses.append(loss.item())
            val_metrics['mae'].append(metrics['mae'])
            val_metrics['rmse'].append(metrics['rmse'])
            val_metrics['correlation'].append(metrics['correlation'])

        # Average metrics
        avg_loss = np.mean(val_losses)
        avg_mae = np.mean(val_metrics['mae'])
        avg_rmse = np.mean(val_metrics['rmse'])
        avg_corr = np.mean(val_metrics['correlation'])

        return {
            'loss': avg_loss,
            'mae': avg_mae,
            'rmse': avg_rmse,
            'correlation': avg_corr
        }

    def train(self, train_loader, val_loader):
        """Main training loop"""
        num_epochs = self.config['training']['num_epochs']
        print(f"\nStarting training for {num_epochs} epochs...")

        for epoch in range(self.current_epoch, num_epochs):
            self.current_epoch = epoch

            # Train
            train_metrics = self.train_epoch(train_loader)
            self.train_losses.append(train_metrics['loss'])

            # Validate
            val_metrics = self.validate(val_loader)
            self.val_losses.append(val_metrics['loss'])

            # Log to tensorboard
            self.writer.add_scalar('train/loss_epoch', train_metrics['loss'], epoch)
            self.writer.add_scalar('train/mae', train_metrics['mae'], epoch)
            self.writer.add_scalar('train/rmse', train_metrics['rmse'], epoch)
            self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
            self.writer.add_scalar('val/mae', val_metrics['mae'], epoch)
            self.writer.add_scalar('val/rmse', val_metrics['rmse'], epoch)
            self.writer.add_scalar('val/correlation', val_metrics['correlation'], epoch)
            self.writer.add_scalar('learning_rate', self.optimizer.param_groups[0]['lr'], epoch)

            # Print epoch summary
            print(f"\nEpoch {epoch} Summary:")
            print(f"  Train Loss: {train_metrics['loss']:.4f}, MAE: {train_metrics['mae']:.4f}")
            print(f"  Val Loss: {val_metrics['loss']:.4f}, MAE: {val_metrics['mae']:.4f}, "
                  f"Correlation: {val_metrics['correlation']:.4f}")

            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['loss'])
                else:
                    self.scheduler.step()

            # Save checkpoint
            is_best = val_metrics['loss'] < self.best_val_loss
            if is_best:
                self.best_val_loss = val_metrics['loss']

            save_checkpoint(
                {
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
                    'train_loss': train_metrics['loss'],
                    'val_loss': val_metrics['loss'],
                    'best_val_loss': self.best_val_loss,
                    'config': self.config
                },
                is_best=is_best,
                checkpoint_dir=self.output_dir
            )

            # Early stopping
            if self.early_stopping(val_metrics['loss']):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break

        print("\nTraining completed!")
        print(f"Best validation loss: {self.best_val_loss:.4f}")
        self.writer.close()


def main():
    parser = argparse.ArgumentParser(description='Train fat fraction prediction model')
    parser.add_argument('--config', type=str, default='config/config_pixel_level/train_config.yaml',
                        help='Path to config file (default: config/config_pixel_level/train_config.yaml)')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    args = parser.parse_args()

    # Load config
    config_path = Path(args.config)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    print(f"Loading config from: {config_path}")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Create or load data splits
    data_dir = config['data']['data_dir']
    splits_file = Path(data_dir) / 'data_splits.json'

    if splits_file.exists():
        print("Loading existing data splits...")
        train_ids, val_ids, test_ids = load_data_splits(data_dir)

        # Validate loaded splits (filter out patients with missing files)
        print("Validating loaded splits...")
        from dataset import LiverFatDataset

        # Create temporary dataset to validate all IDs
        all_ids = train_ids + val_ids + test_ids
        temp_dataset = LiverFatDataset(
            data_dir,
            patient_ids=all_ids,
            use_patient_subdirs=config['data'].get('use_patient_subdirs', False),
            use_subdirs=config['data'].get('use_subdirs', False),
            t2_suffix=config['data'].get('t2_suffix', '_t2_aligned'),
            ff_suffix=config['data'].get('ff_suffix', '_ff_normalized'),
            mask_suffix=config['data'].get('mask_suffix', '_segmentation'),
            validate_files=True  # This will filter out invalid patients
        )

        # Get valid patient IDs after validation
        valid_ids = set(temp_dataset.patient_ids)

        # Filter each split
        train_ids = [pid for pid in train_ids if pid in valid_ids]
        val_ids = [pid for pid in val_ids if pid in valid_ids]
        test_ids = [pid for pid in test_ids if pid in valid_ids]

        print(f"After validation: Train={len(train_ids)}, Val={len(val_ids)}, Test={len(test_ids)}")
    else:
        print("Creating new data splits...")
        train_ids, val_ids, test_ids = create_data_splits(
            data_dir,
            train_ratio=config['data'].get('train_ratio', 0.7),
            val_ratio=config['data'].get('val_ratio', 0.15),
            test_ratio=config['data'].get('test_ratio', 0.15),
            random_seed=config['data'].get('random_seed', 42),
            use_subdirs=config['data'].get('use_subdirs', False),
            use_patient_subdirs=config['data'].get('use_patient_subdirs', False),
            t2_suffix=config['data'].get('t2_suffix', '_t2_aligned')
        )

    # Create inference YAML file for test set
    from dataset import create_inference_yaml
    inference_yaml_path = Path(data_dir) / 'test_inference.yaml'
    create_inference_yaml(
        data_dir=data_dir,
        test_ids=test_ids,
        output_path=str(inference_yaml_path),
        use_patient_subdirs=config['data'].get('use_patient_subdirs', False),
        t2_suffix=config['data'].get('t2_suffix', '_t2_aligned'),
        ff_suffix=config['data'].get('ff_suffix', '_ff_normalized'),
        mask_suffix=config['data'].get('mask_suffix', '_segmentation')
    )

    # Create dataloaders
    print("Creating dataloaders...")
    train_loader, val_loader, test_loader = get_dataloaders(
        data_dir=data_dir,
        batch_size=config['training']['batch_size'],
        num_workers=config['training'].get('num_workers', 4),
        use_patches=config['data'].get('use_patches', False),
        patch_size=tuple(config['data'].get('patch_size', [64, 128, 128])),
        train_ids=train_ids,
        val_ids=val_ids,
        test_ids=test_ids,
        use_subdirs=config['data'].get('use_subdirs', False),
        use_patient_subdirs=config['data'].get('use_patient_subdirs', False),
        t2_suffix=config['data'].get('t2_suffix', '_t2_aligned'),
        ff_suffix=config['data'].get('ff_suffix', '_ff_normalized'),
        mask_suffix=config['data'].get('mask_suffix', '_segmentation'),
        t2_subdir=config['data'].get('t2_subdir', 't2_images'),
        ff_subdir=config['data'].get('ff_subdir', 'fat_fraction_maps'),
        mask_subdir=config['data'].get('mask_subdir', 'liver_masks'),
        log_t2=config['data'].get('log_t2', False),
        mask_erosion=config['data'].get('mask_erosion', 0),
        augment=config['data'].get('augment', False),
        flip_prob=config['data'].get('flip_prob', 0.5),
        rotate_prob=config['data'].get('rotate_prob', 0.2),
        rotate_angle_min=config['data'].get('rotate_angle_min', 1.0),
        rotate_angle_max=config['data'].get('rotate_angle_max', 15.0),
        fat_stratified=config['data'].get('fat_patch_sampling', {}).get('enabled', False),
        fat_bin_edges=config['data'].get('fat_patch_sampling', {}).get('bin_edges'),
        fat_bin_probs=config['data'].get('fat_patch_sampling', {}).get('bin_probs'),
        fat_max_attempts=config['data'].get('fat_patch_sampling', {}).get('max_attempts', 20),
    )

    # Initialize trainer
    trainer = Trainer(config)

    # Resume from checkpoint if specified
    if args.resume:
        print(f"Resuming from checkpoint: {args.resume}")
        checkpoint = load_checkpoint(args.resume, trainer.model, trainer.optimizer, trainer.scheduler)
        trainer.current_epoch = checkpoint['epoch'] + 1
        trainer.best_val_loss = checkpoint['best_val_loss']

    # Train
    trainer.train(train_loader, val_loader)


if __name__ == "__main__":
    main()
