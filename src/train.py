"""
Training script for the hybrid Transformer-Contrastive-GAN framework.
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from pathlib import Path

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_dataset, create_dataloaders
from src.transformer_model import TransformerReconstructor
from src.contrastive_head import ContrastiveHead
from src.gan import Generator, Discriminator, compute_gradient_penalty
from src.augmentations import GeometricMasking, create_two_views
from src.losses import CombinedLoss
from src.utils import (
    set_seed, load_config, save_checkpoint, get_device,
    create_directories, count_parameters
)


class HybridModel(nn.Module):
    """Complete hybrid model combining Transformer, Contrastive, and GAN."""
    
    def __init__(self, config):
        super().__init__()
        
        self.config = config
        model_config = config['model']
        
        # Transformer reconstructor
        self.transformer = TransformerReconstructor(
            input_dim=model_config['input_dim'],
            d_model=model_config['d_model'],
            nhead=model_config['nhead'],
            num_encoder_layers=model_config['num_encoder_layers'],
            num_decoder_layers=model_config['num_decoder_layers'],
            dim_feedforward=model_config['dim_feedforward'],
            dropout=model_config['dropout'],
            max_len=model_config.get('max_len', 5000),
            use_decoder=model_config.get('use_decoder', True)
        )
        
        # Contrastive head
        self.contrastive_head = ContrastiveHead(
            input_dim=model_config['d_model'],
            hidden_dim=model_config.get('contrastive_hidden_dim', 256),
            projection_dim=model_config.get('projection_dim', 128),
            num_layers=model_config.get('contrastive_layers', 2),
            dropout=model_config['dropout'],
            pooling=model_config.get('pooling', 'mean')
        )
        
        # GAN components
        # Generator uses d_model as input_dim when using transformer features
        self.generator = Generator(
            input_dim=model_config['d_model'],  # Use d_model for transformer features
            window_size=config['data']['window_size'],
            latent_dim=model_config.get('latent_dim', 64),
            hidden_dims=model_config.get('generator_hidden_dims', [256, 512, 256]),
            dropout=model_config['dropout'],
            use_transformer_features=True,
            output_input_dim=model_config['input_dim']  # Output original input_dim
        )
        
        # Discriminator uses original input_dim
        self.discriminator = Discriminator(
            input_dim=model_config['input_dim'],
            window_size=config['data']['window_size'],
            hidden_dims=model_config.get('discriminator_hidden_dims', [256, 512, 256, 128]),
            dropout=model_config['dropout'],
            use_spectral_norm=model_config.get('use_spectral_norm', False)
        )
        
        self.discriminator = Discriminator(
            input_dim=model_config['input_dim'],
            window_size=config['data']['window_size'],
            hidden_dims=model_config.get('discriminator_hidden_dims', [256, 512, 256, 128]),
            dropout=model_config['dropout'],
            use_spectral_norm=model_config.get('use_spectral_norm', False)
        )
    
    def forward(self, x, return_features=False):
        """Forward pass.
        
        Args:
            x: Input windows (B, T, D)
            return_features: Whether to return intermediate features
        
        Returns:
            Dictionary with outputs
        """
        # Reconstruct
        reconstructed = self.transformer(x)
        
        # Encode for contrastive learning
        encoded = self.transformer.encode(x)
        
        # Project for contrastive learning
        projected = self.contrastive_head(encoded)
        
        outputs = {
            'reconstructed': reconstructed,
            'encoded': encoded,
            'projected': projected
        }
        
        if return_features:
            outputs['features'] = encoded
        
        return outputs


def train_epoch(
    model: HybridModel,
    discriminator: Discriminator,
    train_loader,
    augmentation: GeometricMasking,
    criterion: CombinedLoss,
    optimizer_transformer: optim.Optimizer,
    optimizer_generator: optim.Optimizer,
    optimizer_discriminator: optim.Optimizer,
    device: torch.device,
    config: dict,
    epoch: int
):
    """Train for one epoch."""
    
    model.train()
    discriminator.train()
    
    total_loss = 0.0
    total_rec_loss = 0.0
    total_con_loss = 0.0
    total_gan_loss = 0.0
    total_disc_loss = 0.0
    
    n_critic = config['training'].get('n_critic', 5)  # Discriminator updates per generator update
    use_wgan = config['training'].get('use_wgan', False)
    lambda_gp = config['training'].get('lambda_gp', 10.0)
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch}')
    
    for batch_idx, batch in enumerate(pbar):
        windows = batch['window'].to(device)  # (B, T, D)
        B, T, D = windows.shape
        
        # ========== Train Discriminator ==========
        if batch_idx % n_critic == 0:
            optimizer_discriminator.zero_grad()
            
            # Real samples
            disc_real = discriminator(windows)
            
            # Generate fake samples
            with torch.no_grad():
                encoded = model.transformer.encode(windows)
                fake_windows = model.generator(transformer_features=encoded)
            
            disc_fake = discriminator(fake_windows.detach())
            
            # Discriminator loss
            if use_wgan:
                # WGAN loss
                disc_loss = disc_fake.mean() - disc_real.mean()
                
                # Gradient penalty
                if lambda_gp > 0:
                    gp = compute_gradient_penalty(
                        discriminator, windows, fake_windows, device
                    )
                    disc_loss = disc_loss + lambda_gp * gp
            else:
                # LSGAN loss
                from src.losses import GANLoss
                gan_loss_fn = GANLoss(gan_type='lsgan')
                disc_loss = gan_loss_fn(disc_real, is_real=True) + gan_loss_fn(disc_fake, is_real=False)
            
            disc_loss.backward()
            optimizer_discriminator.step()
            total_disc_loss += disc_loss.item()
        
        # ========== Train Generator ==========
        optimizer_generator.zero_grad()
        
        # Generate fake samples
        encoded = model.transformer.encode(windows)
        fake_windows = model.generator(transformer_features=encoded)
        disc_fake_gen = discriminator(fake_windows)
        
        # Generator loss
        if use_wgan:
            gen_loss = -disc_fake_gen.mean()
        else:
            from src.losses import GANLoss
            gan_loss_fn = GANLoss(gan_type='lsgan')
            gen_loss = gan_loss_fn(disc_fake_gen, is_real=True)
        
        gen_loss.backward()
        optimizer_generator.step()
        
        # ========== Train Transformer + Contrastive ==========
        optimizer_transformer.zero_grad()
        
        # Create two augmented views
        view1, view2 = create_two_views(windows, augmentation)
        
        # Forward pass
        outputs1 = model(view1)
        outputs2 = model(view2)
        
        # Reconstruction (use original windows as target)
        reconstructed = outputs1['reconstructed']
        
        # Contrastive features
        z1 = outputs1['projected']
        z2 = outputs2['projected']
        
        # Discriminator scores for real samples
        disc_real_trans = discriminator(windows)
        disc_fake_trans = discriminator(reconstructed)
        
        # Combined loss
        losses = criterion(
            reconstructed=reconstructed,
            target=windows,
            z1=z1,
            z2=z2,
            disc_real=disc_real_trans,
            disc_fake=disc_fake_trans
        )
        
        total_loss_batch = losses['total']
        total_loss_batch.backward()
        optimizer_transformer.step()
        
        # Accumulate losses
        total_loss += total_loss_batch.item()
        total_rec_loss += losses['reconstruction'].item()
        total_con_loss += losses['contrastive'].item()
        total_gan_loss += losses['gan'].item()
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{total_loss / (batch_idx + 1):.4f}',
            'rec': f'{total_rec_loss / (batch_idx + 1):.4f}',
            'con': f'{total_con_loss / (batch_idx + 1):.4f}',
            'gan': f'{total_gan_loss / (batch_idx + 1):.4f}',
            'disc': f'{total_disc_loss / (batch_idx + 1):.4f}'
        })
    
    n_batches = len(train_loader)
    return {
        'total_loss': total_loss / n_batches,
        'reconstruction_loss': total_rec_loss / n_batches,
        'contrastive_loss': total_con_loss / n_batches,
        'gan_loss': total_gan_loss / n_batches,
        'discriminator_loss': total_disc_loss / n_batches
    }


def main():
    parser = argparse.ArgumentParser(description='Train Hybrid Transformer-Contrastive-GAN Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--resume', type=str, default=None, help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    args = parser.parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = get_device()
    print(f'Using device: {device}')
    
    # Create directories
    exp_dir = config['experiment']['experiment_dir']
    create_directories(exp_dir, ['checkpoints', 'logs', 'results'])
    
    # Load data
    print('Loading data...')
    data_config = config['data']
    train_data, _ = load_dataset(
        dataset_name=data_config['dataset_name'],
        data_dir=data_config['data_dir'],
        split='train',
        **data_config.get('dataset_kwargs', {})
    )
    
    test_data, test_labels = load_dataset(
        dataset_name=data_config['dataset_name'],
        data_dir=data_config['data_dir'],
        split='test',
        **data_config.get('dataset_kwargs', {})
    )
    
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    
    # Create dataloaders
    train_loader, test_loader = create_dataloaders(
        train_data=train_data,
        test_data=test_data,
        test_labels=test_labels,
        window_size=data_config['window_size'],
        batch_size=config['training']['batch_size'],
        train_stride=data_config.get('train_stride', 1),
        test_stride=data_config.get('test_stride', 1),
        num_workers=config['training'].get('num_workers', 0),
        pin_memory=True,
        anomaly_threshold=data_config.get('anomaly_threshold', 0.5)
    )
    
    # Create model
    print('Creating model...')
    model = HybridModel(config).to(device)
    discriminator = model.discriminator
    
    print(f'Transformer parameters: {count_parameters(model.transformer):,}')
    print(f'Generator parameters: {count_parameters(model.generator):,}')
    print(f'Discriminator parameters: {count_parameters(model.discriminator):,}')
    print(f'Total parameters: {count_parameters(model):,}')
    
    # Augmentation
    aug_config = config.get('augmentation', {})
    augmentation = GeometricMasking(
        temporal_mask_prob=aug_config.get('temporal_mask_prob', 0.3),
        channel_mask_prob=aug_config.get('channel_mask_prob', 0.2),
        temporal_mask_ratio=aug_config.get('temporal_mask_ratio', 0.15),
        channel_mask_ratio=aug_config.get('channel_mask_ratio', 0.1),
        use_time_warping=aug_config.get('use_time_warping', False),
        time_warp_prob=aug_config.get('time_warp_prob', 0.1),
        time_warp_sigma=aug_config.get('time_warp_sigma', 0.2)
    )
    
    # Loss function
    loss_config = config.get('loss', {})
    criterion = CombinedLoss(
        lambda_rec=loss_config.get('lambda_rec', 1.0),
        lambda_con=loss_config.get('lambda_con', 0.5),
        lambda_gan=loss_config.get('lambda_gan', 0.1),
        rec_loss_type=loss_config.get('rec_loss_type', 'mse'),
        gan_type=loss_config.get('gan_type', 'lsgan'),
        contrastive_temperature=loss_config.get('contrastive_temperature', 0.07)
    )
    
    # Optimizers
    training_config = config['training']
    optimizer_transformer = optim.Adam(
        list(model.transformer.parameters()) + list(model.contrastive_head.parameters()),
        lr=training_config['learning_rate'],
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    optimizer_generator = optim.Adam(
        model.generator.parameters(),
        lr=training_config.get('generator_lr', training_config['learning_rate']),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    optimizer_discriminator = optim.Adam(
        model.discriminator.parameters(),
        lr=training_config.get('discriminator_lr', training_config['learning_rate']),
        weight_decay=training_config.get('weight_decay', 1e-5)
    )
    
    # Learning rate schedulers
    scheduler_transformer = optim.lr_scheduler.StepLR(
        optimizer_transformer,
        step_size=training_config.get('lr_step_size', 50),
        gamma=training_config.get('lr_gamma', 0.5)
    )
    
    # TensorBoard writer (optional)
    try:
        writer = SummaryWriter(log_dir=os.path.join(exp_dir, 'logs'))
        use_tensorboard = True
    except Exception as e:
        print(f'Warning: TensorBoard not available ({e}). Continuing without logging.')
        writer = None
        use_tensorboard = False
    
    # Resume from checkpoint
    start_epoch = 0
    if args.resume:
        print(f'Resuming from checkpoint: {args.resume}')
        checkpoint = torch.load(args.resume, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer_transformer.load_state_dict(checkpoint['optimizer_transformer_state_dict'])
        optimizer_generator.load_state_dict(checkpoint['optimizer_generator_state_dict'])
        optimizer_discriminator.load_state_dict(checkpoint['optimizer_discriminator_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
    
    # Training loop
    num_epochs = training_config['num_epochs']
    best_loss = float('inf')
    
    print('Starting training...')
    for epoch in range(start_epoch, num_epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            discriminator=discriminator,
            train_loader=train_loader,
            augmentation=augmentation,
            criterion=criterion,
            optimizer_transformer=optimizer_transformer,
            optimizer_generator=optimizer_generator,
            optimizer_discriminator=optimizer_discriminator,
            device=device,
            config=config,
            epoch=epoch
        )
        
        # Log metrics
        if use_tensorboard:
            for key, value in train_metrics.items():
                writer.add_scalar(f'Train/{key}', value, epoch)
        
        # Update learning rate
        scheduler_transformer.step()
        
        # Save checkpoint
        if train_metrics['total_loss'] < best_loss:
            best_loss = train_metrics['total_loss']
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', 'best_model.pt')
            save_checkpoint(
                model=model,
                optimizer=optimizer_transformer,
                epoch=epoch,
                loss=train_metrics['total_loss'],
                filepath=checkpoint_path,
                additional_info={
                    'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                    'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                    'config': config
                }
            )
        
        # Periodic checkpoint
        if (epoch + 1) % training_config.get('save_interval', 10) == 0:
            checkpoint_path = os.path.join(exp_dir, 'checkpoints', f'checkpoint_epoch_{epoch+1}.pt')
            save_checkpoint(
                model=model,
                optimizer=optimizer_transformer,
                epoch=epoch,
                loss=train_metrics['total_loss'],
                filepath=checkpoint_path,
                additional_info={
                    'optimizer_generator_state_dict': optimizer_generator.state_dict(),
                    'optimizer_discriminator_state_dict': optimizer_discriminator.state_dict(),
                    'config': config
                }
            )
        
        print(f'Epoch {epoch+1}/{num_epochs} - Loss: {train_metrics["total_loss"]:.4f}')
    
    print('Training completed!')
    if use_tensorboard:
        writer.close()


if __name__ == '__main__':
    main()

