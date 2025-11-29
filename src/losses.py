"""
Loss functions for the hybrid framework.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ReconstructionLoss(nn.Module):
    """Reconstruction loss (MSE or MAE)."""
    
    def __init__(self, loss_type: str = 'mse', reduction: str = 'mean'):
        """
        Args:
            loss_type: 'mse' or 'mae'
            reduction: 'mean', 'sum', or 'none'
        """
        super().__init__()
        self.loss_type = loss_type
        self.reduction = reduction
        
        if loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        elif loss_type == 'mae':
            self.criterion = nn.L1Loss(reduction=reduction)
        else:
            raise ValueError(f"Unknown loss type: {loss_type}")
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            reconstructed: Reconstructed tensor (B, T, D)
            target: Target tensor (B, T, D)
            mask: Optional mask to ignore certain positions (B, T)
        
        Returns:
            Loss value
        """
        if mask is not None:
            # Apply mask
            mask_expanded = mask.unsqueeze(-1).expand_as(reconstructed)  # (B, T, D)
            reconstructed = reconstructed * (1 - mask_expanded.float())
            target = target * (1 - mask_expanded.float())
        
        loss = self.criterion(reconstructed, target)
        return loss


class ContrastiveLoss(nn.Module):
    """NT-Xent (InfoNCE) contrastive loss."""
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature parameter for softmax
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            z1: Projected features from view 1 (B, D)
            z2: Projected features from view 2 (B, D)
        
        Returns:
            Contrastive loss
        """
        batch_size = z1.size(0)
        
        # Concatenate features
        z = torch.cat([z1, z2], dim=0)  # (2B, D)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z, z.T) / self.temperature  # (2B, 2B)
        
        # Create labels: positive pairs are (i, i+B) and (i+B, i)
        labels = torch.arange(batch_size, device=z.device)
        labels = torch.cat([labels + batch_size, labels], dim=0)  # (2B,)
        
        # Mask out self-similarity
        mask = torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        similarity_matrix = similarity_matrix.masked_fill(mask, float('-inf'))
        
        # Compute loss
        loss = F.cross_entropy(similarity_matrix, labels)
        
        return loss


class GANLoss(nn.Module):
    """GAN loss (LSGAN or standard)."""
    
    def __init__(self, gan_type: str = 'lsgan', real_label: float = 1.0, fake_label: float = 0.0):
        """
        Args:
            gan_type: 'lsgan' or 'standard'
            real_label: Label for real samples
            fake_label: Label for fake samples
        """
        super().__init__()
        self.gan_type = gan_type
        self.real_label = real_label
        self.fake_label = fake_label
        
        if gan_type == 'lsgan':
            self.criterion = nn.MSELoss()
        elif gan_type == 'standard':
            self.criterion = nn.BCEWithLogitsLoss()
        else:
            raise ValueError(f"Unknown GAN type: {gan_type}")
    
    def forward(
        self,
        prediction: torch.Tensor,
        is_real: bool
    ) -> torch.Tensor:
        """
        Args:
            prediction: Discriminator output
            is_real: Whether the input is real
        
        Returns:
            GAN loss
        """
        if is_real:
            target = torch.ones_like(prediction) * self.real_label
        else:
            target = torch.zeros_like(prediction) * self.fake_label
        
        loss = self.criterion(prediction, target)
        return loss


class CombinedLoss(nn.Module):
    """Combined loss function."""
    
    def __init__(
        self,
        lambda_rec: float = 1.0,
        lambda_con: float = 0.5,
        lambda_gan: float = 0.1,
        rec_loss_type: str = 'mse',
        gan_type: str = 'lsgan',
        contrastive_temperature: float = 0.07
    ):
        """
        Args:
            lambda_rec: Weight for reconstruction loss
            lambda_con: Weight for contrastive loss
            lambda_gan: Weight for GAN loss
            rec_loss_type: Type of reconstruction loss ('mse' or 'mae')
            gan_type: Type of GAN loss ('lsgan' or 'standard')
            contrastive_temperature: Temperature for contrastive loss
        """
        super().__init__()
        
        self.lambda_rec = lambda_rec
        self.lambda_con = lambda_con
        self.lambda_gan = lambda_gan
        
        self.reconstruction_loss = ReconstructionLoss(loss_type=rec_loss_type)
        self.contrastive_loss = ContrastiveLoss(temperature=contrastive_temperature)
        self.gan_loss = GANLoss(gan_type=gan_type)
    
    def forward(
        self,
        reconstructed: torch.Tensor,
        target: torch.Tensor,
        z1: Optional[torch.Tensor] = None,
        z2: Optional[torch.Tensor] = None,
        disc_real: Optional[torch.Tensor] = None,
        disc_fake: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> dict:
        """
        Args:
            reconstructed: Reconstructed tensor (B, T, D)
            target: Target tensor (B, T, D)
            z1: Projected features from view 1 (B, D)
            z2: Projected features from view 2 (B, D)
            disc_real: Discriminator output for real samples
            disc_fake: Discriminator output for fake samples
            mask: Optional mask
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        
        # Reconstruction loss
        rec_loss = self.reconstruction_loss(reconstructed, target, mask)
        losses['reconstruction'] = rec_loss
        
        # Contrastive loss
        con_loss = torch.tensor(0.0, device=reconstructed.device)
        if z1 is not None and z2 is not None:
            con_loss = self.contrastive_loss(z1, z2)
        losses['contrastive'] = con_loss
        
        # GAN loss
        gan_loss = torch.tensor(0.0, device=reconstructed.device)
        if disc_real is not None and disc_fake is not None:
            gan_loss_real = self.gan_loss(disc_real, is_real=True)
            gan_loss_fake = self.gan_loss(disc_fake, is_real=False)
            gan_loss = gan_loss_real + gan_loss_fake
        losses['gan'] = gan_loss
        
        # Total loss
        total_loss = (
            self.lambda_rec * rec_loss +
            self.lambda_con * con_loss +
            self.lambda_gan * gan_loss
        )
        losses['total'] = total_loss
        
        return losses

