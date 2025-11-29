"""
Data augmentation functions for time series using geometric masking.
"""

import torch
import numpy as np
from typing import Tuple, Optional, List
import torch.nn.functional as F


class GeometricMasking:
    """Geometric masking augmentation for time series."""
    
    def __init__(
        self,
        temporal_mask_prob: float = 0.3,
        channel_mask_prob: float = 0.2,
        temporal_mask_ratio: float = 0.15,
        channel_mask_ratio: float = 0.1,
        use_time_warping: bool = False,
        time_warp_prob: float = 0.1,
        time_warp_sigma: float = 0.2
    ):
        """
        Args:
            temporal_mask_prob: Probability of applying temporal masking
            channel_mask_prob: Probability of applying channel masking
            temporal_mask_ratio: Ratio of time steps to mask
            channel_mask_ratio: Ratio of channels to mask
            use_time_warping: Whether to use time warping
            time_warp_prob: Probability of applying time warping
            time_warp_sigma: Standard deviation for time warping
        """
        self.temporal_mask_prob = temporal_mask_prob
        self.channel_mask_prob = channel_mask_prob
        self.temporal_mask_ratio = temporal_mask_ratio
        self.channel_mask_ratio = channel_mask_ratio
        self.use_time_warping = use_time_warping
        self.time_warp_prob = time_warp_prob
        self.time_warp_sigma = time_warp_sigma
    
    def temporal_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random temporal masking.
        
        Args:
            x: Input tensor of shape (B, T, D) or (T, D)
        
        Returns:
            Masked tensor
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, T, D = x.shape
        x_masked = x.clone()
        
        for b in range(B):
            if torch.rand(1).item() < self.temporal_mask_prob:
                # Randomly select time steps to mask
                num_mask = int(T * self.temporal_mask_ratio)
                mask_indices = torch.randperm(T)[:num_mask]
                # Mask with zeros
                x_masked[b, mask_indices, :] = 0.0
        
        if squeeze:
            x_masked = x_masked.squeeze(0)
        
        return x_masked
    
    def channel_masking(self, x: torch.Tensor) -> torch.Tensor:
        """Apply random channel masking.
        
        Args:
            x: Input tensor of shape (B, T, D) or (T, D)
        
        Returns:
            Masked tensor
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, T, D = x.shape
        x_masked = x.clone()
        
        for b in range(B):
            if torch.rand(1).item() < self.channel_mask_prob:
                # Randomly select channels to mask
                num_mask = int(D * self.channel_mask_ratio)
                mask_indices = torch.randperm(D)[:num_mask]
                # Mask entire channels
                x_masked[b, :, mask_indices] = 0.0
        
        if squeeze:
            x_masked = x_masked.squeeze(0)
        
        return x_masked
    
    def time_warping(self, x: torch.Tensor) -> torch.Tensor:
        """Apply time warping augmentation.
        
        Args:
            x: Input tensor of shape (B, T, D) or (T, D)
        
        Returns:
            Warped tensor
        """
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
            squeeze = True
        else:
            squeeze = False
        
        B, T, D = x.shape
        x_warped = x.clone()
        
        for b in range(B):
            if torch.rand(1).item() < self.time_warp_prob:
                # Generate warping curve
                warp_curve = torch.cumsum(
                    torch.ones(T) + torch.randn(T) * self.time_warp_sigma,
                    dim=0
                )
                warp_curve = (warp_curve - warp_curve.min()) / (warp_curve.max() - warp_curve.min() + 1e-8)
                warp_curve = warp_curve * (T - 1)
                
                # Interpolate for each channel
                for d in range(D):
                    x_warped[b, :, d] = torch.from_numpy(
                        np.interp(
                            np.arange(T),
                            warp_curve.numpy(),
                            x[b, :, d].numpy()
                        )
                    ).float()
        
        if squeeze:
            x_warped = x_warped.squeeze(0)
        
        return x_warped
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply augmentations to input.
        
        Args:
            x: Input tensor of shape (B, T, D) or (T, D)
        
        Returns:
            Augmented tensor
        """
        x_aug = x.clone()
        
        # Apply temporal masking
        x_aug = self.temporal_masking(x_aug)
        
        # Apply channel masking
        x_aug = self.channel_masking(x_aug)
        
        # Apply time warping if enabled
        if self.use_time_warping:
            x_aug = self.time_warping(x_aug)
        
        return x_aug


def create_two_views(x: torch.Tensor, augmentation: GeometricMasking) -> Tuple[torch.Tensor, torch.Tensor]:
    """Create two augmented views of the same input for contrastive learning.
    
    Args:
        x: Input tensor of shape (B, T, D)
        augmentation: Augmentation function
    
    Returns:
        Tuple of two augmented views
    """
    view1 = augmentation(x)
    view2 = augmentation(x)
    return view1, view2


class AugmentationPipeline:
    """Pipeline for applying multiple augmentations."""
    
    def __init__(self, augmentations: List[GeometricMasking]):
        """
        Args:
            augmentations: List of augmentation functions
        """
        self.augmentations = augmentations
    
    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Apply all augmentations sequentially."""
        x_aug = x.clone()
        for aug in self.augmentations:
            x_aug = aug(x_aug)
        return x_aug

