"""
GAN components: Generator and Discriminator for normal pattern synthesis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class Generator(nn.Module):
    """Generator that reconstructs or synthesizes normal windows."""
    
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        latent_dim: int = 64,
        hidden_dims: list = [256, 512, 256],
        dropout: float = 0.1,
        use_transformer_features: bool = True,
        output_input_dim: Optional[int] = None
    ):
        """
        Args:
            input_dim: Input feature dimension
            window_size: Time series window size
            latent_dim: Latent dimension for noise input
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            use_transformer_features: Whether to use transformer features as input
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        self.latent_dim = latent_dim
        self.use_transformer_features = use_transformer_features
        self.output_input_dim = output_input_dim if output_input_dim is not None else input_dim
        
        # Input dimension depends on whether we use transformer features
        if use_transformer_features:
            # Assume transformer outputs d_model dimension
            gen_input_dim = input_dim  # This will be transformer d_model
        else:
            gen_input_dim = input_dim * window_size  # Flattened input
        
        # Build generator network
        layers = []
        in_dim = latent_dim + gen_input_dim if use_transformer_features else latent_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(in_dim, self.output_input_dim * window_size))
        layers.append(nn.Tanh())  # Normalize output to [-1, 1]
        
        self.generator = nn.Sequential(*layers)
    
    def forward(
        self,
        noise: Optional[torch.Tensor] = None,
        transformer_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            noise: Random noise tensor (B, latent_dim) or None
            transformer_features: Transformer encoded features (B, T, D) or (B, D)
        
        Returns:
            Generated window (B, window_size, input_dim)
        """
        batch_size = None
        
        if self.use_transformer_features and transformer_features is not None:
            # Use transformer features
            if len(transformer_features.shape) == 3:
                # (B, T, D) -> (B, T*D) or pool to (B, D)
                B, T, D = transformer_features.shape
                batch_size = B
                # Pool features (mean pooling)
                features = transformer_features.mean(dim=1)  # (B, D)
            else:
                features = transformer_features  # (B, D)
                batch_size = features.size(0)
            
            # Generate noise if not provided
            if noise is None:
                noise = torch.randn(batch_size, self.latent_dim, device=features.device)
            
            # Concatenate noise and features
            gen_input = torch.cat([noise, features], dim=1)  # (B, latent_dim + D)
        else:
            # Use only noise
            if noise is None:
                batch_size = 32  # Default batch size
                noise = torch.randn(batch_size, self.latent_dim, device=next(self.parameters()).device)
            else:
                batch_size = noise.size(0)
            gen_input = noise
        
        # Generate
        output = self.generator(gen_input)  # (B, output_input_dim * window_size)
        
        # Reshape to window format
        generated = output.view(batch_size, self.window_size, self.output_input_dim)
        
        return generated


class Discriminator(nn.Module):
    """Discriminator that distinguishes real vs generated normal patterns."""
    
    def __init__(
        self,
        input_dim: int,
        window_size: int,
        hidden_dims: list = [256, 512, 256, 128],
        dropout: float = 0.1,
        use_spectral_norm: bool = False
    ):
        """
        Args:
            input_dim: Input feature dimension
            window_size: Time series window size
            hidden_dims: Hidden layer dimensions
            dropout: Dropout probability
            use_spectral_norm: Whether to use spectral normalization for stability
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.window_size = window_size
        
        # Build discriminator network
        layers = []
        in_dim = input_dim * window_size  # Flattened input
        
        for hidden_dim in hidden_dims:
            if use_spectral_norm:
                layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, hidden_dim)))
            else:
                layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Output layer (single value: real/fake probability)
        if use_spectral_norm:
            layers.append(nn.utils.spectral_norm(nn.Linear(in_dim, 1)))
        else:
            layers.append(nn.Linear(in_dim, 1))
        
        self.discriminator = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input window (B, window_size, input_dim) or (B, T, D)
        
        Returns:
            Discriminator output (B, 1)
        """
        B, T, D = x.shape
        
        # Flatten window
        x_flat = x.view(B, -1)  # (B, T*D)
        
        # Discriminate
        output = self.discriminator(x_flat)  # (B, 1)
        
        return output


class WGANGenerator(Generator):
    """WGAN Generator with gradient penalty support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove Tanh for WGAN
        # Replace last layer activation
        if isinstance(self.generator[-1], nn.Tanh):
            self.generator = nn.Sequential(*list(self.generator[:-1]))


class WGANDiscriminator(Discriminator):
    """WGAN Discriminator (Critic) with gradient penalty support."""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Remove sigmoid for WGAN (output is unbounded)


def compute_gradient_penalty(
    discriminator: nn.Module,
    real_samples: torch.Tensor,
    fake_samples: torch.Tensor,
    device: torch.device
) -> torch.Tensor:
    """Compute gradient penalty for WGAN-GP.
    
    Args:
        discriminator: Discriminator model
        real_samples: Real samples (B, T, D)
        fake_samples: Fake samples (B, T, D)
        device: Device
    
    Returns:
        Gradient penalty value
    """
    batch_size = real_samples.size(0)
    
    # Random interpolation
    alpha = torch.rand(batch_size, 1, 1, device=device)
    interpolated = alpha * real_samples + (1 - alpha) * fake_samples
    interpolated.requires_grad_(True)
    
    # Discriminator output
    d_interpolated = discriminator(interpolated)
    
    # Compute gradients
    gradients = torch.autograd.grad(
        outputs=d_interpolated,
        inputs=interpolated,
        grad_outputs=torch.ones_like(d_interpolated),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # Gradient penalty
    gradient_penalty = ((gradients.norm(2, dim=(1, 2)) - 1) ** 2).mean()
    
    return gradient_penalty

