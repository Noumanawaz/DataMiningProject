"""
Contrastive learning head with projector network.
"""

import torch
import torch.nn as nn
from typing import Tuple


class Projector(nn.Module):
    """Projector network for contrastive learning."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden layer dimension
            output_dim: Output projection dimension
            num_layers: Number of projection layers
            dropout: Dropout probability
        """
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for i in range(num_layers - 1):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            in_dim = hidden_dim
        
        # Final projection layer
        layers.append(nn.Linear(in_dim, output_dim))
        
        self.projector = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, T, D) or (B, D)
        
        Returns:
            Projected features
        """
        if len(x.shape) == 3:
            # (B, T, D) -> (B*T, D)
            B, T, D = x.shape
            x = x.reshape(B * T, D)
            projected = self.projector(x)
            # (B*T, output_dim) -> (B, T, output_dim)
            return projected.reshape(B, T, -1)
        else:
            # (B, D) -> (B, output_dim)
            return self.projector(x)


class ContrastiveHead(nn.Module):
    """Contrastive learning head with projector."""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        projection_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
        pooling: str = 'mean'
    ):
        """
        Args:
            input_dim: Input feature dimension (from encoder)
            hidden_dim: Hidden dimension in projector
            projection_dim: Final projection dimension
            num_layers: Number of projection layers
            dropout: Dropout probability
            pooling: Pooling method ('mean', 'max', 'last', 'cls')
        """
        super().__init__()
        
        self.pooling = pooling
        self.projection_dim = projection_dim
        
        # Pooling dimension adjustment
        if pooling == 'cls':
            # Assume first token is CLS token
            pool_dim = input_dim
        elif pooling in ['mean', 'max', 'last']:
            pool_dim = input_dim
        else:
            raise ValueError(f"Unknown pooling method: {pooling}")
        
        # Projector network
        self.projector = Projector(
            input_dim=pool_dim,
            hidden_dim=hidden_dim,
            output_dim=projection_dim,
            num_layers=num_layers,
            dropout=dropout
        )
    
    def pool_features(self, x: torch.Tensor) -> torch.Tensor:
        """Pool sequence features.
        
        Args:
            x: Input features of shape (B, T, D)
        
        Returns:
            Pooled features of shape (B, D)
        """
        if self.pooling == 'mean':
            return x.mean(dim=1)  # (B, D)
        elif self.pooling == 'max':
            return x.max(dim=1)[0]  # (B, D)
        elif self.pooling == 'last':
            return x[:, -1, :]  # (B, D)
        elif self.pooling == 'cls':
            return x[:, 0, :]  # (B, D)
        else:
            raise ValueError(f"Unknown pooling method: {self.pooling}")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input features of shape (B, T, D)
        
        Returns:
            Projected features of shape (B, projection_dim)
        """
        # Pool sequence to single vector
        pooled = self.pool_features(x)  # (B, D)
        
        # Project
        projected = self.projector(pooled)  # (B, projection_dim)
        
        # L2 normalize
        projected = nn.functional.normalize(projected, p=2, dim=1)
        
        return projected

