"""
Transformer encoder-decoder model for time series reconstruction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer."""
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: Model dimension
            max_len: Maximum sequence length
            dropout: Dropout probability
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
        
        Returns:
            Positionally encoded tensor
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class TransformerEncoder(nn.Module):
    """Transformer encoder for feature extraction."""
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of encoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, d_model)
            mask: Optional attention mask
        
        Returns:
            Encoded features of shape (B, T, d_model)
        """
        return self.encoder(x, src_key_padding_mask=mask)


class TransformerDecoder(nn.Module):
    """Transformer decoder for sequence reconstruction."""
    
    def __init__(
        self,
        d_model: int = 128,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        activation: str = 'gelu'
    ):
        """
        Args:
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            activation: Activation function
        """
        super().__init__()
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation=activation,
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
    
    def forward(
        self,
        tgt: torch.Tensor,
        memory: torch.Tensor,
        tgt_mask: Optional[torch.Tensor] = None,
        memory_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            tgt: Target sequence (B, T, d_model)
            memory: Encoder output (B, T, d_model)
            tgt_mask: Target mask
            memory_mask: Memory mask
        
        Returns:
            Decoded sequence (B, T, d_model)
        """
        return self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_key_padding_mask=memory_mask)


class TransformerReconstructor(nn.Module):
    """Complete transformer encoder-decoder for time series reconstruction."""
    
    def __init__(
        self,
        input_dim: int,
        d_model: int = 128,
        nhead: int = 8,
        num_encoder_layers: int = 4,
        num_decoder_layers: int = 4,
        dim_feedforward: int = 512,
        dropout: float = 0.1,
        max_len: int = 5000,
        use_decoder: bool = True
    ):
        """
        Args:
            input_dim: Input feature dimension
            d_model: Model dimension
            nhead: Number of attention heads
            num_encoder_layers: Number of encoder layers
            num_decoder_layers: Number of decoder layers
            dim_feedforward: Feedforward dimension
            dropout: Dropout probability
            max_len: Maximum sequence length
            use_decoder: Whether to use decoder or MLP head
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.d_model = d_model
        self.use_decoder = use_decoder
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_len, dropout)
        
        # Encoder
        self.encoder = TransformerEncoder(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_encoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout
        )
        
        # Decoder or MLP head
        if use_decoder:
            self.decoder = TransformerDecoder(
                d_model=d_model,
                nhead=nhead,
                num_layers=num_decoder_layers,
                dim_feedforward=dim_feedforward,
                dropout=dropout
            )
            # Output projection
            self.output_projection = nn.Linear(d_model, input_dim)
        else:
            # MLP head for reconstruction
            self.mlp_head = nn.Sequential(
                nn.Linear(d_model, dim_feedforward),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim_feedforward, input_dim)
            )
    
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask
        
        Returns:
            Reconstructed tensor of shape (B, T, D)
        """
        B, T, D = x.shape
        
        # Project input to model dimension
        x_proj = self.input_projection(x)  # (B, T, d_model)
        
        # Add positional encoding
        x_pos = self.pos_encoder(x_proj)
        
        # Encode
        encoded = self.encoder(x_pos, mask=mask)  # (B, T, d_model)
        
        # Decode or use MLP
        if self.use_decoder:
            # Use decoder with encoded features as memory
            decoded = self.decoder(encoded, encoded, memory_mask=mask)  # (B, T, d_model)
            # Project to output dimension
            reconstructed = self.output_projection(decoded)  # (B, T, D)
        else:
            # Use MLP head
            reconstructed = self.mlp_head(encoded)  # (B, T, D)
        
        return reconstructed
    
    def encode(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Extract encoded features.
        
        Args:
            x: Input tensor of shape (B, T, D)
            mask: Optional attention mask
        
        Returns:
            Encoded features of shape (B, T, d_model)
        """
        B, T, D = x.shape
        x_proj = self.input_projection(x)
        x_pos = self.pos_encoder(x_proj)
        encoded = self.encoder(x_pos, mask=mask)
        return encoded

