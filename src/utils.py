"""
Utility functions for the anomaly detection framework.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict, Any
import yaml
from pathlib import Path


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_config(config_path: str) -> Dict[str, Any]:
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def save_config(config: Dict[str, Any], save_path: str) -> None:
    """Save configuration to YAML file."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False)


def create_directories(base_path: str, subdirs: list) -> None:
    """Create directory structure."""
    for subdir in subdirs:
        os.makedirs(os.path.join(base_path, subdir), exist_ok=True)


def get_device() -> torch.device:
    """Get available device (CUDA or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    filepath: str,
    additional_info: Optional[Dict] = None
) -> None:
    """Save model checkpoint."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    if additional_info:
        checkpoint.update(additional_info)
    torch.save(checkpoint, filepath)


def load_checkpoint(
    filepath: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: Optional[torch.device] = None
) -> Dict[str, Any]:
    """Load model checkpoint."""
    if device is None:
        device = get_device()
    checkpoint = torch.load(filepath, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint


def normalize_data(data: np.ndarray, method: str = 'standard') -> Tuple[np.ndarray, Dict]:
    """Normalize data using specified method.
    
    Args:
        data: Input data array
        method: Normalization method ('standard', 'minmax', 'robust')
    
    Returns:
        Normalized data and normalization parameters
    """
    if method == 'standard':
        mean = np.mean(data, axis=0, keepdims=True)
        std = np.std(data, axis=0, keepdims=True)
        std = np.where(std == 0, 1, std)  # Avoid division by zero
        normalized = (data - mean) / std
        params = {'mean': mean, 'std': std}
    
    elif method == 'minmax':
        min_val = np.min(data, axis=0, keepdims=True)
        max_val = np.max(data, axis=0, keepdims=True)
        range_val = max_val - min_val
        range_val = np.where(range_val == 0, 1, range_val)
        normalized = (data - min_val) / range_val
        params = {'min': min_val, 'max': max_val}
    
    elif method == 'robust':
        median = np.median(data, axis=0, keepdims=True)
        q75 = np.percentile(data, 75, axis=0, keepdims=True)
        q25 = np.percentile(data, 25, axis=0, keepdims=True)
        iqr = q75 - q25
        iqr = np.where(iqr == 0, 1, iqr)
        normalized = (data - median) / iqr
        params = {'median': median, 'iqr': iqr}
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized, params


def denormalize_data(data: np.ndarray, params: Dict) -> np.ndarray:
    """Denormalize data using stored parameters."""
    if 'mean' in params and 'std' in params:
        return data * params['std'] + params['mean']
    elif 'min' in params and 'max' in params:
        return data * (params['max'] - params['min']) + params['min']
    elif 'median' in params and 'iqr' in params:
        return data * params['iqr'] + params['median']
    else:
        raise ValueError("Invalid normalization parameters")


def sliding_window(data: np.ndarray, window_size: int, stride: int = 1) -> np.ndarray:
    """Create sliding windows from time series data.
    
    Args:
        data: Input time series data (T, D)
        window_size: Size of each window
        stride: Stride for sliding window
    
    Returns:
        Windowed data (N, window_size, D)
    """
    T, D = data.shape
    windows = []
    for i in range(0, T - window_size + 1, stride):
        windows.append(data[i:i + window_size])
    return np.array(windows)


def compute_anomaly_score(
    recon_error: np.ndarray,
    disc_score: np.ndarray,
    contrastive_score: np.ndarray,
    weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)
) -> np.ndarray:
    """Combine multiple anomaly scores.
    
    Args:
        recon_error: Reconstruction error scores
        disc_score: Discriminator scores
        contrastive_score: Contrastive outlierness scores
        weights: Weights for (reconstruction, discriminator, contrastive)
    
    Returns:
        Combined anomaly scores
    """
    # Normalize each score to [0, 1]
    def normalize_score(score):
        score_min = np.min(score)
        score_max = np.max(score)
        if score_max - score_min > 0:
            return (score - score_min) / (score_max - score_min)
        return score
    
    recon_norm = normalize_score(recon_error)
    disc_norm = normalize_score(disc_score)
    contrastive_norm = normalize_score(contrastive_score)
    
    combined = (weights[0] * recon_norm + 
                weights[1] * disc_norm + 
                weights[2] * contrastive_norm)
    
    return combined

