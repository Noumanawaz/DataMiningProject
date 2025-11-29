"""
Dataset loading and preprocessing for multivariate time series anomaly detection.
"""

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Optional, Dict, List
import pickle
from pathlib import Path


class TimeSeriesDataset(Dataset):
    """Dataset class for multivariate time series windows."""
    
    def __init__(
        self,
        data: np.ndarray,
        labels: Optional[np.ndarray] = None,
        window_size: int = 100,
        stride: int = 1,
        normalize: bool = True,
        norm_params: Optional[Dict] = None,
        anomaly_threshold: float = 0.5
    ):
        """
        Args:
            data: Time series data of shape (T, D) where T is time steps, D is features
            labels: Anomaly labels of shape (T,) or None
            window_size: Size of sliding window
            stride: Stride for sliding window
            normalize: Whether to normalize the data
            norm_params: Pre-computed normalization parameters
            anomaly_threshold: Fraction of points in window that must be anomalous (0.0-1.0)
                              Default 0.5 means >50% of points must be anomalous
        """
        self.window_size = window_size
        self.stride = stride
        self.anomaly_threshold = anomaly_threshold
        
        # Normalize data
        if normalize:
            if norm_params is None:
                from src.utils import normalize_data
                self.data, self.norm_params = normalize_data(data, method='standard')
            else:
                from src.utils import normalize_data
                self.data, _ = normalize_data(data, method='standard')
                self.norm_params = norm_params
        else:
            self.data = data
            self.norm_params = None
        
        # Create sliding windows
        self.windows = self._create_windows(self.data)
        
        # Create window labels (window is anomalous if threshold fraction of points are anomalous)
        if labels is not None:
            self.labels = self._create_window_labels(labels)
        else:
            self.labels = None
    
    def _create_windows(self, data: np.ndarray) -> np.ndarray:
        """Create sliding windows from time series."""
        T, D = data.shape
        windows = []
        for i in range(0, T - self.window_size + 1, self.stride):
            windows.append(data[i:i + self.window_size])
        return np.array(windows, dtype=np.float32)
    
    def _create_window_labels(self, labels: np.ndarray) -> np.ndarray:
        """Create labels for windows.
        
        Window is labeled as anomalous if the fraction of anomalous points 
        in the window exceeds the threshold.
        """
        window_labels = []
        for i in range(0, len(labels) - self.window_size + 1, self.stride):
            window_slice = labels[i:i + self.window_size]
            anomaly_ratio = np.mean(window_slice == 1)
            # Window is anomalous if anomaly ratio exceeds threshold
            window_label = 1 if anomaly_ratio > self.anomaly_threshold else 0
            window_labels.append(window_label)
        return np.array(window_labels, dtype=np.int64)
    
    def __len__(self) -> int:
        return len(self.windows)
    
    def __getitem__(self, idx: int) -> Dict:
        """Get a window and its label."""
        window = torch.from_numpy(self.windows[idx])
        sample = {'window': window}
        if self.labels is not None:
            sample['label'] = torch.tensor(self.labels[idx], dtype=torch.long)
        return sample


def load_smap_data(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load SMAP dataset.
    
    Args:
        data_dir: Directory containing SMAP data
        split: 'train' or 'test'
    
    Returns:
        Tuple of (data, labels) where labels are None for train split
    """
    if split == 'train':
        data_path = os.path.join(data_dir, 'train', 'data.npy')
        data = np.load(data_path)
        return data, None
    else:
        data_path = os.path.join(data_dir, 'test', 'data.npy')
        label_path = os.path.join(data_dir, 'test', 'label.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return data, labels


def load_msl_data(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load MSL dataset."""
    if split == 'train':
        data_path = os.path.join(data_dir, 'train', 'data.npy')
        data = np.load(data_path)
        return data, None
    else:
        data_path = os.path.join(data_dir, 'test', 'data.npy')
        label_path = os.path.join(data_dir, 'test', 'label.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return data, labels


def load_smd_data(data_dir: str, machine: str = 'machine-1-1', split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load SMD (Server Machine Dataset).
    
    Args:
        data_dir: Directory containing SMD data
        machine: Machine name (e.g., 'machine-1-1')
        split: 'train' or 'test'
    """
    if split == 'train':
        data_path = os.path.join(data_dir, 'train', f'{machine}.npy')
        data = np.load(data_path)
        return data, None
    else:
        data_path = os.path.join(data_dir, 'test', f'{machine}.npy')
        label_path = os.path.join(data_dir, 'test_label', f'{machine}.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return data, labels


def load_ebay_data(data_dir: str, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load eBay anomalies dataset."""
    if split == 'train':
        data_path = os.path.join(data_dir, 'train', 'data.npy')
        data = np.load(data_path)
        return data, None
    else:
        data_path = os.path.join(data_dir, 'test', 'data.npy')
        label_path = os.path.join(data_dir, 'test', 'label.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return data, labels


def load_ntu_data(data_dir: str, dataset_name: str, split: str = 'train') -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Load NTU dataset."""
    if split == 'train':
        data_path = os.path.join(data_dir, 'train', f'{dataset_name}.npy')
        data = np.load(data_path)
        return data, None
    else:
        data_path = os.path.join(data_dir, 'test', f'{dataset_name}.npy')
        label_path = os.path.join(data_dir, 'test', f'{dataset_name}_label.npy')
        data = np.load(data_path)
        labels = np.load(label_path)
        return data, labels


def load_dataset(
    dataset_name: str,
    data_dir: str,
    split: str = 'train',
    **kwargs
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Generic dataset loader.
    
    Args:
        dataset_name: Name of dataset ('smap', 'msl', 'smd', 'ebay', 'ntu')
        data_dir: Directory containing dataset
        split: 'train' or 'test'
        **kwargs: Additional arguments for specific datasets
    
    Returns:
        Tuple of (data, labels)
    """
    dataset_name = dataset_name.lower()
    
    if dataset_name == 'smap':
        return load_smap_data(data_dir, split)
    elif dataset_name == 'msl':
        return load_msl_data(data_dir, split)
    elif dataset_name == 'smd':
        machine = kwargs.get('machine', 'machine-1-1')
        return load_smd_data(data_dir, machine, split)
    elif dataset_name == 'ebay':
        return load_ebay_data(data_dir, split)
    elif dataset_name == 'ntu':
        ntu_name = kwargs.get('dataset_name', 'dataset1')
        return load_ntu_data(data_dir, ntu_name, split)
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")


def create_dataloaders(
    train_data: np.ndarray,
    test_data: np.ndarray,
    test_labels: np.ndarray,
    window_size: int = 100,
    batch_size: int = 32,
    train_stride: int = 1,
    test_stride: int = 1,
    num_workers: int = 0,
    pin_memory: bool = True,
    anomaly_threshold: float = 0.5
) -> Tuple[DataLoader, DataLoader]:
    """Create train and test dataloaders.
    
    Args:
        train_data: Training data (T, D)
        test_data: Test data (T, D)
        test_labels: Test labels (T,)
        window_size: Window size
        batch_size: Batch size
        train_stride: Stride for training windows
        test_stride: Stride for test windows
        num_workers: Number of workers for DataLoader
        pin_memory: Whether to pin memory
        anomaly_threshold: Fraction of points in window that must be anomalous (0.0-1.0)
    
    Returns:
        Tuple of (train_loader, test_loader)
    """
    # Create datasets
    train_dataset = TimeSeriesDataset(
        data=train_data,
        labels=None,
        window_size=window_size,
        stride=train_stride,
        normalize=True,
        anomaly_threshold=anomaly_threshold
    )
    
    # Use train normalization params for test
    test_dataset = TimeSeriesDataset(
        data=test_data,
        labels=test_labels,
        window_size=window_size,
        stride=test_stride,
        normalize=True,
        norm_params=train_dataset.norm_params,
        anomaly_threshold=anomaly_threshold
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    
    return train_loader, test_loader

