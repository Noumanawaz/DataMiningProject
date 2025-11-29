"""
Data preprocessing script for time series datasets.
"""

import os
import argparse
import numpy as np
from pathlib import Path
from src.utils import normalize_data, sliding_window


def preprocess_dataset(
    data_dir: str,
    dataset_name: str,
    output_dir: str,
    window_size: int = 100,
    stride: int = 1,
    normalize: bool = True,
    **kwargs
):
    """Preprocess a dataset.
    
    Args:
        data_dir: Directory containing raw data
        dataset_name: Name of dataset
        output_dir: Directory to save preprocessed data
        window_size: Window size for sliding windows
        stride: Stride for sliding windows
        normalize: Whether to normalize data
        **kwargs: Additional dataset-specific arguments
    """
    from src.datasets import load_dataset
    
    print(f'Preprocessing {dataset_name} dataset...')
    
    # Load data
    train_data, _ = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='train',
        **kwargs
    )
    
    test_data, test_labels = load_dataset(
        dataset_name=dataset_name,
        data_dir=data_dir,
        split='test',
        **kwargs
    )
    
    print(f'Train data shape: {train_data.shape}')
    print(f'Test data shape: {test_data.shape}')
    
    # Normalize
    if normalize:
        train_data_norm, norm_params = normalize_data(train_data, method='standard')
        
        # Apply same normalization to test data
        test_data_norm = (test_data - norm_params['mean']) / norm_params['std']
        
        # Save normalization parameters
        norm_path = os.path.join(output_dir, f'{dataset_name}_norm_params.npz')
        np.savez(norm_path, **norm_params)
        print(f'Saved normalization parameters to: {norm_path}')
    else:
        train_data_norm = train_data
        test_data_norm = test_data
        norm_params = None
    
    # Create sliding windows
    print(f'Creating sliding windows (window_size={window_size}, stride={stride})...')
    
    train_windows = sliding_window(train_data_norm, window_size, stride)
    test_windows = sliding_window(test_data_norm, window_size, stride)
    
    # Create window labels for test set
    test_window_labels = []
    for i in range(0, len(test_labels) - window_size + 1, stride):
        window_label = 1 if np.any(test_labels[i:i+window_size] == 1) else 0
        test_window_labels.append(window_label)
    test_window_labels = np.array(test_window_labels)
    
    print(f'Train windows shape: {train_windows.shape}')
    print(f'Test windows shape: {test_windows.shape}')
    print(f'Test window labels shape: {test_window_labels.shape}')
    print(f'Anomaly ratio in test windows: {np.mean(test_window_labels):.4f}')
    
    # Save preprocessed data
    os.makedirs(output_dir, exist_ok=True)
    
    # Save train data
    train_path = os.path.join(output_dir, f'{dataset_name}_train.npz')
    np.savez(train_path, data=train_data_norm, windows=train_windows)
    print(f'Saved train data to: {train_path}')
    
    # Save test data
    test_path = os.path.join(output_dir, f'{dataset_name}_test.npz')
    np.savez(test_path, 
             data=test_data_norm, 
             windows=test_windows,
             labels=test_labels,
             window_labels=test_window_labels)
    print(f'Saved test data to: {test_path}')
    
    # Save statistics
    stats = {
        'train_shape': train_data.shape,
        'test_shape': test_data.shape,
        'num_features': train_data.shape[1],
        'window_size': window_size,
        'stride': stride,
        'train_windows': len(train_windows),
        'test_windows': len(test_windows),
        'anomaly_ratio': float(np.mean(test_window_labels))
    }
    
    stats_path = os.path.join(output_dir, f'{dataset_name}_stats.npz')
    np.savez(stats_path, **stats)
    print(f'Saved statistics to: {stats_path}')


def main():
    parser = argparse.ArgumentParser(description='Preprocess Time Series Datasets')
    parser.add_argument('--data_dir', type=str, required=True, help='Directory containing raw data')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save preprocessed data')
    parser.add_argument('--dataset', type=str, required=True,
                       choices=['smap', 'msl', 'smd', 'ebay'],
                       help='Dataset to preprocess')
    parser.add_argument('--window_size', type=int, default=100, help='Window size')
    parser.add_argument('--stride', type=int, default=1, help='Stride for sliding windows')
    parser.add_argument('--normalize', action='store_true', default=True, help='Normalize data')
    parser.add_argument('--machine', type=str, default='machine-1-1', help='Machine name for SMD')
    
    args = parser.parse_args()
    
    kwargs = {}
    if args.dataset == 'smd':
        kwargs['machine'] = args.machine
    
    preprocess_dataset(
        data_dir=args.data_dir,
        dataset_name=args.dataset,
        output_dir=args.output_dir,
        window_size=args.window_size,
        stride=args.stride,
        normalize=args.normalize,
        **kwargs
    )
    
    print('Preprocessing completed!')


if __name__ == '__main__':
    main()

