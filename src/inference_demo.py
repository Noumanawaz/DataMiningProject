"""
Inference demo script for real-time anomaly detection.
"""

import os
import sys
import argparse
import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import TimeSeriesDataset
from src.transformer_model import TransformerReconstructor
from src.contrastive_head import ContrastiveHead
from src.gan import Generator, Discriminator
from src.augmentations import GeometricMasking
from src.utils import load_config, get_device, load_checkpoint, sliding_window
from src.train import HybridModel


def detect_anomalies(
    model: HybridModel,
    data: np.ndarray,
    window_size: int,
    device: torch.device,
    augmentation: GeometricMasking,
    batch_size: int = 32,
    score_weights: tuple = (0.5, 0.3, 0.2)
) -> dict:
    """Detect anomalies in time series data.
    
    Args:
        model: Trained model
        data: Time series data (T, D)
        window_size: Window size
        device: Device
        augmentation: Augmentation function
        batch_size: Batch size for inference
        score_weights: Weights for combining scores
    
    Returns:
        Dictionary with anomaly scores and predictions
    """
    model.eval()
    
    # Create windows
    windows = sliding_window(data, window_size, stride=1)
    n_windows = len(windows)
    
    all_recon_errors = []
    all_disc_scores = []
    all_contrastive_scores = []
    
    with torch.no_grad():
        for i in range(0, n_windows, batch_size):
            batch_windows = windows[i:i+batch_size]
            batch_windows = torch.from_numpy(batch_windows).float().to(device)
            
            B, T, D = batch_windows.shape
            
            # Reconstruction error
            reconstructed = model.transformer(batch_windows)
            recon_error = torch.mean((reconstructed - batch_windows) ** 2, dim=(1, 2)).cpu().numpy()
            
            # Discriminator score
            disc_real = model.discriminator(batch_windows)
            disc_scores = disc_real.squeeze().cpu().numpy()
            disc_anomaly_scores = -disc_scores
            
            # Contrastive outlierness
            view1, view2 = augmentation(batch_windows), augmentation(batch_windows)
            encoded1 = model.transformer.encode(view1)
            encoded2 = model.transformer.encode(view2)
            z1 = model.contrastive_head(encoded1)
            z2 = model.contrastive_head(encoded2)
            similarity = torch.sum(z1 * z2, dim=1).cpu().numpy()
            contrastive_scores = -similarity
            
            all_recon_errors.extend(recon_error)
            all_disc_scores.extend(disc_anomaly_scores)
            all_contrastive_scores.extend(contrastive_scores)
    
    # Combine scores
    from src.utils import compute_anomaly_score
    combined_scores = compute_anomaly_score(
        np.array(all_recon_errors),
        np.array(all_disc_scores),
        np.array(all_contrastive_scores),
        weights=score_weights
    )
    
    # Convert window-level scores to point-level
    point_scores = np.zeros(len(data))
    for i, score in enumerate(combined_scores):
        point_scores[i:i+window_size] = np.maximum(point_scores[i:i+window_size], score)
    
    return {
        'reconstruction_errors': np.array(all_recon_errors),
        'discriminator_scores': np.array(all_disc_scores),
        'contrastive_scores': np.array(all_contrastive_scores),
        'combined_scores': combined_scores,
        'point_scores': point_scores
    }


def plot_inference_results(data: np.ndarray, scores: dict, threshold: float = None, save_path: str = None):
    """Plot inference results.
    
    Args:
        data: Original time series data (T, D)
        scores: Dictionary with anomaly scores
        threshold: Anomaly threshold (if None, use median + 2*std)
        save_path: Path to save plot
    """
    T, D = data.shape
    
    # Determine threshold
    if threshold is None:
        threshold = np.median(scores['point_scores']) + 2 * np.std(scores['point_scores'])
    
    # Anomaly predictions
    predictions = (scores['point_scores'] > threshold).astype(int)
    
    # Plot
    fig, axes = plt.subplots(min(D, 5) + 1, 1, figsize=(14, 3 * (min(D, 5) + 1)))
    if D == 1:
        axes = [axes]
    
    # Plot each channel
    for d in range(min(D, 5)):
        axes[d].plot(data[:, d], label=f'Channel {d+1}', alpha=0.7)
        # Highlight anomalies
        anomaly_indices = np.where(predictions == 1)[0]
        if len(anomaly_indices) > 0:
            axes[d].scatter(
                anomaly_indices,
                data[anomaly_indices, d],
                color='red',
                s=20,
                alpha=0.6,
                label='Anomaly'
            )
        axes[d].set_ylabel(f'Value (Channel {d+1})')
        axes[d].legend()
        axes[d].grid(True, alpha=0.3)
    
    # Plot anomaly scores
    axes[-1].plot(scores['point_scores'], label='Anomaly Score', color='orange')
    axes[-1].axhline(y=threshold, color='r', linestyle='--', label=f'Threshold ({threshold:.4f})')
    axes[-1].fill_between(
        range(len(predictions)),
        0,
        scores['point_scores'],
        where=predictions == 1,
        alpha=0.3,
        color='red',
        label='Predicted Anomalies'
    )
    axes[-1].set_xlabel('Time Step')
    axes[-1].set_ylabel('Anomaly Score')
    axes[-1].legend()
    axes[-1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300)
        print(f'Plot saved to: {save_path}')
    else:
        plt.show()
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Anomaly Detection Inference Demo')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--data', type=str, required=True, help='Path to input data file (.npy)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    parser.add_argument('--threshold', type=float, default=None, help='Anomaly threshold')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = get_device()
    print(f'Using device: {device}')
    
    # Output directory
    if args.output is None:
        args.output = os.path.join(config['experiment']['experiment_dir'], 'inference_results')
    os.makedirs(args.output, exist_ok=True)
    
    # Load data
    print(f'Loading data from: {args.data}')
    data = np.load(args.data)
    print(f'Data shape: {data.shape}')
    
    # Normalize data (use training statistics if available)
    # For demo, we'll normalize on the fly
    from src.utils import normalize_data
    data_norm, _ = normalize_data(data, method='standard')
    
    # Create model
    print('Loading model...')
    model = HybridModel(config).to(device)
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f'Loaded checkpoint from epoch {checkpoint["epoch"]}')
    
    # Augmentation
    aug_config = config.get('augmentation', {})
    augmentation = GeometricMasking(
        temporal_mask_prob=aug_config.get('temporal_mask_prob', 0.3),
        channel_mask_prob=aug_config.get('channel_mask_prob', 0.2),
        temporal_mask_ratio=aug_config.get('temporal_mask_ratio', 0.15),
        channel_mask_ratio=aug_config.get('channel_mask_ratio', 0.1),
        use_time_warping=aug_config.get('use_time_warping', False)
    )
    
    # Detect anomalies
    print('Detecting anomalies...')
    window_size = config['data']['window_size']
    score_weights = config.get('evaluation', {}).get('score_weights', (0.5, 0.3, 0.2))
    
    scores = detect_anomalies(
        model=model,
        data=data_norm,
        window_size=window_size,
        device=device,
        augmentation=augmentation,
        batch_size=config['training']['batch_size'],
        score_weights=score_weights
    )
    
    # Plot results
    print('Generating visualization...')
    plot_inference_results(
        data=data,
        scores=scores,
        threshold=args.threshold,
        save_path=os.path.join(args.output, 'anomaly_detection.png')
    )
    
    # Save results
    np.savez(
        os.path.join(args.output, 'detection_results.npz'),
        **scores,
        threshold=args.threshold if args.threshold else np.median(scores['point_scores']) + 2 * np.std(scores['point_scores'])
    )
    
    # Print summary
    threshold = args.threshold if args.threshold else np.median(scores['point_scores']) + 2 * np.std(scores['point_scores'])
    n_anomalies = np.sum(scores['point_scores'] > threshold)
    anomaly_ratio = n_anomalies / len(scores['point_scores']) * 100
    
    print(f'\n=== Detection Summary ===')
    print(f'Total time steps: {len(scores["point_scores"])}')
    print(f'Anomaly threshold: {threshold:.4f}')
    print(f'Detected anomalies: {n_anomalies} ({anomaly_ratio:.2f}%)')
    print(f'Max anomaly score: {np.max(scores["point_scores"]):.4f}')
    print(f'Mean anomaly score: {np.mean(scores["point_scores"]):.4f}')
    print(f'\nResults saved to: {args.output}')


if __name__ == '__main__':
    main()

