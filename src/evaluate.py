"""
Evaluation script for anomaly detection.
"""

import os
import sys
import argparse
import torch
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.datasets import load_dataset, create_dataloaders
from src.transformer_model import TransformerReconstructor
from src.contrastive_head import ContrastiveHead
from src.gan import Generator, Discriminator
from src.augmentations import GeometricMasking, create_two_views
from src.utils import load_config, get_device, load_checkpoint
from src.train import HybridModel


def compute_anomaly_scores(
    model: HybridModel,
    test_loader,
    device: torch.device,
    augmentation: GeometricMasking
) -> dict:
    """Compute anomaly scores for test data.
    
    Returns:
        Dictionary with scores and labels
    """
    model.eval()
    
    all_recon_errors = []
    all_disc_scores = []
    all_contrastive_scores = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc='Computing scores'):
            windows = batch['window'].to(device)
            labels = batch['label'].cpu().numpy()
            
            B, T, D = windows.shape
            
            # Reconstruction error
            reconstructed = model.transformer(windows)
            recon_error = torch.mean((reconstructed - windows) ** 2, dim=(1, 2)).cpu().numpy()
            
            # Discriminator score (lower = more anomalous)
            disc_real = model.discriminator(windows)
            disc_scores = disc_real.squeeze().cpu().numpy()
            # Convert to anomaly score (invert)
            disc_anomaly_scores = -disc_scores
            
            # Contrastive outlierness
            # Create two views
            view1, view2 = create_two_views(windows, augmentation)
            
            # Get projections
            encoded1 = model.transformer.encode(view1)
            encoded2 = model.transformer.encode(view2)
            
            z1 = model.contrastive_head(encoded1)
            z2 = model.contrastive_head(encoded2)
            
            # Compute similarity (lower = more anomalous)
            similarity = torch.sum(z1 * z2, dim=1).cpu().numpy()
            contrastive_scores = -similarity  # Invert for anomaly score
            
            all_recon_errors.extend(recon_error)
            all_disc_scores.extend(disc_anomaly_scores)
            all_contrastive_scores.extend(contrastive_scores)
            all_labels.extend(labels)
    
    return {
        'reconstruction_errors': np.array(all_recon_errors),
        'discriminator_scores': np.array(all_disc_scores),
        'contrastive_scores': np.array(all_contrastive_scores),
        'labels': np.array(all_labels)
    }


def combine_scores(scores: dict, weights: tuple = (0.5, 0.3, 0.2)) -> np.ndarray:
    """Combine multiple anomaly scores.
    
    Args:
        scores: Dictionary with score arrays
        weights: Weights for (reconstruction, discriminator, contrastive)
    
    Returns:
        Combined scores
    """
    from src.utils import compute_anomaly_score
    
    return compute_anomaly_score(
        scores['reconstruction_errors'],
        scores['discriminator_scores'],
        scores['contrastive_scores'],
        weights=weights
    )


def plot_roc_curve(y_true, y_scores, save_path: str):
    """Plot ROC curve."""
    try:
        if len(np.unique(y_true)) < 2:
            # Cannot compute ROC with only one class
            plt.figure(figsize=(8, 6))
            plt.text(0.5, 0.5, 'Cannot compute ROC curve:\nOnly one class in labels', 
                    ha='center', va='center', fontsize=12)
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('ROC Curve (Not Available)')
            plt.xlim(0, 1)
            plt.ylim(0, 1)
            plt.savefig(save_path)
            plt.close()
            return np.nan
        
        fpr, tpr, _ = roc_curve(y_true, y_scores)
        auc = roc_auc_score(y_true, y_scores)
        
        plt.figure(figsize=(8, 6))
        plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.4f})')
        plt.plot([0, 1], [0, 1], 'k--', label='Random')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve')
        plt.legend()
        plt.grid(True)
        plt.savefig(save_path)
        plt.close()
        
        return auc
    except ValueError as e:
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.5, f'Error computing ROC curve:\n{str(e)}', 
                ha='center', va='center', fontsize=12)
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC Curve (Error)')
        plt.savefig(save_path)
        plt.close()
        return np.nan


def plot_pr_curve(y_true, y_scores, save_path: str):
    """Plot Precision-Recall curve."""
    precision, recall, _ = precision_recall_curve(y_true, y_scores)
    ap = average_precision_score(y_true, y_scores)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label=f'PR Curve (AP = {ap:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve')
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
    
    return ap


def plot_anomaly_timeline(scores: np.ndarray, labels: np.ndarray, save_path: str, window_size: int = 100):
    """Plot anomaly scores over time."""
    # Convert window-level scores to point-level
    point_scores = np.zeros(len(scores) + window_size - 1)
    point_labels = np.zeros(len(scores) + window_size - 1)
    
    for i, (score, label) in enumerate(zip(scores, labels)):
        point_scores[i:i+window_size] = np.maximum(point_scores[i:i+window_size], score)
        point_labels[i:i+window_size] = np.maximum(point_labels[i:i+window_size], label)
    
    # Downsample for visualization
    step = max(1, len(point_scores) // 10000)
    point_scores = point_scores[::step]
    point_labels = point_labels[::step]
    
    plt.figure(figsize=(14, 6))
    plt.plot(point_scores, label='Anomaly Score', alpha=0.7)
    plt.fill_between(
        range(len(point_labels)),
        0,
        point_scores,
        where=point_labels == 1,
        alpha=0.3,
        color='red',
        label='Anomaly Ground Truth'
    )
    plt.xlabel('Time Step')
    plt.ylabel('Anomaly Score')
    plt.title('Anomaly Scores Over Time')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_score_distributions(scores: dict, labels: np.ndarray, save_path: str):
    """Plot distributions of scores for normal vs anomalous samples."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # Reconstruction errors
    axes[0].hist(scores['reconstruction_errors'][normal_mask], bins=50, alpha=0.5, label='Normal', density=True)
    axes[0].hist(scores['reconstruction_errors'][anomaly_mask], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[0].set_xlabel('Reconstruction Error')
    axes[0].set_ylabel('Density')
    axes[0].set_title('Reconstruction Error Distribution')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Discriminator scores
    axes[1].hist(scores['discriminator_scores'][normal_mask], bins=50, alpha=0.5, label='Normal', density=True)
    axes[1].hist(scores['discriminator_scores'][anomaly_mask], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[1].set_xlabel('Discriminator Score')
    axes[1].set_ylabel('Density')
    axes[1].set_title('Discriminator Score Distribution')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Contrastive scores
    axes[2].hist(scores['contrastive_scores'][normal_mask], bins=50, alpha=0.5, label='Normal', density=True)
    axes[2].hist(scores['contrastive_scores'][anomaly_mask], bins=50, alpha=0.5, label='Anomaly', density=True)
    axes[2].set_xlabel('Contrastive Score')
    axes[2].set_ylabel('Density')
    axes[2].set_title('Contrastive Score Distribution')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()


def main():
    parser = argparse.ArgumentParser(description='Evaluate Anomaly Detection Model')
    parser.add_argument('--config', type=str, required=True, help='Path to config file')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    parser.add_argument('--output_dir', type=str, default=None, help='Output directory for results')
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Device
    device = get_device()
    print(f'Using device: {device}')
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.join(config['experiment']['experiment_dir'], 'results')
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    print('Loading test data...')
    data_config = config['data']
    test_data, test_labels = load_dataset(
        dataset_name=data_config['dataset_name'],
        data_dir=data_config['data_dir'],
        split='test',
        **data_config.get('dataset_kwargs', {})
    )
    
    # Create test dataloader (need train data for normalization)
    train_data, _ = load_dataset(
        dataset_name=data_config['dataset_name'],
        data_dir=data_config['data_dir'],
        split='train',
        **data_config.get('dataset_kwargs', {})
    )
    
    from src.datasets import TimeSeriesDataset
    from torch.utils.data import DataLoader
    
    anomaly_threshold = data_config.get('anomaly_threshold', 0.5)
    
    train_dataset = TimeSeriesDataset(
        data=train_data,
        labels=None,
        window_size=data_config['window_size'],
        stride=1,
        normalize=True,
        anomaly_threshold=anomaly_threshold
    )
    
    test_dataset = TimeSeriesDataset(
        data=test_data,
        labels=test_labels,
        window_size=data_config['window_size'],
        stride=1,
        normalize=True,
        norm_params=train_dataset.norm_params,
        anomaly_threshold=anomaly_threshold
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config['training']['batch_size'],
        shuffle=False,
        num_workers=0
    )
    
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
    
    # Compute anomaly scores
    print('Computing anomaly scores...')
    scores = compute_anomaly_scores(model, test_loader, device, augmentation)
    
    # Combine scores
    score_weights = config.get('evaluation', {}).get('score_weights', (0.5, 0.3, 0.2))
    combined_scores = combine_scores(scores, weights=score_weights)
    
    # Evaluate individual scores
    print('\n=== Individual Score Performance ===')
    
    # Check label distribution
    unique_labels = np.unique(scores['labels'])
    label_counts = {int(label): int(np.sum(scores['labels'] == label)) for label in unique_labels}
    print(f'Label distribution: {label_counts}')
    
    if len(unique_labels) < 2:
        print(f'\n⚠️  Warning: Only one class present in labels (all {unique_labels[0]}).')
        print('   AUC-ROC cannot be computed. This may indicate:')
        print('   - All test samples are normal (label=0)')
        print('   - All test samples are anomalous (label=1)')
        print('   - Dataset preprocessing issue')
        print('   - Using dummy/test data without proper labels\n')
    
    metrics = {}
    
    for score_name, score_values in [
        ('Reconstruction Error', scores['reconstruction_errors']),
        ('Discriminator Score', scores['discriminator_scores']),
        ('Contrastive Score', scores['contrastive_scores']),
        ('Combined Score', combined_scores)
    ]:
        try:
            if len(unique_labels) < 2:
                auc = np.nan
                print(f'{score_name}:')
                print(f'  AUC-ROC: NaN (only one class in labels)')
            else:
                auc = roc_auc_score(scores['labels'], score_values)
                print(f'{score_name}:')
                print(f'  AUC-ROC: {auc:.4f}')
            
            ap = average_precision_score(scores['labels'], score_values)
            print(f'  AUC-PR:  {ap:.4f}')
            metrics[score_name] = {'AUC': auc, 'AP': ap}
        except ValueError as e:
            print(f'{score_name}: Error computing metrics - {e}')
            metrics[score_name] = {'AUC': np.nan, 'AP': np.nan}
    
    # Plot results
    print('\nGenerating plots...')
    
    # ROC curves
    plot_roc_curve(
        scores['labels'],
        combined_scores,
        os.path.join(args.output_dir, 'roc_curve.png')
    )
    
    # PR curves
    plot_pr_curve(
        scores['labels'],
        combined_scores,
        os.path.join(args.output_dir, 'pr_curve.png')
    )
    
    # Anomaly timeline
    plot_anomaly_timeline(
        combined_scores,
        scores['labels'],
        os.path.join(args.output_dir, 'anomaly_timeline.png'),
        window_size=data_config['window_size']
    )
    
    # Score distributions
    plot_score_distributions(
        scores,
        scores['labels'],
        os.path.join(args.output_dir, 'score_distributions.png')
    )
    
    # Save scores
    np.savez(
        os.path.join(args.output_dir, 'anomaly_scores.npz'),
        reconstruction_errors=scores['reconstruction_errors'],
        discriminator_scores=scores['discriminator_scores'],
        contrastive_scores=scores['contrastive_scores'],
        combined_scores=combined_scores,
        labels=scores['labels']
    )
    
    # Save metrics
    import json
    with open(os.path.join(args.output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    print(f'\nResults saved to: {args.output_dir}')
    print('Evaluation completed!')


if __name__ == '__main__':
    main()

