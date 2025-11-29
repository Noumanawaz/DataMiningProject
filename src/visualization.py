"""
Visualization script for analyzing anomaly detection results.
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve
import argparse
import json

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

sns.set_style('whitegrid')
plt.rcParams['figure.figsize'] = (12, 6)


def load_results(results_path: str) -> dict:
    """Load evaluation results from npz file.
    
    Args:
        results_path: Path to results npz file
    
    Returns:
        Dictionary with scores and labels
    """
    if not os.path.exists(results_path):
        raise FileNotFoundError(f'Results file not found: {results_path}')
    
    results = np.load(results_path)
    
    return {
        'reconstruction_errors': results['reconstruction_errors'],
        'discriminator_scores': results['discriminator_scores'],
        'contrastive_scores': results['contrastive_scores'],
        'combined_scores': results['combined_scores'],
        'labels': results['labels']
    }


def plot_score_distributions(scores: dict, labels: np.ndarray, save_path: str):
    """Plot distributions of scores for normal vs anomalous samples.
    
    Args:
        scores: Dictionary with score arrays
        labels: Ground truth labels
        save_path: Path to save plot
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    normal_mask = labels == 0
    anomaly_mask = labels == 1
    
    # Reconstruction errors
    axes[0, 0].hist(scores['reconstruction_errors'][normal_mask], bins=50, alpha=0.5, 
                   label='Normal', density=True, color='blue')
    axes[0, 0].hist(scores['reconstruction_errors'][anomaly_mask], bins=50, alpha=0.5, 
                   label='Anomaly', density=True, color='red')
    axes[0, 0].set_xlabel('Reconstruction Error')
    axes[0, 0].set_ylabel('Density')
    axes[0, 0].set_title('Reconstruction Error Distribution')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Discriminator scores
    axes[0, 1].hist(scores['discriminator_scores'][normal_mask], bins=50, alpha=0.5, 
                   label='Normal', density=True, color='blue')
    axes[0, 1].hist(scores['discriminator_scores'][anomaly_mask], bins=50, alpha=0.5, 
                   label='Anomaly', density=True, color='red')
    axes[0, 1].set_xlabel('Discriminator Score')
    axes[0, 1].set_ylabel('Density')
    axes[0, 1].set_title('Discriminator Score Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Contrastive scores
    axes[1, 0].hist(scores['contrastive_scores'][normal_mask], bins=50, alpha=0.5, 
                   label='Normal', density=True, color='blue')
    axes[1, 0].hist(scores['contrastive_scores'][anomaly_mask], bins=50, alpha=0.5, 
                   label='Anomaly', density=True, color='red')
    axes[1, 0].set_xlabel('Contrastive Score')
    axes[1, 0].set_ylabel('Density')
    axes[1, 0].set_title('Contrastive Score Distribution')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Combined scores
    axes[1, 1].hist(scores['combined_scores'][normal_mask], bins=50, alpha=0.5, 
                   label='Normal', density=True, color='blue')
    axes[1, 1].hist(scores['combined_scores'][anomaly_mask], bins=50, alpha=0.5, 
                   label='Anomaly', density=True, color='red')
    axes[1, 1].set_xlabel('Combined Anomaly Score')
    axes[1, 1].set_ylabel('Density')
    axes[1, 1].set_title('Combined Score Distribution')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Saved score distributions to: {save_path}')


def compute_metrics(scores: dict, labels: np.ndarray) -> dict:
    """Compute performance metrics for each score type.
    
    Args:
        scores: Dictionary with score arrays
        labels: Ground truth labels
    
    Returns:
        Dictionary with metrics
    """
    score_types = {
        'Reconstruction Error': scores['reconstruction_errors'],
        'Discriminator Score': scores['discriminator_scores'],
        'Contrastive Score': scores['contrastive_scores'],
        'Combined Score': scores['combined_scores']
    }
    
    metrics = {}
    for name, score_values in score_types.items():
        auc = roc_auc_score(labels, score_values)
        ap = average_precision_score(labels, score_values)
        metrics[name] = {'AUC-ROC': float(auc), 'AUC-PR': float(ap)}
        print(f'{name}:')
        print(f'  AUC-ROC: {auc:.4f}')
        print(f'  AUC-PR:  {ap:.4f}')
    
    return metrics


def plot_curves(scores: dict, labels: np.ndarray, save_dir: str):
    """Plot ROC and PR curves.
    
    Args:
        scores: Dictionary with score arrays
        labels: Ground truth labels
        save_dir: Directory to save plots
    """
    score_types = {
        'Reconstruction Error': scores['reconstruction_errors'],
        'Discriminator Score': scores['discriminator_scores'],
        'Contrastive Score': scores['contrastive_scores'],
        'Combined Score': scores['combined_scores']
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # ROC curves
    for name, score_values in score_types.items():
        fpr, tpr, _ = roc_curve(labels, score_values)
        auc = roc_auc_score(labels, score_values)
        axes[0].plot(fpr, tpr, label=f'{name} (AUC={auc:.4f})', linewidth=2)
    
    axes[0].plot([0, 1], [0, 1], 'k--', label='Random', linewidth=1)
    axes[0].set_xlabel('False Positive Rate', fontsize=12)
    axes[0].set_ylabel('True Positive Rate', fontsize=12)
    axes[0].set_title('ROC Curves', fontsize=14)
    axes[0].legend(fontsize=10)
    axes[0].grid(True, alpha=0.3)
    
    # PR curves
    for name, score_values in score_types.items():
        precision, recall, _ = precision_recall_curve(labels, score_values)
        ap = average_precision_score(labels, score_values)
        axes[1].plot(recall, precision, label=f'{name} (AP={ap:.4f})', linewidth=2)
    
    axes[1].set_xlabel('Recall', fontsize=12)
    axes[1].set_ylabel('Precision', fontsize=12)
    axes[1].set_title('Precision-Recall Curves', fontsize=14)
    axes[1].legend(fontsize=10)
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, 'curves_comparison.png'), dpi=300)
    plt.close()
    print(f'Saved curves to: {os.path.join(save_dir, "curves_comparison.png")}')


def plot_anomaly_timeline(scores: dict, labels: np.ndarray, window_size: int, save_path: str):
    """Plot anomaly scores over time.
    
    Args:
        scores: Dictionary with score arrays
        labels: Ground truth labels
        window_size: Window size used for sliding windows
        save_path: Path to save plot
    """
    combined_scores = scores['combined_scores']
    
    # Convert window-level scores to point-level
    point_scores = np.zeros(len(labels) + window_size - 1)
    point_labels = np.zeros(len(labels) + window_size - 1)
    
    for i, (score, label) in enumerate(zip(combined_scores, labels)):
        point_scores[i:i+window_size] = np.maximum(point_scores[i:i+window_size], score)
        point_labels[i:i+window_size] = np.maximum(point_labels[i:i+window_size], label)
    
    # Downsample for visualization
    step = max(1, len(point_scores) // 10000)
    point_scores_ds = point_scores[::step]
    point_labels_ds = point_labels[::step]
    
    plt.figure(figsize=(16, 6))
    plt.plot(point_scores_ds, label='Anomaly Score', alpha=0.7, linewidth=1)
    plt.fill_between(
        range(len(point_labels_ds)),
        0,
        point_scores_ds,
        where=point_labels_ds == 1,
        alpha=0.3,
        color='red',
        label='Anomaly Ground Truth'
    )
    plt.xlabel('Time Step', fontsize=12)
    plt.ylabel('Anomaly Score', fontsize=12)
    plt.title('Anomaly Scores Over Time', fontsize=14)
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Saved anomaly timeline to: {save_path}')


def plot_score_correlation(scores: dict, save_path: str):
    """Plot correlation matrix of different scores.
    
    Args:
        scores: Dictionary with score arrays
        save_path: Path to save plot
    """
    # Compute correlation matrix
    score_matrix = np.column_stack([
        scores['reconstruction_errors'],
        scores['discriminator_scores'],
        scores['contrastive_scores'],
        scores['combined_scores']
    ])
    
    correlation_matrix = np.corrcoef(score_matrix.T)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        fmt='.3f',
        cmap='coolwarm',
        center=0,
        square=True,
        xticklabels=['Reconstruction', 'Discriminator', 'Contrastive', 'Combined'],
        yticklabels=['Reconstruction', 'Discriminator', 'Contrastive', 'Combined']
    )
    plt.title('Score Correlation Matrix', fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f'Saved correlation matrix to: {save_path}')


def main():
    parser = argparse.ArgumentParser(description='Visualize Anomaly Detection Results')
    parser.add_argument('--results', type=str, required=True, 
                       help='Path to results npz file (anomaly_scores.npz)')
    parser.add_argument('--output_dir', type=str, default=None,
                       help='Output directory for plots (default: same as results file directory)')
    parser.add_argument('--window_size', type=int, default=100,
                       help='Window size used for sliding windows')
    args = parser.parse_args()
    
    # Load results
    print(f'Loading results from: {args.results}')
    scores = load_results(args.results)
    labels = scores['labels']
    
    print(f'Loaded {len(labels)} samples')
    print(f'Anomaly ratio: {np.mean(labels):.4f}')
    
    # Output directory
    if args.output_dir is None:
        args.output_dir = os.path.dirname(os.path.abspath(args.results))
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Compute and print metrics
    print('\n=== Performance Metrics ===')
    metrics = compute_metrics(scores, labels)
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'visualization_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f'\nSaved metrics to: {metrics_path}')
    
    # Generate plots
    print('\n=== Generating Visualizations ===')
    
    # Score distributions
    plot_score_distributions(
        scores, labels,
        os.path.join(args.output_dir, 'score_distributions_detailed.png')
    )
    
    # ROC and PR curves
    plot_curves(scores, labels, args.output_dir)
    
    # Anomaly timeline
    plot_anomaly_timeline(
        scores, labels, args.window_size,
        os.path.join(args.output_dir, 'anomaly_timeline_detailed.png')
    )
    
    # Score correlation
    plot_score_correlation(
        scores,
        os.path.join(args.output_dir, 'score_correlation.png')
    )
    
    print(f'\nAll visualizations saved to: {args.output_dir}')


if __name__ == '__main__':
    main()

