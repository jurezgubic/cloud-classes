"""
Height-based cloud classification using 1D Gaussian Mixture Model.

Classifies clouds based solely on their lifetime maximum cloud-top height.
Each cloud gets one scalar value: max(max_height[track, :]) over its lifetime.

The number of classes is determined automatically using BIC (Bayesian Information
Criterion) - we fit GMMs with k=1,2,...,k_max components and select the k that
minimizes BIC.
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
from scipy.stats import norm
import seaborn as sns
from sklearn.mixture import GaussianMixture


def get_lifetime_max_heights(
    ds: xr.Dataset,
    min_timesteps: int = 3,
    only_valid: bool = True
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract lifetime maximum cloud-top height for each cloud.
    
    For each cloud track, we take the maximum of max_height over all timesteps,
    giving one scalar per cloud representing how high it reached during its life.
    
    Args:
        ds: Dataset from cloud_results.nc with 'max_height', 'size', 'valid_track'
        min_timesteps: Minimum active timesteps required
        only_valid: If True, use only valid (complete lifetime) tracks
        
    Returns:
        heights: Array of lifetime max heights [m], shape (n_clouds,)
        track_ids: Array of corresponding track indices
    """
    # Filter: valid tracks only
    if only_valid:
        valid_mask = ds['valid_track'].values == 1
    else:
        valid_mask = np.ones(ds.sizes['track'], dtype=bool)
    
    # Filter: minimum active timesteps
    size = ds['size'].values  # [track, time]
    active_counts = np.sum(np.isfinite(size) & (size > 0), axis=1)
    timestep_mask = active_counts >= min_timesteps
    
    # Combined mask
    combined_mask = valid_mask & timestep_mask
    candidate_indices = np.nonzero(combined_mask)[0]
    
    # Extract max_height and compute lifetime maximum
    max_height = ds['max_height'].values  # [track, time]
    
    heights = []
    track_ids = []
    
    for idx in candidate_indices:
        h_lifetime = np.nanmax(max_height[idx, :])
        if np.isfinite(h_lifetime):
            heights.append(h_lifetime)
            track_ids.append(idx)
    
    return np.array(heights), np.array(track_ids)


def fit_gmm_1d_with_bic(
    heights: np.ndarray,
    k_max: int = 6,
    k_fixed: int = None,
    random_seed: int = 0
) -> dict:
    """
    Fit 1D GMM, either with fixed k or automatic selection via BIC.
    
    If k_fixed is provided, use that number of components directly.
    Otherwise, fit GMMs with k=1,2,...,k_max and select k that minimizes BIC.
    
    Args:
        heights: Array of height values, shape (n_clouds,)
        k_max: Maximum number of components to try (ignored if k_fixed is set)
        k_fixed: If set, use this exact number of clusters (skip BIC selection)
        random_seed: Random seed for reproducibility
        
    Returns:
        Dictionary with:
            - labels: Cluster assignments, shape (n_clouds,)
            - n_clusters: Selected number of clusters
            - gmm: Fitted GaussianMixture object
            - bic_scores: Dict {k: BIC} for all k values (empty if k_fixed)
            - means: Cluster means (sorted by height)
            - stds: Cluster standard deviations
    """
    X = heights.reshape(-1, 1)  # sklearn expects 2D
    
    bic_scores = {}
    
    if k_fixed is not None:
        # Use fixed k directly
        optimal_k = k_fixed
        best_gmm = GaussianMixture(
            n_components=k_fixed,
            covariance_type='full',
            random_state=random_seed,
            n_init=5
        )
        best_gmm.fit(X)
    else:
        # Fit GMMs for k = 1 to k_max and select by BIC
        gmms = {}
        for k in range(1, k_max + 1):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type='full',
                random_state=random_seed,
                n_init=5
            )
            gmm.fit(X)
            bic_scores[k] = gmm.bic(X)
            gmms[k] = gmm
        
        optimal_k = min(bic_scores, key=bic_scores.get)
        best_gmm = gmms[optimal_k]
    
    # Get labels and sort classes by mean height (ascending)
    labels_raw = best_gmm.predict(X)
    means_raw = best_gmm.means_.flatten()
    
    # Reorder classes so class 0 = lowest height, class k-1 = highest
    sort_order = np.argsort(means_raw)
    label_map = {old: new for new, old in enumerate(sort_order)}
    labels = np.array([label_map[l] for l in labels_raw])
    
    # Sorted means and stds
    means = means_raw[sort_order]
    stds = np.sqrt(best_gmm.covariances_.flatten()[sort_order])
    
    return {
        'labels': labels,
        'n_clusters': optimal_k,
        'gmm': best_gmm,
        'bic_scores': bic_scores,
        'means': means,
        'stds': stds,
    }


def compute_class_height_stats(
    heights: np.ndarray,
    labels: np.ndarray
) -> dict:
    """
    Compute height statistics per class.
    
    Args:
        heights: Array of height values
        labels: Cluster assignments
        
    Returns:
        Dictionary with per-class statistics
    """
    n_classes = len(np.unique(labels))
    
    stats = {
        'n_classes': n_classes,
        'counts': [],
        'means': [],
        'stds': [],
        'medians': [],
        'p10': [],
        'p90': [],
        'min': [],
        'max': [],
    }
    
    for k in range(n_classes):
        h_k = heights[labels == k]
        stats['counts'].append(len(h_k))
        stats['means'].append(np.mean(h_k))
        stats['stds'].append(np.std(h_k))
        stats['medians'].append(np.median(h_k))
        stats['p10'].append(np.percentile(h_k, 10))
        stats['p90'].append(np.percentile(h_k, 90))
        stats['min'].append(np.min(h_k))
        stats['max'].append(np.max(h_k))
    
    return stats


# =============================================================================
# Plotting
# =============================================================================

def plot_height_histogram_with_gmm(heights: np.ndarray, labels: np.ndarray,
                                    height_results: dict, outdir: Path) -> None:
    """
    Plot histogram of lifetime max heights with GMM fit overlay and BIC curve.
    
    Args:
        heights: Array of height values [m]
        labels: Cluster assignments
        height_results: Dict from fit_gmm_1d_with_bic
        outdir: Output directory
    """
    n_classes = height_results['n_clusters']
    means = height_results['means']
    stds = height_results['stds']
    bic_scores = height_results['bic_scores']
    colors = sns.color_palette('tab10', n_classes)
    
    # Use 2-panel layout only if BIC scores exist (auto k selection)
    has_bic = len(bic_scores) > 0
    if has_bic:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes[0]
    else:
        fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    bins = np.arange(
        np.floor(heights.min() / 50) * 50,
        np.ceil(heights.max() / 50) * 50 + 50,
        50
    )
    
    # Plot histogram for each class
    for k in range(n_classes):
        h_k = heights[labels == k]
        ax.hist(h_k, bins=bins, alpha=0.5, color=colors[k], 
                label=f'Class {k} (n={len(h_k)})', edgecolor='black', linewidth=0.5)
    
    # Overlay GMM Gaussian curves
    x_range = np.linspace(heights.min() - 100, heights.max() + 100, 500)
    total_pdf = np.zeros_like(x_range)
    
    for k in range(n_classes):
        n_k = np.sum(labels == k)
        weight = n_k / len(heights)
        pdf_k = weight * norm.pdf(x_range, means[k], stds[k])
        total_pdf += pdf_k
        
        # Scale to match histogram
        scale = len(heights) * (bins[1] - bins[0])
        ax.plot(x_range, pdf_k * scale, color=colors[k], linewidth=2, linestyle='--')
    
    ax.plot(x_range, total_pdf * scale, 'k-', linewidth=2, label='GMM fit')
    
    ax.set_xlabel('Lifetime maximum cloud-top height [m]', fontsize=12)
    ax.set_ylabel('Number of clouds', fontsize=12)
    ax.set_title(f'Height Distribution with GMM Fit (k={n_classes})', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Right panel: BIC curve (only if auto k selection was used)
    if has_bic:
        ax = axes[1]
        ks = sorted(bic_scores.keys())
        bics = [bic_scores[k] for k in ks]
        
        ax.plot(ks, bics, 'o-', linewidth=2, markersize=10, color='steelblue')
        ax.axvline(n_classes, color='red', linestyle='--', linewidth=2, 
                   label=f'Selected k={n_classes}')
        ax.scatter([n_classes], [bic_scores[n_classes]], s=200, c='red', 
                   zorder=5, edgecolors='black', linewidth=2)
        
        ax.set_xlabel('Number of components (k)', fontsize=12)
        ax.set_ylabel('BIC (lower is better)', fontsize=12)
        ax.set_title('Model Selection via BIC', fontsize=13, fontweight='bold')
        ax.set_xticks(ks)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    plt.savefig(outdir / '01_height_histogram_gmm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 01_height_histogram_gmm.png")


def plot_height_boxplots(heights: np.ndarray, labels: np.ndarray,
                         height_results: dict, outdir: Path) -> None:
    """
    Plot box plots showing height distribution per class.
    
    Args:
        heights: Array of height values [m]
        labels: Cluster assignments
        height_results: Dict from fit_gmm_1d_with_bic
        outdir: Output directory
    """
    n_classes = height_results['n_clusters']
    colors = sns.color_palette('tab10', n_classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Prepare data for boxplot
    data_by_class = [heights[labels == k] for k in range(n_classes)]
    
    bp = ax.boxplot(data_by_class, patch_artist=True, 
                    labels=[f'Class {k}\n(n={len(data_by_class[k])})' for k in range(n_classes)])
    
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)
    
    # Add mean markers
    means = height_results['means']
    for k, mean in enumerate(means):
        ax.scatter([k + 1], [mean], marker='D', s=100, c='black', zorder=5, label='GMM mean' if k == 0 else None)
    
    ax.set_xlabel('Height Class', fontsize=12)
    ax.set_ylabel('Lifetime max height [m]', fontsize=12)
    ax.set_title('Height Distribution by Class', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(fontsize=10)
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / '02_height_boxplots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 02_height_boxplots.png")


def plot_height_cumulative(heights: np.ndarray, labels: np.ndarray,
                           height_results: dict, outdir: Path) -> None:
    """
    Plot cumulative distribution of heights by class.
    
    Args:
        heights: Array of height values [m]
        labels: Cluster assignments
        height_results: Dict from fit_gmm_1d_with_bic
        outdir: Output directory
    """
    n_classes = height_results['n_clusters']
    colors = sns.color_palette('tab10', n_classes)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Plot CDF for each class
    for k in range(n_classes):
        h_k = np.sort(heights[labels == k])
        cdf = np.arange(1, len(h_k) + 1) / len(h_k)
        ax.plot(h_k, cdf, color=colors[k], linewidth=2, label=f'Class {k} (n={len(h_k)})')
    
    # Plot overall CDF
    h_all = np.sort(heights)
    cdf_all = np.arange(1, len(h_all) + 1) / len(h_all)
    ax.plot(h_all, cdf_all, 'k--', linewidth=2, alpha=0.5, label='All clouds')
    
    ax.set_xlabel('Lifetime maximum cloud-top height [m]', fontsize=12)
    ax.set_ylabel('Cumulative fraction', fontsize=12)
    ax.set_title('Cumulative Distribution by Class', fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / '03_height_cumulative.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 03_height_cumulative.png")


def plot_height_boundaries(heights: np.ndarray, labels: np.ndarray,
                           height_results: dict, outdir: Path) -> None:
    """
    Plot class assignments with decision boundaries.
    
    Args:
        heights: Array of height values [m]
        labels: Cluster assignments
        height_results: Dict from fit_gmm_1d_with_bic
        outdir: Output directory
    """
    n_classes = height_results['n_clusters']
    means = height_results['means']
    stds = height_results['stds']
    colors = sns.color_palette('tab10', n_classes)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    
    # Plot each cloud as a point at its height, y-position by class
    for k in range(n_classes):
        h_k = heights[labels == k]
        y_jitter = np.random.uniform(-0.2, 0.2, size=len(h_k))  # small jitter for visibility
        ax.scatter(h_k, np.ones_like(h_k) * k + y_jitter, c=[colors[k]], s=40, alpha=0.6)
    
    # Compute and plot decision boundaries between adjacent classes
    boundaries = []
    if n_classes > 1:
        for k in range(n_classes - 1):
            m1, s1 = means[k], stds[k]
            m2, s2 = means[k + 1], stds[k + 1]
            # Boundary where posterior probabilities are equal (simplified)
            boundary = (m1 * s2 + m2 * s1) / (s1 + s2)
            boundaries.append(boundary)
            ax.axvline(boundary, color='red', linestyle='--', linewidth=2, alpha=0.8)
            ax.text(boundary, n_classes - 0.5, f'{boundary:.0f}m', 
                    ha='center', fontsize=11, color='red', fontweight='bold')
    
    # Add class labels on y-axis
    ax.set_yticks(range(n_classes))
    ax.set_yticklabels([f'Class {k}\n(n={np.sum(labels==k)})' for k in range(n_classes)])
    ax.set_ylim(-0.5, n_classes - 0.5)
    
    ax.set_xlabel('Lifetime maximum cloud-top height [m]', fontsize=12)
    ax.set_ylabel('Class', fontsize=12)
    ax.set_title('Class Assignments and Decision Boundaries', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / '04_height_boundaries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 04_height_boundaries.png")


def plot_all_height_diagnostics(heights: np.ndarray, labels: np.ndarray,
                                 height_results: dict, outdir: Path) -> None:
    """
    Generate all diagnostic plots for height-based classification.
    
    Args:
        heights: Array of height values [m]
        labels: Cluster assignments
        height_results: Dict from fit_gmm_1d_with_bic
        outdir: Output directory
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Plotting Height Classification Diagnostics]")
    plot_height_histogram_with_gmm(heights, labels, height_results, outdir)
    plot_height_boxplots(heights, labels, height_results, outdir)
    plot_height_cumulative(heights, labels, height_results, outdir)
    plot_height_boundaries(heights, labels, height_results, outdir)
    print(f"[Plotting Complete] All plots saved to: {outdir}\n")
