"""
Wasserstein k-means clustering for cloud vertical profiles.

Implements k-means clustering using 1D Wasserstein distance (optimal transport)
on normalized profiles j(z), treating them as probability distributions over height.
"""

import numpy as np
import ot


def wasserstein_kmeans(profiles, z_vals, n_clusters=3, max_iter=100, random_seed=0):
    """
    Cluster profiles using 1D Wasserstein distance on j(z).
    
    Args:
        profiles: Array [n_clouds, n_levels] of normalized j(z) profiles
        z_vals: Height values [n_levels]
        n_clusters: Number of clusters
        max_iter: Maximum iterations
        random_seed: Random seed for initialization
        
    Returns:
        labels: Cluster assignments [n_clouds]
        centroids: Cluster centroids [n_clusters, n_levels]
        n_iter: Number of iterations until convergence
    """
    n_clouds, n_levels = profiles.shape
    rng = np.random.default_rng(random_seed)
    
    # Normalize profiles to be valid probability distributions
    profiles_norm = profiles / (profiles.sum(axis=1, keepdims=True) + 1e-12)
    
    # Create cost matrix: distance between height levels
    # For 1D Wasserstein on a grid, cost is absolute difference in height
    # Normalize to [0, 1] to avoid numerical issues in barycenter computation
    cost_matrix = np.abs(z_vals[:, None] - z_vals[None, :])
    cost_matrix_normalized = cost_matrix / (cost_matrix.max() + 1e-12)
    
    # k-means++ initialization
    centroids = _kmeans_plus_plus_init(profiles_norm, n_clusters, cost_matrix_normalized, rng)
    
    labels = np.zeros(n_clouds, dtype=int)
    
    for iteration in range(max_iter):
        # Assignment step: assign each profile to nearest centroid
        new_labels = np.zeros(n_clouds, dtype=int)
        for i in range(n_clouds):
            distances = np.zeros(n_clusters)
            for k in range(n_clusters):
                # Compute Wasserstein distance between profile i and centroid k
                distances[k] = ot.emd2(profiles_norm[i], centroids[k], cost_matrix_normalized)
            new_labels[i] = np.argmin(distances)
        
        # Check convergence
        if np.array_equal(labels, new_labels):
            print(f"[Wasserstein] Converged after {iteration} iterations")
            break
        
        labels = new_labels
        
        # Update step: compute Wasserstein barycenter for each cluster
        for k in range(n_clusters):
            cluster_members = profiles_norm[labels == k]
            if len(cluster_members) > 0:
                if len(cluster_members) == 1:
                    # Single member: barycenter is itself
                    centroids[k] = cluster_members[0]
                else:
                    # Compute true Wasserstein barycenter using Sinkhorn algorithm
                    # barycenter expects: (n_features, n_distributions)
                    centroids[k] = ot.bregman.barycenter(
                        cluster_members.T,  # Shape: (n_levels, n_members)
                        cost_matrix_normalized,
                        reg=0.1,  # Entropic regularization (larger for stability)
                        weights=None,  # Uniform weights over cluster members
                        numItermax=1000,
                        stopThr=1e-6
                    )
    else:
        print(f"[Wasserstein] Reached max iterations ({max_iter})")
    
    return labels, centroids, iteration + 1


def _kmeans_plus_plus_init(profiles, n_clusters, cost_matrix, rng):
    """k-means++ initialization for Wasserstein k-means."""
    n_clouds = len(profiles)
    
    # First centroid: random
    centroids = [profiles[rng.integers(0, n_clouds)].copy()]
    
    # Subsequent centroids: pick proportional to distance squared
    for _ in range(1, n_clusters):
        # Compute min distance to existing centroids for each profile
        min_dists = np.full(n_clouds, np.inf)
        for i in range(n_clouds):
            for c in centroids:
                # 1D Wasserstein distance
                dist = ot.emd2(profiles[i], c, cost_matrix)
                min_dists[i] = min(min_dists[i], dist)
        
        # Pick next centroid proportional to squared distance
        probs = min_dists**2
        probs /= probs.sum()
        next_idx = rng.choice(n_clouds, p=probs)
        centroids.append(profiles[next_idx].copy())
    
    return np.array(centroids)
