"""
Wasserstein-based clustering for cloud vertical profiles.

Implements clustering using 1D Wasserstein distance (optimal transport)
on normalized profiles j(z), treating them as probability distributions over height.

Methods:
- wasserstein_kmeans: Fixed k, iterative k-means with Wasserstein distance
- wasserstein_auto_k: Hierarchical Ward-like clustering with automatic k selection
"""

import numpy as np
import ot
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import squareform
from sklearn.metrics import silhouette_score


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


# =============================================================================
# Auto-k selection: Hierarchical Ward-like clustering with Wasserstein distance
# =============================================================================

def compute_wasserstein_distance_matrix(profiles: np.ndarray, z_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute pairwise Wasserstein distance matrix between all cloud profiles.
    
    Args:
        profiles: Array [n_clouds, n_levels] of j(z) profiles (will be normalized)
        z_vals: Height values [n_levels]
        
    Returns:
        dist_matrix: Symmetric [n_clouds, n_clouds] pairwise Wasserstein distances
        cost_matrix_normalized: [n_levels, n_levels] cost matrix for barycenter computation
    """
    n_clouds, n_levels = profiles.shape
    
    # Normalize profiles to probability distributions
    profiles_norm = profiles / (profiles.sum(axis=1, keepdims=True) + 1e-12)
    
    # Cost matrix: absolute height difference, normalized to [0, 1]
    cost_matrix = np.abs(z_vals[:, None] - z_vals[None, :])
    cost_matrix_normalized = cost_matrix / (cost_matrix.max() + 1e-12)
    
    # Compute pairwise Wasserstein distances
    print(f"[auto_k] Computing pairwise Wasserstein distances for {n_clouds} clouds...")
    dist_matrix = np.zeros((n_clouds, n_clouds))
    for i in range(n_clouds):
        for j in range(i + 1, n_clouds):
            dist = ot.emd2(profiles_norm[i], profiles_norm[j], cost_matrix_normalized)
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist
    
    return dist_matrix, cost_matrix_normalized


def _preprocess_profiles(profiles: np.ndarray, z_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Preprocess profiles before clustering (outlier handling placeholder).
    
    Currently a pass-through. Future: implement outlier detection/removal.
    
    Args:
        profiles: Array [n_clouds, n_levels] of j(z) profiles
        z_vals: Height values [n_levels]
        
    Returns:
        profiles_clean: Cleaned profiles (same shape or fewer rows if outliers removed)
        keep_mask: Boolean mask indicating which profiles were kept
    """
    n_clouds = profiles.shape[0]
    
    # -------------------------------------------------------------------------
    # TODO: Outlier detection placeholder
    # Options to implement later:
    # - Remove profiles with anomalously high Wasserstein distance to mean
    # - Remove profiles with mass concentrated in single level
    # - Use isolation forest or LOF on distance matrix
    # -------------------------------------------------------------------------
    
    # For now: keep all profiles
    keep_mask = np.ones(n_clouds, dtype=bool)
    profiles_clean = profiles[keep_mask]
    
    return profiles_clean, keep_mask


def _compute_cluster_variance(members: np.ndarray, barycenter: np.ndarray, 
                               cost_matrix: np.ndarray) -> float:
    """
    Compute within-cluster variance as sum of squared Wasserstein distances to barycenter.
    
    Args:
        members: Array [n_members, n_levels] of normalized profiles in cluster
        barycenter: Array [n_levels] cluster barycenter
        cost_matrix: Normalized cost matrix for Wasserstein computation
        
    Returns:
        Total within-cluster variance (sum of W_1^2)
    """
    variance = 0.0
    for member in members:
        dist = ot.emd2(member, barycenter, cost_matrix)
        variance += dist ** 2
    return variance


def _compute_barycenter(members: np.ndarray, cost_matrix: np.ndarray) -> np.ndarray:
    """
    Compute Wasserstein barycenter of a set of profiles.
    
    Args:
        members: Array [n_members, n_levels] of normalized profiles
        cost_matrix: Normalized cost matrix
        
    Returns:
        barycenter: Array [n_levels]
    """
    if len(members) == 1:
        return members[0].copy()
    
    return ot.bregman.barycenter(
        members.T,  # Shape: (n_levels, n_members)
        cost_matrix,
        reg=0.1,
        weights=None,
        numItermax=1000,
        stopThr=1e-6
    )


def _select_k_silhouette(dist_matrix: np.ndarray, linkage_matrix: np.ndarray, 
                          k_max: int) -> tuple[int, dict[int, float]]:
    """
    Select optimal k by maximizing silhouette score.
    
    Args:
        dist_matrix: Precomputed pairwise distance matrix
        linkage_matrix: Scipy linkage matrix from hierarchical clustering
        k_max: Maximum k to evaluate
        
    Returns:
        optimal_k: Best number of clusters
        scores: Dict mapping k -> silhouette score
    """
    n_clouds = dist_matrix.shape[0]
    k_max = min(k_max, n_clouds - 1)  # Can't have more clusters than clouds - 1
    
    scores = {}
    for k in range(2, k_max + 1):
        labels_k = fcluster(linkage_matrix, t=k, criterion='maxclust') - 1  # 0-indexed
        
        # Check that we actually have k clusters (some might be empty after fcluster)
        n_actual = len(np.unique(labels_k))
        if n_actual < 2:
            continue
            
        score = silhouette_score(dist_matrix, labels_k, metric='precomputed')
        scores[k] = score
        print(f"[auto_k]   k={k}: silhouette={score:.4f}")
    
    optimal_k = max(scores, key=scores.get)
    print(f"[auto_k] Selected k={optimal_k} (silhouette={scores[optimal_k]:.4f})")
    
    return optimal_k, scores


# def _select_k_stability(dist_matrix: np.ndarray, linkage_matrix: np.ndarray,
#                          k_max: int, n_bootstrap: int = 50, 
#                          subsample_frac: float = 0.8, random_seed: int = 0) -> tuple[int, dict]:
#     """
#     Select optimal k via bootstrap stability analysis.
#     
#     For each k, subsample the data multiple times and measure how stable
#     the cluster assignments are across subsamples. More stable = better k.
#     
#     TODO: Implement when silhouette proves insufficient.
#     """
#     raise NotImplementedError("Stability-based k selection not yet implemented")


def wasserstein_ward_linkage(profiles: np.ndarray, z_vals: np.ndarray,
                              dist_matrix: np.ndarray = None,
                              cost_matrix: np.ndarray = None) -> np.ndarray:
    """
    Perform Ward-like hierarchical clustering using Wasserstein distance.
    
    Ward's method minimizes within-cluster variance. For Wasserstein geometry,
    variance is defined as sum of squared Wasserstein distances to the cluster
    barycenter. 
    
    Implementation: We use scipy's linkage with 'ward' on the distance matrix.
    Note: scipy's Ward requires the condensed distance matrix and computes
    linkage based on variance increase. For Wasserstein, this is an approximation
    since true Ward would recompute barycenters at each merge. For practical
    purposes with 1D Wasserstein on probability distributions, this works well.
    
    Args:
        profiles: Array [n_clouds, n_levels] of j(z) profiles
        z_vals: Height values [n_levels]
        dist_matrix: Precomputed distance matrix (computed if None)
        cost_matrix: Precomputed cost matrix (computed if None)
        
    Returns:
        linkage_matrix: Scipy linkage matrix [n_clouds-1, 4]
    """
    if dist_matrix is None or cost_matrix is None:
        dist_matrix, cost_matrix = compute_wasserstein_distance_matrix(profiles, z_vals)
    
    # Convert to condensed form for scipy
    condensed = squareform(dist_matrix)
    
    # Ward-like linkage: minimize variance increase at each merge
    # Using 'ward' linkage which minimizes within-cluster variance
    linkage_matrix = linkage(condensed, method='ward')
    
    return linkage_matrix


def wasserstein_auto_k(profiles: np.ndarray, z_vals: np.ndarray, 
                        k_max: int = 10, random_seed: int = 0,
                        selection_method: str = 'silhouette') -> dict:
    """
    Hierarchical Wasserstein clustering with automatic k selection.
    
    Pipeline:
    1. Preprocess profiles (outlier handling placeholder)
    2. Compute pairwise Wasserstein distance matrix
    3. Run Ward-like hierarchical clustering
    4. Select optimal k via silhouette maximization
    5. Compute Wasserstein barycenters as cluster centroids
    
    Args:
        profiles: Array [n_clouds, n_levels] of normalized j(z) profiles
        z_vals: Height values [n_levels]
        k_max: Maximum number of clusters to consider
        random_seed: Random seed (for future stochastic methods)
        selection_method: Method for k selection ('silhouette' or 'stability')
        
    Returns:
        Dictionary containing:
            labels: Cluster assignments [n_clouds], 0-indexed
            centroids: Wasserstein barycenters [n_clusters, n_levels]
            n_clusters: Selected number of clusters
            silhouette_scores: Dict {k: score} for all evaluated k
            linkage_matrix: Scipy linkage matrix for dendrogram
            dist_matrix: Pairwise Wasserstein distance matrix
            keep_mask: Boolean mask of profiles kept after preprocessing
    """
    print(f"[auto_k] Starting hierarchical Wasserstein clustering (k_max={k_max})")
    
    # Step 1: Preprocess profiles (outlier handling placeholder)
    profiles_clean, keep_mask = _preprocess_profiles(profiles, z_vals)
    n_clouds = profiles_clean.shape[0]
    print(f"[auto_k] Profiles after preprocessing: {n_clouds}")
    
    # Step 2: Compute pairwise Wasserstein distance matrix
    dist_matrix, cost_matrix = compute_wasserstein_distance_matrix(profiles_clean, z_vals)
    
    # Normalize profiles for barycenter computation
    profiles_norm = profiles_clean / (profiles_clean.sum(axis=1, keepdims=True) + 1e-12)
    
    # Step 3: Ward-like hierarchical clustering
    print("[auto_k] Running Ward-like hierarchical clustering...")
    linkage_matrix = wasserstein_ward_linkage(
        profiles_clean, z_vals, dist_matrix=dist_matrix, cost_matrix=cost_matrix
    )
    
    # Step 4: Select optimal k
    print(f"[auto_k] Selecting optimal k via {selection_method}...")
    if selection_method == 'silhouette':
        optimal_k, silhouette_scores = _select_k_silhouette(dist_matrix, linkage_matrix, k_max)
    # elif selection_method == 'stability':
    #     optimal_k, silhouette_scores = _select_k_stability(dist_matrix, linkage_matrix, k_max)
    else:
        raise ValueError(f"Unknown selection method: {selection_method}")
    
    # Step 5: Get final cluster labels
    labels = fcluster(linkage_matrix, t=optimal_k, criterion='maxclust') - 1  # 0-indexed
    
    # Step 6: Compute Wasserstein barycenters for each cluster
    print(f"[auto_k] Computing Wasserstein barycenters for {optimal_k} clusters...")
    centroids = np.zeros((optimal_k, profiles_clean.shape[1]))
    for k in range(optimal_k):
        members = profiles_norm[labels == k]
        if len(members) > 0:
            centroids[k] = _compute_barycenter(members, cost_matrix)
    
    # -------------------------------------------------------------------------
    # TODO: Minimum cluster size enforcement (placeholder)
    # If implemented, small clusters would be merged into nearest neighbor
    # min_cluster_size = None  # Set to integer to enable
    # if min_cluster_size is not None:
    #     labels, centroids = _enforce_min_cluster_size(
    #         labels, centroids, dist_matrix, min_cluster_size
    #     )
    # -------------------------------------------------------------------------
    
    print(f"[auto_k] Complete. Selected {optimal_k} clusters.")
    cluster_counts = {k: int(np.sum(labels == k)) for k in range(optimal_k)}
    print(f"[auto_k] Cluster sizes: {cluster_counts}")
    
    return {
        'labels': labels,
        'centroids': centroids,
        'n_clusters': optimal_k,
        'silhouette_scores': silhouette_scores,
        'linkage_matrix': linkage_matrix,
        'dist_matrix': dist_matrix,
        'keep_mask': keep_mask,
    }
