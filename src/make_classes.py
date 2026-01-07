import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture

# Use local helpers in this repo instead of CloudTracking submodule
from features import reduce_all_tracks
from density import compute_rho0_from_raw
from plotting import plot_all_diagnostics
from wasserstein_clustering import wasserstein_kmeans, wasserstein_auto_k
from height_classification import (
    get_lifetime_max_heights, fit_gmm_1d_with_bic, 
    compute_class_height_stats, plot_all_height_diagnostics
)

"""
First-pass lifetime class learning from CloudTracking output (RICO).

Physics sketch:
- Compute reference density profile rho0(z) from raw data
- Reduce each cloud track to lifetime vertical profiles using updraft-only transport
- Normalize each cloud's vertical shape: j(z) = J(z)/M
- Perform PCA on j(z) to capture dominant vertical structure modes
- Cluster clouds using Gaussian Mixture on [PCs, logM]
- Save per-class vertical templates phi_k(z) and per-cloud labels
"""


def parse_args():
    p = argparse.ArgumentParser("make lifetime classes (v1)")
    # Clustering method (REQUIRED)
    p.add_argument("--clustering-method", required=True,
                   choices=["gmm", "wasserstein", "auto_k", "height"],
                   help="clustering method: gmm (Gaussian Mixture), wasserstein (fixed k), auto_k (hierarchical with auto k selection), or height (1D GMM on max cloud height)")
    # Path to CloudTracker output file containing tracked cloud data
    p.add_argument("--cloud-nc", default="../../cloud_results.nc", help="path to cloud_results.nc")
    # Path to directory with raw RICO NetCDF files (l, q, p, t) - not used by height method
    p.add_argument("--raw-path", default="../../../RICO_1hr/", help="path to RICO raw data (not used by height method)")
    # Number of cloud classes to learn (ignored for auto_k and height)
    p.add_argument("--n-classes", type=int, default=3, help="number of cloud classes (ignored for auto_k and height)")
    # Maximum number of classes for auto_k and height methods
    p.add_argument("--k-max", type=int, default=10, help="maximum k to consider for auto_k/height methods")
    # Number of principal components to use in clustering (not used by height)
    p.add_argument("--n-pcs", type=int, default=3, help="number of PCA components (not used by height)")
    # Minimum number of active timesteps for a cloud to be included
    p.add_argument("--min-timesteps", type=int, default=3, help="minimum timesteps per cloud")
    # Whether to use only positive (upward) mass flux (not used by height)
    p.add_argument("--positive-only", type=int, default=1, help="1=updraft only, 0=all vertical motion (not used by height)")
    # Fraction of spatial domain to sample when computing rho0 (not used by height)
    p.add_argument("--rho-sample-frac", type=float, default=0.05, help="spatial sampling fraction for rho0 (not used by height)")
    # Whether to generate diagnostic plots
    p.add_argument("--no-plots", action="store_true", help="disable diagnostic plotting")
    # Number of individual cloud profiles to plot (not used by height)
    p.add_argument("--n-sample-clouds", type=int, default=5, help="number of individual clouds to plot (not used by height)")
    return p.parse_args()


def main():
    args = parse_args()
    # Use absolute path for artefacts directory (in project root, not src/)
    outdir = Path(__file__).parent.parent / "artefacts"
    outdir.mkdir(parents=True, exist_ok=True)

    ds = xr.open_dataset(args.cloud_nc)
    z = ds["height"]
    z_vals = z.values
    # Fixed timestep for lifetime integration
    dt = 60.0
    
    method = args.clustering_method
    
    # ==========================================================================
    # HEIGHT METHOD: Simple classification based only on lifetime max height
    # ==========================================================================
    if method == "height":
        print("Height-based classification (1D GMM on lifetime max cloud height)")
        
        # Extract heights directly from dataset - no profile reduction needed
        heights, track_ids = get_lifetime_max_heights(
            ds, min_timesteps=args.min_timesteps, only_valid=True
        )
        
        if len(heights) == 0:
            print("ERROR: No clouds with valid height data.")
            exit(1)
        
        print(f"Clouds with valid height data: {len(heights)}")
        
        # Fit 1D GMM with automatic k selection via BIC
        height_results = fit_gmm_1d_with_bic(heights, k_max=args.k_max, random_seed=0)
        labels = height_results['labels']
        n_classes = height_results['n_clusters']
        
        # Compute statistics
        stats = compute_class_height_stats(heights, labels)
        
        # Save per-cloud labels to Parquet
        data = {
            "cloud_id": track_ids,
            "class_k": labels.astype(int),
            "height_max": heights,
        }
        pd.DataFrame(data).to_parquet(outdir / "cloud_labels.parquet", index=False)
        
        # Print summary
        counts = {k: int(np.sum(labels == k)) for k in range(n_classes)}
        print(f"\nClustering method: {method}")
        print(f"Total clouds: {len(heights)}")
        print(f"Auto-selected k (BIC): {n_classes}")
        print(f"BIC scores: {height_results['bic_scores']}")
        print("Height class statistics:")
        for k in range(n_classes):
            print(f"  Class {k}: n={stats['counts'][k]}, "
                  f"mean={stats['means'][k]:.0f}m, "
                  f"range=[{stats['min'][k]:.0f}, {stats['max'][k]:.0f}]m")
        print(f"Class counts: {counts}")
        
        # Generate plots
        if not args.no_plots:
            plotdir = Path(__file__).parent.parent / "plots" / method
            plotdir = Path(__file__).parent.parent / "plots" / "height_gmm"
            plot_all_height_diagnostics(
                heights=heights,
                labels=labels,
                height_results=height_results,
                outdir=plotdir,
            )
        
        return  # Done - exit early for height method
    
    # ==========================================================================
    # OTHER METHODS: Require profile reduction, PCA, etc.
    # ==========================================================================

    # Compute reference density profile rho0(z) from raw RICO data
    rho0 = compute_rho0_from_raw(
        base_path=args.raw_path,
        reduce="median",
        sample_frac=args.rho_sample_frac,
        time_max=5
    )
    
    # Ensure rho0 has vertical dimension named 'z'
    if "z" not in rho0.dims:
        rho0 = rho0.rename({list(rho0.dims)[0]: "z"})
    
    # Interpolate rho0 to match cloud grid
    rho0 = rho0.interp(z=xr.DataArray(z_vals, dims=["z"], coords={"z": z_vals}))

    # Reduce all tracks to lifetime vertical profiles (use valid tracks only!)
    clouds = reduce_all_tracks(
        ds,
        dt=dt,
        rho0=rho0,
        only_valid=True,
        min_timesteps=int(args.min_timesteps),
        positive_only=bool(args.positive_only),
    )

    if len(clouds) == 0:
        print("ERROR: No clouds passed the filtering criteria.")
        exit(1)

    # Extract normalized vertical shapes j(z) = J(z)/M for each cloud
    j_rows = []
    Ms = []
    Tcs = []
    cloud_ids = []
    for c in clouds:
        J = c["J"].values
        M = np.nansum(J)
        if M == 0 or not np.isfinite(M):
            continue
        Ms.append(M)
        j_rows.append(J / M)
        Tcs.append(c.attrs.get("T_c", np.nan))
        cloud_ids.append(c.attrs.get("track_index"))

    if len(j_rows) == 0:
        print("ERROR: All clouds have zero or invalid mass flux.")
        exit(1)

    # Stack into matrix [n_clouds, n_levels]
    Jmat = np.vstack(j_rows)
    Jmat_filled = np.nan_to_num(Jmat, nan=0.0, posinf=0.0, neginf=0.0)

    # Center the data for PCA
    Jmean = Jmat_filled.mean(axis=0)
    J0 = Jmat_filled - Jmean

    # Principal component analysis
    n_pcs = int(args.n_pcs)
    pca = PCA(n_components=n_pcs)
    pcs = pca.fit_transform(J0)
    exp_var_cum = float(np.sum(pca.explained_variance_ratio_))

    # Build feature vector: [PC1, PC2, PC3, log(M)]
    logM = np.log(np.maximum(np.asarray(Ms), 1e-9))
    X = np.column_stack([pcs, logM])
    X_mean = X.mean(axis=0)
    X_std = X.std(axis=0)
    X_std[X_std == 0] = 1.0
    Xs = (X - X_mean) / X_std

    # Cluster clouds using selected method
    n_classes = int(args.n_classes)
    method = args.clustering_method
    
    # Initialize variables that may or may not be set depending on method
    n_iter = None
    auto_k_results = None
    
    print(f"Clustering using method: {method}")
    
    if method == "gmm":
        # Gaussian Mixture Model on PCA features + logM
        gmm = GaussianMixture(n_components=n_classes, covariance_type="full", random_state=0)
        labels = gmm.fit_predict(Xs)
        
    elif method == "wasserstein":
        # Wasserstein k-means on normalized profiles j(z)
        labels, centroids, n_iter = wasserstein_kmeans(
            Jmat_filled, z_vals, n_clusters=n_classes, max_iter=100, random_seed=0
        )
    
    elif method == "auto_k":
        # Hierarchical Wasserstein clustering with automatic k selection
        auto_k_results = wasserstein_auto_k(
            Jmat_filled, z_vals, k_max=args.k_max, random_seed=0
        )
        labels = auto_k_results['labels']
        n_classes = auto_k_results['n_clusters']
    
    else:
        raise ValueError(f"Unknown clustering method: {method}")


    # Compute per-class vertical templates phi_k(z)
    phi = np.zeros((n_classes, Jmat_filled.shape[1]))
    phi_p10 = np.zeros_like(phi)
    phi_p90 = np.zeros_like(phi)
    for k in range(n_classes):
        members = Jmat_filled[labels == k]
        if members.size > 0:
            phi[k] = members.mean(axis=0)
            phi_p10[k] = np.quantile(members, 0.10, axis=0)
            phi_p90[k] = np.quantile(members, 0.90, axis=0)

    # Save class templates to NetCDF
    ds_out = xr.Dataset(
        {
            "phi": xr.DataArray(
                phi, dims=["k", "z"], coords={"k": np.arange(n_classes), "z": z_vals}
            ),
            "phi_p10": xr.DataArray(
                phi_p10, dims=["k", "z"], coords={"k": np.arange(n_classes), "z": z_vals}
            ),
            "phi_p90": xr.DataArray(
                phi_p90, dims=["k", "z"], coords={"k": np.arange(n_classes), "z": z_vals}
            ),
        }
    )
    for i, r in enumerate(pca.explained_variance_ratio_):
        ds_out.attrs[f"pca_explained_variance_ratio_{i+1}"] = float(r)
    ds_out.to_netcdf(outdir / "class_templates.nc", engine="h5netcdf")

    # Save per-cloud labels and features to Parquet
    # We save the original track index (cloud_id) so we can map back to raw data
    
    data = {
        "cloud_id": np.array(cloud_ids),  # Use actual track indices, not 0..N
        "class_k": labels.astype(int),
        "logM": logM,
        "T_c": np.array(Tcs),
    }
    for i in range(pcs.shape[1]):
        data[f"PC{i+1}"] = pcs[:, i]
    pd.DataFrame(data).to_parquet(outdir / "cloud_labels.parquet", index=False)

    # Print summary
    counts = {k: int(np.sum(labels == k)) for k in range(n_classes)}
    print(f"Reduced clouds: {len(labels)}")
    print(f"Clustering method: {method}")
    if n_iter is not None:
        print(f"Converged in {n_iter} iterations")
    if method == "gmm":
        print(f"PCA variance explained: {exp_var_cum:.3f}")
    if method == "auto_k" and auto_k_results is not None:
        print(f"Auto-selected k: {n_classes}")
        print(f"Silhouette scores: {auto_k_results['silhouette_scores']}")
    print(f"Class counts: {counts}")

    # Generate diagnostic plots
    if not args.no_plots:
        # Create method-specific subdirectory in base plots folder (not src/)
        plotdir = Path(__file__).parent.parent / "plots" / method
        plotdir.mkdir(parents=True, exist_ok=True)
        
        plot_all_diagnostics(
            rho0=rho0,
            z_vals=z_vals,
            clouds=clouds,
            Jmat=Jmat_filled,
            Jmean=Jmean,
            pca=pca,
            pcs=pcs,
            logM=logM,
            labels=labels,
            phi=phi,
            phi_p10=phi_p10,
            phi_p90=phi_p90,
            outdir=plotdir,
            n_sample_clouds=args.n_sample_clouds,
            clustering_method=method,
            auto_k_results=auto_k_results,
        )


if __name__ == "__main__":
    main()
