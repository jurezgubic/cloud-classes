"""
Post-processing diagnostics for cloud clusters.

This script loads the cluster labels and the original cloud tracking data
to compute physical statistics for each cluster, such as:
- Cloud lifetime (from actual age variable)
- Cloud base/top height
- Mean/Max vertical velocity
- Cloud size/volume
- Merge and split counts
- Cloud radii
- Vertical profiles of w, area, and compactness

It generates boxplots, profile plots, and summary tables to help interpret
the physical meaning of each cluster.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

# Time step in seconds (for converting age steps to real time)
DT_SECONDS = 60.0


def parse_args():
    # Determine project root (parent of src/) to set robust default paths
    project_root = Path(__file__).resolve().parent.parent
    
    p = argparse.ArgumentParser("Cluster physical diagnostics")
    p.add_argument("--cloud-nc", default="/Users/jure/PhD/coding/tracking/cloud_results.nc", help="path to cloud_results.nc")
    p.add_argument("--labels-parquet", default=str(project_root / "artefacts/cloud_labels.parquet"), help="path to cloud_labels.parquet")
    p.add_argument("--outdir", default=str(project_root / "plots/cluster_physics"), help="output directory for plots")
    return p.parse_args()

def compute_cloud_physics(ds: xr.Dataset, cloud_ids: np.ndarray) -> pd.DataFrame:
    """
    Compute physical properties for a list of cloud IDs.
    
    Uses the actual age variable from cloud tracking to determine lifetime,
    and includes merge/split counts and radius statistics.
    
    Args:
        ds: xarray.Dataset with cloud tracking data
        cloud_ids: Array of cloud indices (track dimension) to process
        
    Returns:
        DataFrame with physical properties for each cloud
    """
    stats = []
    
    # Get height coordinate
    z = ds['height'].values
    
    print(f"Computing physics for {len(cloud_ids)} clouds...")
    
    for i, idx in enumerate(cloud_ids):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(cloud_ids)}", end='\r')
            
        # Check if track is valid
        if not ds['valid_track'].isel(track=idx).values:
            continue
        
        # -- Lifetime from actual age variable --
        # Use the final age value - this is the actual age of the cloud at its last timestep
        age = ds['age'].isel(track=idx).values
        valid_age = age[np.isfinite(age)]
        if len(valid_age) == 0:
            continue
        final_age = float(valid_age[-1])  # Last recorded age value
        lifetime_s = final_age * DT_SECONDS
        
        # -- Merge and split counts --
        # These are per-timestep flags (1 when event occurred), so count the 1s
        merges = ds['merges_count'].isel(track=idx).values
        splits = ds['splits_count'].isel(track=idx).values
        total_merges = int(np.nansum(merges == 1))
        total_splits = int(np.nansum(splits == 1))
        
        # -- Radius statistics --
        # max_equiv_radius: maximum equivalent radius at each timestep
        max_equiv_rad = ds['max_equiv_radius'].isel(track=idx).values
        max_radius = float(np.nanmax(max_equiv_rad)) if np.any(np.isfinite(max_equiv_rad)) else np.nan
        mean_radius = float(np.nanmean(max_equiv_rad)) if np.any(np.isfinite(max_equiv_rad)) else np.nan
        
        # base_radius_diagnosed: radius at cloud base
        base_rad = ds['base_radius_diagnosed'].isel(track=idx).values
        mean_base_radius = float(np.nanmean(base_rad)) if np.any(np.isfinite(base_rad)) else np.nan
        max_base_radius = float(np.nanmax(base_rad)) if np.any(np.isfinite(base_rad)) else np.nan
            
        # -- Get time-height arrays for cloud structure --
        area = ds['area_per_level'].isel(track=idx).values
        w = ds['w_per_level'].isel(track=idx).values
        
        # Mask for active cloud points
        is_cloud = np.isfinite(area) & (area > 0)
        
        if not np.any(is_cloud):
            continue
            
        # -- Cloud Base and Top statistics --
        cloud_bases = []
        cloud_tops = []
        
        # Iterate over time steps where cloud exists
        for t in range(area.shape[0]):
            profile = is_cloud[t, :]
            if np.any(profile):
                z_cloud = z[profile]
                cloud_bases.append(z_cloud.min())
                cloud_tops.append(z_cloud.max())
        
        mean_base = np.mean(cloud_bases) if cloud_bases else np.nan
        mean_top = np.mean(cloud_tops) if cloud_tops else np.nan
        max_top = np.max(cloud_tops) if cloud_tops else np.nan
        
        # -- Vertical Velocity --
        w_cloud = np.where(is_cloud, w, np.nan)
        mean_w = float(np.nanmean(w_cloud))
        max_w = float(np.nanmax(w_cloud))
        
        # -- Cloud Size (max area) --
        max_area = float(np.nanmax(area))
        mean_area = float(np.nanmean(np.where(is_cloud, area, np.nan)))
        
        stats.append({
            'cloud_id': int(idx),
            'lifetime_s': lifetime_s,
            'total_merges': total_merges,
            'total_splits': total_splits,
            'max_radius_m': max_radius,
            'mean_radius_m': mean_radius,
            'mean_base_radius_m': mean_base_radius,
            'max_base_radius_m': max_base_radius,
            'mean_base_m': mean_base,
            'mean_top_m': mean_top,
            'max_top_m': max_top,
            'depth_m': mean_top - mean_base,
            'mean_w_ms': mean_w,
            'max_w_ms': max_w,
            'mean_area_m2': mean_area,
            'max_area_m2': max_area
        })
    
    print()  # newline after progress
    return pd.DataFrame(stats)

def plot_cluster_physics(df: pd.DataFrame, outdir: Path) -> None:
    """Generate boxplots for physical properties by cluster."""
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    # Define variables to plot
    vars_to_plot = [
        ('lifetime_s', 'Lifetime [s]'),
        ('mean_base_m', 'Mean Cloud Base [m]'),
        ('mean_top_m', 'Mean Cloud Top [m]'),
        ('max_top_m', 'Max Cloud Top [m]'),
        ('depth_m', 'Mean Depth [m]'),
        ('mean_w_ms', 'Mean Vertical Velocity [m/s]'),
        ('max_w_ms', 'Max Vertical Velocity [m/s]'),
        ('max_area_m2', 'Max Area [m²]')
    ]
    
    n_vars = len(vars_to_plot)
    n_cols = 2
    n_rows = (n_vars + 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 4 * n_rows))
    axes = axes.flatten()
    
    for i, (col, label) in enumerate(vars_to_plot):
        ax = axes[i]
        
        # Boxplot
        sns.boxplot(x='class_k', y=col, hue='class_k', data=df, ax=ax, 
                    palette='tab10', showfliers=False, legend=False)
        
        # Add mean markers
        sns.pointplot(x='class_k', y=col, data=df, ax=ax, 
                      estimator=np.mean, color='black', markers='D', 
                      markersize=6, linestyle='none')
        
        ax.set_xlabel('Cluster Class')
        ax.set_ylabel(label)
        ax.set_title(label)
        ax.grid(True, alpha=0.3)
        
        # Log scale for area if needed
        if 'area' in col:
            ax.set_yscale('log')
            
    plt.tight_layout()
    plt.savefig(outdir / 'cluster_physics_boxplots.png', dpi=150)
    plt.close()
    print(f"Saved physics boxplots to {outdir / 'cluster_physics_boxplots.png'}")
    
    # Save summary table
    summary = df.groupby('class_k')[['lifetime_s', 'mean_top_m', 'max_top_m', 'mean_w_ms', 'max_area_m2']].agg(['mean', 'std'])
    summary.to_csv(outdir / 'cluster_physics_summary.csv')
    print(f"Saved summary table to {outdir / 'cluster_physics_summary.csv'}")


def plot_lifespan_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot cloud lifespan comparison by cluster class.
    
    Shows lifetime distribution from the actual age variable.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Convert to minutes for better readability
    df = df.copy()
    df['lifetime_min'] = df['lifetime_s'] / 60.0
    
    # Left: Boxplot of lifetime
    ax = axes[0]
    sns.boxplot(x='class_k', y='lifetime_min', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='lifetime_min', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Lifetime [min]')
    ax.set_title('Cloud Lifespan by Cluster Class')
    ax.grid(True, alpha=0.3)
    
    # Right: Violin plot for distribution shape
    ax = axes[1]
    sns.violinplot(x='class_k', y='lifetime_min', hue='class_k', data=df, ax=ax, 
                   palette='tab10', inner='quartile', legend=False)
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Lifetime [min]')
    ax.set_title('Cloud Lifespan Distribution')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'lifespan_comparison.png', dpi=150)
    plt.close()
    print(f"Saved lifespan plot to {outdir / 'lifespan_comparison.png'}")


def plot_merge_split_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot merge and split counts by cluster class.
    
    Compares how often clouds in each class undergo merging or splitting events.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Left: Merges boxplot
    ax = axes[0]
    sns.boxplot(x='class_k', y='total_merges', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='total_merges', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Total Merge Count')
    ax.set_title('Cloud Merges by Class')
    ax.grid(True, alpha=0.3)
    
    # Middle: Splits boxplot
    ax = axes[1]
    sns.boxplot(x='class_k', y='total_splits', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='total_splits', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Total Split Count')
    ax.set_title('Cloud Splits by Class')
    ax.grid(True, alpha=0.3)
    
    # Right: Stacked bar chart of mean merge+split
    ax = axes[2]
    class_means = df.groupby('class_k')[['total_merges', 'total_splits']].mean()
    class_means.plot(kind='bar', stacked=True, ax=ax, color=['#1f77b4', '#ff7f0e'])
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Mean Count')
    ax.set_title('Mean Merges & Splits per Cloud')
    ax.legend(['Merges', 'Splits'], loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
    
    plt.tight_layout()
    plt.savefig(outdir / 'merge_split_comparison.png', dpi=150)
    plt.close()
    print(f"Saved merge/split plot to {outdir / 'merge_split_comparison.png'}")


def plot_radius_comparison(df: pd.DataFrame, outdir: Path) -> None:
    """
    Plot cloud radius statistics by cluster class.
    
    Compares maximum equivalent radius and base radius across classes.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Top-left: Max equivalent radius
    ax = axes[0, 0]
    sns.boxplot(x='class_k', y='max_radius_m', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='max_radius_m', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Max Equivalent Radius [m]')
    ax.set_title('Maximum Cloud Radius by Class')
    ax.grid(True, alpha=0.3)
    
    # Top-right: Mean equivalent radius
    ax = axes[0, 1]
    sns.boxplot(x='class_k', y='mean_radius_m', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='mean_radius_m', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Mean Equivalent Radius [m]')
    ax.set_title('Mean Cloud Radius by Class')
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Mean base radius
    ax = axes[1, 0]
    sns.boxplot(x='class_k', y='mean_base_radius_m', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='mean_base_radius_m', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Mean Base Radius [m]')
    ax.set_title('Cloud Base Radius by Class')
    ax.grid(True, alpha=0.3)
    
    # Bottom-right: Max base radius
    ax = axes[1, 1]
    sns.boxplot(x='class_k', y='max_base_radius_m', hue='class_k', data=df, ax=ax, 
                palette='tab10', showfliers=False, legend=False)
    sns.pointplot(x='class_k', y='max_base_radius_m', data=df, ax=ax,
                  estimator=np.mean, color='black', markers='D', 
                  markersize=6, linestyle='none')
    ax.set_xlabel('Cluster Class')
    ax.set_ylabel('Max Base Radius [m]')
    ax.set_title('Maximum Base Radius by Class')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(outdir / 'radius_comparison.png', dpi=150)
    plt.close()
    print(f"Saved radius plot to {outdir / 'radius_comparison.png'}")


def compute_vertical_profiles(ds: xr.Dataset, labels_df: pd.DataFrame) -> dict:
    """
    Compute mean vertical profiles of w, area, and compactness for each cluster class.
    
    For each class, averages over all clouds and all timesteps to get a representative
    vertical distribution of these properties.
    
    Args:
        ds: xarray.Dataset with cloud tracking data
        labels_df: DataFrame with cloud_id and class_k columns
        
    Returns:
        Dictionary with height array and profile statistics for each variable and class
    """
    z = ds['height'].values
    n_levels = len(z)
    classes = sorted(labels_df['class_k'].unique())
    
    # Variables to profile: (nc_name, label)
    profile_vars = [
        ('w_per_level', 'Vertical Velocity [m/s]'),
        ('area_per_level', 'Area [m²]'),
        ('compactness_per_level', 'Compactness'),
        ('equiv_radius_per_level', 'Equivalent Radius [m]')
    ]
    
    results = {'height': z, 'classes': classes, 'profiles': {}}
    
    # Preload valid_track to avoid repeated xarray calls
    valid_track = ds['valid_track'].values
    
    n_vars = len(profile_vars)
    for var_idx, (var_name, var_label) in enumerate(profile_vars):
        print(f"  Variable {var_idx+1}/{n_vars}: {var_name}")
        
        # Preload entire variable into numpy array (avoids repeated .isel() calls)
        var_data = ds[var_name].values  # shape: (track, time, level)
        
        results['profiles'][var_name] = {
            'label': var_label,
            'mean': {},
            'std': {},
            'raw': {}  # Store raw profiles for distribution analysis
        }
        
        for k in classes:
            # Get cloud IDs for this class
            cloud_ids = labels_df[labels_df['class_k'] == k]['cloud_id'].values
            print(f"    Class {k}: {len(cloud_ids)} clouds...", end=" ", flush=True)
            
            # Collect all profiles for this class
            all_profiles = []
            
            for idx in cloud_ids:
                if not valid_track[idx]:
                    continue
                    
                # Get variable data: shape (time, level)
                data = var_data[idx]
                
                # For each timestep, if cloud exists, add profile
                for t in range(data.shape[0]):
                    profile = data[t, :]
                    if np.any(np.isfinite(profile) & (profile != 0)):
                        all_profiles.append(profile)
            
            if len(all_profiles) == 0:
                print("no valid profiles")
                results['profiles'][var_name]['mean'][k] = np.full(n_levels, np.nan)
                results['profiles'][var_name]['std'][k] = np.full(n_levels, np.nan)
                results['profiles'][var_name]['raw'][k] = np.array([])
                continue
                
            # Stack and compute statistics
            stacked = np.array(all_profiles)
            results['profiles'][var_name]['mean'][k] = np.nanmean(stacked, axis=0)
            results['profiles'][var_name]['std'][k] = np.nanstd(stacked, axis=0)
            results['profiles'][var_name]['raw'][k] = stacked  # Store for distribution analysis
            print(f"{len(all_profiles)} profiles")
    
    return results


def plot_distribution_at_heights(profile_data: dict, outdir: Path, 
                                  var_name: str = 'area_per_level',
                                  heights_m: list = [800, 1000, 1200, 1500]) -> None:
    """
    Plot histograms of profile values at selected heights for each class.
    
    This helps diagnose whether the distributions are Gaussian or skewed/multimodal.
    
    Args:
        profile_data: Output from compute_vertical_profiles
        outdir: Output directory
        var_name: Which variable to analyze
        heights_m: List of heights (in meters) to examine
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    z = profile_data['height']
    classes = profile_data['classes']
    var_data = profile_data['profiles'][var_name]
    
    # Find closest level indices for requested heights
    height_indices = [np.argmin(np.abs(z - h)) for h in heights_m]
    actual_heights = [z[i] for i in height_indices]
    
    n_heights = len(heights_m)
    n_classes = len(classes)
    
    fig, axes = plt.subplots(n_heights, n_classes, figsize=(5 * n_classes, 4 * n_heights))
    if n_classes == 1:
        axes = axes.reshape(-1, 1)
    if n_heights == 1:
        axes = axes.reshape(1, -1)
    
    colors = plt.cm.tab10(np.linspace(0, 1, n_classes))
    
    for i, (h_idx, h_actual) in enumerate(zip(height_indices, actual_heights)):
        for j, k in enumerate(classes):
            ax = axes[i, j]
            
            raw = var_data['raw'].get(k, np.array([]))
            if raw.size == 0:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Get values at this height level
            values = raw[:, h_idx]
            values = values[np.isfinite(values) & (values != 0)]
            
            if len(values) == 0:
                ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                continue
            
            # Plot histogram
            ax.hist(values, bins=50, density=True, alpha=0.7, color=colors[j], edgecolor='black', linewidth=0.5)
            
            # Add mean and median lines
            mean_val = np.mean(values)
            median_val = np.median(values)
            ax.axvline(mean_val, color='red', linestyle='-', linewidth=2, label=f'Mean: {mean_val:.0f}')
            ax.axvline(median_val, color='green', linestyle='--', linewidth=2, label=f'Median: {median_val:.0f}')
            
            # Compute skewness
            if len(values) > 2:
                skew = ((values - mean_val) ** 3).mean() / (values.std() ** 3)
                ax.set_title(f'Class {k} @ {h_actual:.0f}m\nN={len(values)}, skew={skew:.2f}')
            else:
                ax.set_title(f'Class {k} @ {h_actual:.0f}m\nN={len(values)}')
            
            ax.legend(fontsize=8)
            ax.set_xlabel(var_data['label'])
            ax.set_ylabel('Density')
            ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    outpath = outdir / f'distribution_{var_name}.png'
    plt.savefig(outpath, dpi=150)
    plt.close()
    print(f"Saved distribution plot to {outpath}")


def plot_vertical_profiles(profile_data: dict, outdir: Path) -> None:
    """
    Plot mean vertical profiles with std shading for each variable.
    
    Creates one subplot per variable, showing all classes on the same axes.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    z = profile_data['height']
    classes = profile_data['classes']
    profiles = profile_data['profiles']
    
    # Color palette
    colors = plt.cm.tab10(np.linspace(0, 1, len(classes)))
    
    n_vars = len(profiles)
    fig, axes = plt.subplots(1, n_vars, figsize=(5 * n_vars, 8))
    if n_vars == 1:
        axes = [axes]
    
    for ax, (var_name, var_data) in zip(axes, profiles.items()):
        for i, k in enumerate(classes):
            mean_profile = var_data['mean'][k]
            std_profile = var_data['std'][k]
            
            # Plot mean line
            ax.plot(mean_profile, z, color=colors[i], linewidth=2, label=f'Class {k}')
            
            # Add std shading
            ax.fill_betweenx(z, 
                             mean_profile - std_profile, 
                             mean_profile + std_profile,
                             color=colors[i], alpha=0.2)
        
        ax.set_xlabel(var_data['label'])
        ax.set_ylabel('Height [m]')
        ax.set_title(f"Vertical Profile: {var_data['label']}")
        ax.grid(True, alpha=0.3)
        ax.legend(loc='upper right')
        
        # Limit height range to where clouds typically exist
        ax.set_ylim([0, 4000])
    
    plt.tight_layout()
    plt.savefig(outdir / 'vertical_profiles.png', dpi=150)
    plt.close()
    print(f"Saved vertical profiles to {outdir / 'vertical_profiles.png'}")

def main():
    args = parse_args()
    
    # Load labels
    if not Path(args.labels_parquet).exists():
        print(f"Error: Labels file not found at {args.labels_parquet}")
        return
        
    print(f"Loading labels from {args.labels_parquet}...")
    labels_df = pd.read_parquet(args.labels_parquet)
    
    # Load cloud data
    if not Path(args.cloud_nc).exists():
        print(f"Error: Cloud data not found at {args.cloud_nc}")
        return
        
    print(f"Loading cloud data from {args.cloud_nc}...")
    ds = xr.open_dataset(args.cloud_nc)
    
    # Compute physics
    # We only need to compute for clouds that are in our labels file
    # The labels file has 'cloud_id' which corresponds to the track index
    cloud_ids = labels_df['cloud_id'].values
    
    physics_df = compute_cloud_physics(ds, cloud_ids)
    
    # Merge with labels
    full_df = pd.merge(labels_df, physics_df, on='cloud_id')
    
    outdir = Path(args.outdir)
    
    # Generate all plots
    print("\n=== Generating cluster physics plots ===")
    
    # 1. Original physics boxplots
    plot_cluster_physics(full_df, outdir)
    
    # 2. Lifespan comparison (using actual age variable)
    plot_lifespan_comparison(full_df, outdir)
    
    # 3. Merge and split comparison
    plot_merge_split_comparison(full_df, outdir)
    
    # 4. Radius comparison
    plot_radius_comparison(full_df, outdir)
    
    # 5. Vertical profiles (w, area, compactness)
    print("\nComputing vertical profiles...")
    profile_data = compute_vertical_profiles(ds, labels_df)
    plot_vertical_profiles(profile_data, outdir)
    
    # 6. Distribution analysis at selected heights
    print("\nPlotting distribution diagnostics...")
    plot_distribution_at_heights(profile_data, outdir, var_name='area_per_level')
    plot_distribution_at_heights(profile_data, outdir, var_name='w_per_level')
    
    ds.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
