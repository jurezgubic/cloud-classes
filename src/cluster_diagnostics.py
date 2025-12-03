"""
Post-processing diagnostics for cloud clusters.

This script loads the cluster labels and the original cloud tracking data
to compute physical statistics for each cluster, such as:
- Cloud lifetime
- Cloud base/top height
- Mean/Max vertical velocity
- Cloud size/volume

It generates boxplots and summary tables to help interpret the physical
meaning of each cluster.
"""

import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import seaborn as sns

def parse_args():
    # Determine project root (parent of src/) to set robust default paths
    project_root = Path(__file__).resolve().parent.parent
    
    p = argparse.ArgumentParser("Cluster physical diagnostics")
    p.add_argument("--cloud-nc", default="/Users/jure/PhD/coding/tracking/cloud_results.nc", help="path to cloud_results.nc")
    p.add_argument("--labels-parquet", default=str(project_root / "artefacts/cloud_labels.parquet"), help="path to cloud_labels.parquet")
    p.add_argument("--outdir", default=str(project_root / "plots/cluster_physics"), help="output directory for plots")
    return p.parse_args()

def compute_cloud_physics(ds, cloud_ids):
    """
    Compute physical properties for a list of cloud IDs.
    
    Args:
        ds: xarray.Dataset with cloud tracking data
        cloud_ids: List of cloud indices to process
        
    Returns:
        DataFrame with physical properties for each cloud
    """
    stats = []
    
    # Get height coordinate
    z = ds['height'].values
    dz = np.gradient(z)
    
    print(f"Computing physics for {len(cloud_ids)} clouds...")
    
    for i, idx in enumerate(cloud_ids):
        if i % 100 == 0:
            print(f"  Processing {i}/{len(cloud_ids)}", end='\r')
            
        # Extract single cloud data
        # We need to handle the fact that cloud_ids might be indices into the original file
        # or a subset. Assuming cloud_ids matches 'track' dimension index.
        
        # Check if track is valid
        if not ds['valid_track'][idx]:
            continue
            
        # Get time-height arrays
        area = ds['area_per_level'].isel(track=idx)
        w = ds['w_per_level'].isel(track=idx)
        
        # Mask for active cloud points
        is_cloud = np.isfinite(area) & (area > 0)
        
        if not is_cloud.any():
            continue
            
        # 1. Lifetime (already in labels, but good to double check or get from age)
        age = ds['age'].isel(track=idx)
        lifetime_steps = float(age.max())
        dt = 60.0 # Assuming 60s timestep, should ideally read from attrs
        lifetime_s = lifetime_steps * dt
        
        # 2. Cloud Base and Top (lifetime statistics)
        # For each timestep, find min/max height
        cloud_bases = []
        cloud_tops = []
        
        # Iterate over time steps where cloud exists
        times = np.where(is_cloud.any(dim='level' if 'level' in is_cloud.dims else 'height'))[0]
        
        for t in times:
            # Get profile at this time
            if 'level' in is_cloud.dims:
                profile = is_cloud.isel(time=t).values
            else:
                profile = is_cloud.isel(time=t).values
                
            z_cloud = z[profile]
            if len(z_cloud) > 0:
                cloud_bases.append(z_cloud.min())
                cloud_tops.append(z_cloud.max())
        
        mean_base = np.mean(cloud_bases) if cloud_bases else np.nan
        mean_top = np.mean(cloud_tops) if cloud_tops else np.nan
        max_top = np.max(cloud_tops) if cloud_tops else np.nan
        
        # 3. Vertical Velocity
        # Mean updraft velocity (conditional on being in cloud)
        w_cloud = w.where(is_cloud)
        mean_w = float(w_cloud.mean())
        max_w = float(w_cloud.max())
        
        # 4. Cloud Size (max area)
        max_area = float(area.max())
        mean_area = float(area.where(is_cloud).mean())
        
        stats.append({
            'cloud_id': int(idx),
            'lifetime_s': lifetime_s,
            'mean_base_m': mean_base,
            'mean_top_m': mean_top,
            'max_top_m': max_top,
            'depth_m': mean_top - mean_base,
            'mean_w_ms': mean_w,
            'max_w_ms': max_w,
            'mean_area_m2': mean_area,
            'max_area_m2': max_area
        })
        
    return pd.DataFrame(stats)

def plot_cluster_physics(df, outdir):
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
        sns.boxplot(x='class_k', y=col, data=df, ax=ax, palette='tab10', showfliers=False)
        
        # Add mean markers
        sns.pointplot(x='class_k', y=col, data=df, ax=ax, 
                      estimator=np.mean, color='black', markers='D', scale=0.7, join=False)
        
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
    
    # Plot
    plot_cluster_physics(full_df, args.outdir)
    
    print("Done.")

if __name__ == "__main__":
    main()
