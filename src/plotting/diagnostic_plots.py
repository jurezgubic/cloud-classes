"""
Diagnostic plots for cloud classification pipeline.

Creates visualizations at each step to verify:
- Reference density profile
- Raw vs normalized vertical profiles
- PCA dimensionality reduction
- GMM clustering in feature space
- Final class templates
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import ListedColormap
import seaborn as sns


def plot_individual_cloud_profiles(clouds, z_vals, outdir, n_sample=5, random_seed=42):
    """Plot vertical profiles for a random sample of individual clouds.
    
    Shows both raw J(z) and normalized j(z) = J(z)/M for selected clouds.
    
    Args:
        clouds: List of xarray.Dataset with reduced cloud profiles
        z_vals: Height values [m]
        outdir: Output directory path
        n_sample: Number of clouds to plot
        random_seed: Random seed for reproducibility
    """
    outdir = Path(outdir) / 'individual_profiles'
    outdir.mkdir(parents=True, exist_ok=True)
    
    n_clouds = len(clouds)
    n_sample = min(n_sample, n_clouds)
    
    # Randomly sample clouds
    rng = np.random.default_rng(random_seed)
    indices = rng.choice(n_clouds, size=n_sample, replace=False)
    
    for idx in indices:
        c = clouds[idx]
        track_idx = c.attrs.get('track_index', idx)
        
        # Extract profiles
        J = c['J'].values
        M = np.nansum(J)
        j_norm = J / M if M > 0 else J
        
        S_a = c['S_a'].values
        S_aw = c['S_aw'].values
        J_rho = c['J_rho'].values
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 6))
        
        # Left: Raw profiles
        ax1.plot(J, z_vals, 'b-', linewidth=2, label=f'J(z), M={M:.2e} kg')
        ax1.plot(J_rho, z_vals, 'r--', linewidth=1.5, alpha=0.7, label='J_rho(z)')
        ax1.set_xlabel('Mass Flux [kg]', fontsize=11)
        ax1.set_ylabel('Height [m]', fontsize=11)
        ax1.set_title('Raw Profiles', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=9)
        ax1.grid(True, alpha=0.3)
        
        # Right: Normalized profile
        ax2.plot(j_norm, z_vals, 'g-', linewidth=2, label='j(z) = J(z)/M')
        ax2.set_xlabel('Normalized j(z) [1/m]', fontsize=11)
        ax2.set_ylabel('Height [m]', fontsize=11)
        ax2.set_title('Normalized Profile', fontsize=12, fontweight='bold')
        ax2.legend(fontsize=9)
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'Cloud Track {track_idx}', fontsize=13, fontweight='bold')
        plt.tight_layout()
        
        outfile = outdir / f'profile_track{track_idx}.png'
        plt.savefig(outfile, dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"  → Saved {n_sample} individual cloud profiles to: {outdir}")


def plot_density_profile(rho0, z_vals, outdir):
    """Plot reference density profile rho0(z).
    
    Args:
        rho0: xarray.DataArray of density [kg/m^3]
        z_vals: Height values [m]
        outdir: Output directory path
    """
    fig, ax = plt.subplots(figsize=(6, 8))
    
    ax.plot(rho0.values, z_vals, 'k-', linewidth=2, label='ρ₀(z)')
    ax.set_xlabel('Density [kg/m³]', fontsize=12)
    ax.set_ylabel('Height [m]', fontsize=12)
    ax.set_title('Reference Density Profile', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / '01_density_profile.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 01_density_profile.png")


def plot_raw_profiles(clouds, z_vals, outdir, n_sample=10):
    """Plot sample of raw (unnormalized) J(z) profiles.
    
    Shows that clouds have different total masses but similar shapes.
    
    Args:
        clouds: List of xarray.Dataset with 'J' variable
        z_vals: Height values [m]
        outdir: Output directory path
        n_sample: Number of clouds to show
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    
    # Sample clouds evenly
    n_clouds = len(clouds)
    indices = np.linspace(0, n_clouds - 1, min(n_sample, n_clouds), dtype=int)
    
    masses = []
    
    for i in indices:
        c = clouds[i]
        J = c['J'].values
        M = np.nansum(J)
        masses.append(M)
        
        # Left: raw profiles
        ax1.plot(J, z_vals, alpha=0.7, linewidth=2, 
                label=f'Cloud {i} (M={M:.1e} kg)')
        
        # Right: normalized profiles
        j_norm = J / M if M > 0 else J
        ax2.plot(j_norm, z_vals, alpha=0.7, linewidth=2)
    
    # Left plot
    ax1.set_xlabel('Mass Flux J(z) [kg]', fontsize=12)
    ax1.set_ylabel('Height [m]', fontsize=12)
    ax1.set_title('Raw Profiles: Different Magnitudes', fontsize=13, fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc='best')
    
    # Right plot
    ax2.set_xlabel('Normalized j(z) = J(z)/M [1/m]', fontsize=12)
    ax2.set_ylabel('Height [m]', fontsize=12)
    ax2.set_title('Normalized Profiles: Similar Shapes', fontsize=13, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    
    plt.suptitle(f'Mass range: {min(masses):.1e} - {max(masses):.1e} kg', 
                 fontsize=11, y=1.02)
    plt.tight_layout()
    plt.savefig(Path(outdir) / '02_raw_vs_normalized.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 02_raw_vs_normalized.png")


def plot_normalized_profiles(Jmat, z_vals, labels, outdir):
    """Plot all normalized profiles colored by assigned class.
    
    Args:
        Jmat: Array [n_clouds, n_levels] of normalized j(z)
        z_vals: Height values [m]
        labels: Array [n_clouds] of class assignments
        outdir: Output directory path
    """
    n_classes = len(np.unique(labels))
    colors = sns.color_palette('tab10', n_classes)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    for k in range(n_classes):
        mask = labels == k
        n_k = np.sum(mask)
        
        # Plot all members of this class
        for j_prof in Jmat[mask]:
            ax.plot(j_prof, z_vals, color=colors[k], alpha=0.3, linewidth=1)
        
        # Plot class mean as thick line
        mean_prof = Jmat[mask].mean(axis=0)
        ax.plot(mean_prof, z_vals, color=colors[k], linewidth=3, 
               label=f'Class {k} (n={n_k})')
    
    ax.set_xlabel('Normalized j(z) [1/m]', fontsize=12)
    ax.set_ylabel('Height [m]', fontsize=12)
    ax.set_title('All Normalized Profiles by Class', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    plt.savefig(Path(outdir) / '03_profiles_by_class.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 03_profiles_by_class.png")


def plot_pca_analysis(pca, pcs, z_vals, Jmean, outdir):
    """Plot PCA diagnostics: variance explained and PC loadings.
    
    Args:
        pca: Fitted sklearn PCA object
        pcs: Principal component scores [n_clouds, n_pcs]
        z_vals: Height values [m]
        Jmean: Mean profile subtracted before PCA
        outdir: Output directory path
    """
    n_pcs = pca.n_components_
    
    fig = plt.figure(figsize=(14, 5))
    gs = gridspec.GridSpec(1, 3, width_ratios=[1, 1.5, 1])
    
    # 1. Scree plot
    ax1 = fig.add_subplot(gs[0])
    var_ratio = pca.explained_variance_ratio_
    cum_var = np.cumsum(var_ratio)
    
    x_pos = np.arange(1, len(var_ratio) + 1)
    ax1.bar(x_pos, var_ratio * 100, alpha=0.7, color='steelblue', 
           label='Individual')
    ax1.plot(x_pos, cum_var * 100, 'ro-', linewidth=2, markersize=8,
            label='Cumulative')
    ax1.axhline(95, color='gray', linestyle='--', alpha=0.5, label='95%')
    ax1.set_xlabel('Principal Component', fontsize=11)
    ax1.set_ylabel('Variance Explained [%]', fontsize=11)
    ax1.set_title('PCA Variance', fontsize=12, fontweight='bold')
    ax1.set_xticks(x_pos)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=9)
    
    # 2. PC loadings (what each PC represents in vertical space)
    ax2 = fig.add_subplot(gs[1])
    
    # Plot mean profile first
    ax2.plot(Jmean, z_vals, 'k--', linewidth=2, alpha=0.5, label='Mean j(z)')
    
    # Plot each PC loading
    colors_pc = ['#d62728', '#2ca02c', '#ff7f0e']
    for i in range(min(n_pcs, 3)):
        loading = pca.components_[i, :]
        # Scale for visibility
        scaled = Jmean + loading * np.std(pcs[:, i]) * 2
        ax2.plot(scaled, z_vals, color=colors_pc[i], linewidth=2,
                label=f'PC{i+1} ({var_ratio[i]*100:.1f}%)')
    
    ax2.set_xlabel('Profile Value', fontsize=11)
    ax2.set_ylabel('Height [m]', fontsize=11)
    ax2.set_title('PC Loadings (±2σ from mean)', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=9)
    
    # 3. PC score distributions
    ax3 = fig.add_subplot(gs[2])
    
    for i in range(min(n_pcs, 3)):
        ax3.hist(pcs[:, i], bins=15, alpha=0.6, color=colors_pc[i],
                label=f'PC{i+1}', density=True)
    
    ax3.set_xlabel('PC Score', fontsize=11)
    ax3.set_ylabel('Density', fontsize=11)
    ax3.set_title('PC Distributions', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    ax3.legend(fontsize=9)
    
    plt.suptitle(f'PCA Analysis: {cum_var[n_pcs-1]*100:.1f}% variance in {n_pcs} components',
                fontsize=13, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(Path(outdir) / '04_pca_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 04_pca_analysis.png")


def plot_feature_space(pcs, logM, labels, outdir):
    """Plot feature space with GMM cluster assignments.
    
    Shows 2D projections of [PC1, PC2, PC3, log(M)] colored by class.
    
    Args:
        pcs: Principal component scores [n_clouds, n_pcs]
        logM: Log of total mass [n_clouds]
        labels: Class assignments [n_clouds]
        outdir: Output directory path
    """
    n_pcs = pcs.shape[1]
    n_classes = len(np.unique(labels))
    colors = sns.color_palette('tab10', n_classes)
    
    # Create all pairwise plots
    fig = plt.figure(figsize=(12, 10))
    
    # Features: PC1, PC2, PC3 (if available), logM
    features = []
    feature_names = []
    
    for i in range(min(n_pcs, 3)):
        features.append(pcs[:, i])
        feature_names.append(f'PC{i+1}')
    features.append(logM)
    feature_names.append('log(M)')
    
    n_features = len(features)
    
    plot_idx = 1
    for i in range(n_features):
        for j in range(i + 1, n_features):
            ax = fig.add_subplot(n_features - 1, n_features - 1, plot_idx)
            
            # Scatter plot colored by class
            for k in range(n_classes):
                mask = labels == k
                ax.scatter(features[i][mask], features[j][mask], 
                          c=[colors[k]], s=80, alpha=0.7, 
                          label=f'Class {k}' if plot_idx == 1 else None,
                          edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(feature_names[i], fontsize=10)
            ax.set_ylabel(feature_names[j], fontsize=10)
            ax.grid(True, alpha=0.3)
            
            if plot_idx == 1:
                ax.legend(fontsize=9, loc='best')
            
            plot_idx += 1
    
    plt.suptitle('GMM Clustering in Feature Space', fontsize=14, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig(Path(outdir) / '05_feature_space_gmm.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 05_feature_space_gmm.png")


def plot_class_templates(phi, phi_p10, phi_p90, z_vals, labels, outdir):
    """Plot final class templates with uncertainty bands.
    
    Args:
        phi: Mean profiles [n_classes, n_levels]
        phi_p10: 10th percentile [n_classes, n_levels]
        phi_p90: 90th percentile [n_classes, n_levels]
        z_vals: Height values [m]
        labels: Class assignments (for counting)
        outdir: Output directory path
    """
    n_classes = phi.shape[0]
    colors = sns.color_palette('tab10', n_classes)
    
    fig, axes = plt.subplots(1, n_classes, figsize=(5 * n_classes, 7), sharey=True)
    
    if n_classes == 1:
        axes = [axes]
    
    for k in range(n_classes):
        ax = axes[k]
        n_k = np.sum(labels == k)
        
        # Uncertainty band (10th to 90th percentile)
        ax.fill_betweenx(z_vals, phi_p10[k], phi_p90[k], 
                        color=colors[k], alpha=0.3, label='P10-P90 range')
        
        # Mean template
        ax.plot(phi[k], z_vals, color=colors[k], linewidth=3, 
               label=f'Mean φ_{k}(z)')
        
        ax.set_xlabel('Normalized Profile [1/m]', fontsize=12)
        if k == 0:
            ax.set_ylabel('Height [m]', fontsize=12)
        ax.set_title(f'Class {k}\n({n_k} clouds)', fontsize=13, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=10, loc='best')
    
    plt.suptitle('Cloud Class Templates φₖ(z)', fontsize=15, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(Path(outdir) / '06_class_templates.png', dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  → Saved: 06_class_templates.png")


def plot_all_diagnostics(rho0, z_vals, clouds, Jmat, Jmean, pca, pcs, 
                        logM, labels, phi, phi_p10, phi_p90, outdir, n_sample_clouds=5):
    """Convenience function to generate all diagnostic plots.
    
    Args:
        rho0: Reference density profile
        z_vals: Height values [m]
        clouds: List of cloud datasets
        Jmat: Normalized profile matrix [n_clouds, n_levels]
        Jmean: Mean profile
        pca: Fitted PCA object
        pcs: PC scores
        logM: Log masses
        labels: Class labels
        phi: Class templates
        phi_p10: Template 10th percentiles
        phi_p90: Template 90th percentiles
        outdir: Output directory
        n_sample_clouds: Number of individual clouds to plot
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    
    print("\n[Plotting Diagnostics]")
    
    plot_density_profile(rho0, z_vals, outdir)
    plot_raw_profiles(clouds, z_vals, outdir, n_sample=10)
    plot_normalized_profiles(Jmat, z_vals, labels, outdir)
    plot_pca_analysis(pca, pcs, z_vals, Jmean, outdir)
    plot_feature_space(pcs, logM, labels, outdir)
    plot_class_templates(phi, phi_p10, phi_p90, z_vals, labels, outdir)
    
    # Plot individual cloud profiles
    if n_sample_clouds > 0:
        plot_individual_cloud_profiles(clouds, z_vals, outdir, n_sample=n_sample_clouds)
    
    print(f"[Plotting Complete] All plots saved to: {outdir}\n")
