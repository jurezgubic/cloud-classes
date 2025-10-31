# Cloud Classes

Learn cloud lifetime classes from tracked LES data to inform mass-flux parameterizations.

## What it does

Takes cloud tracking output (from CloudTracker) and raw LES fields (RICO), then:

1. **Computes reference density** \rho_0(z) from thermodynamic variables
2. **Reduces cloud lifetimes** to vertical mass flux profiles J(z)
3. **Normalizes by total mass** to get vertical shapes j(z) = J(z)/M
4. **Performs PCA** to capture dominant vertical structure modes (for visualization/analysis)
5. **Clusters clouds** using one of two methods:
   - **GMM**: Gaussian Mixture Model on PCA features + log(M)
   - **Wasserstein**: k-means with Wasserstein distance on j(z) using optimal transport
6. **Outputs class templates** \phi_k(z) for each cloud type

**Next step:** Map these learned templates to bulk variables (cloud base, CAPE, etc.) that a parameterization can predict.

## Installation

```bash
python -m venv classes_env
source classes_env/bin/activate  # or activate.fish for fish shell
pip install -r requirements.txt
```

## Usage

**Note:** Update paths in `make_classes.py` to point to your data before running.

```bash
cd src
python make_classes.py --clustering-method gmm --min-timesteps=10
```

**Required arguments:**
- `--clustering-method {gmm,wasserstein}` - clustering method to use

**Outputs:**
- `artefacts/class_templates.nc` - vertical templates \phi_k(z) for each class
- `artefacts/cloud_labels.parquet` - per-cloud class labels and features
- `plots/gmm/` or `plots/wasserstein/` - diagnostic plots (method-specific):
  - Density profile, PCA analysis, clustering, class templates
  - `individual_profiles/` - sample of 5 random cloud profiles (raw and normalized)

**Key options:**
- `--min-timesteps N` - minimum cloud lifetime (default: 3)
- `--n-classes K` - number of cloud classes (default: 3)
- `--n-sample-clouds N` - number of individual clouds to plot (default: 5)
- `--no-plots` - skip diagnostic plotting

## Clustering Methods

### GMM (Gaussian Mixture Model)
Clusters clouds in PCA feature space [PC1, PC2, PC3, log(M)]. This combines vertical shape information from PCA with cloud size (total mass M). Uses probabilistic soft clustering with full covariance.

### Wasserstein
k-means clustering using Wasserstein distance (optimal transport) on normalized profiles j(z). This measures the "shape distance" between vertical profiles, treating them as probability distributions over height. Insensitive to total cloud mass - focuses purely on vertical structure.

**Physics:** Wasserstein distance measures the minimum "cost" of transforming one vertical mass distribution into another, where cost is the amount of mass times the distance it must be moved.

**Implementation:**
- Uses k-means++ initialization for robustness
- Convergence when cluster assignments stabilize  
- Centroids computed as true Wasserstein barycenters using Sinkhorn algorithm
- Maximum 100 iterations with early stopping

**Example comparison:**
```bash
# Compare clustering methods
python make_classes.py --clustering-method gmm --min-timesteps 10
python make_classes.py --clustering-method wasserstein --min-timesteps 10
```

## Physics

Convective mass flux M(z) = \rho_0(z) * area(z) * w(z)

We integrate over cloud lifetimes to get total upward mass transport J(z). The shape j(z) = J(z)/M is independent of cloud size. Clouds with similar j(z) belong to the same class (shallow, congestus, deep).

