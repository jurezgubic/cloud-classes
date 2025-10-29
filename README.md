# Cloud Classes

Learn cloud lifetime classes from tracked LES data to inform mass-flux parameterizations.

## What it does

Takes cloud tracking output (from CloudTracker) and raw LES fields (RICO), then:

1. **Computes reference density** \rho_0(z) from thermodynamic variables
2. **Reduces cloud lifetimes** to vertical mass flux profiles J(z)
3. **Normalizes by total mass** to get vertical shapes j(z) = J(z)/M
4. **Performs PCA** to capture dominant vertical structure modes
5. **Clusters with GMM** to identify distinct cloud classes
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
python make_classes.py --min-timesteps=10
```

**Outputs:**
- `artefacts/class_templates.nc` - vertical templates \phi_k(z) for each class
- `artefacts/cloud_labels.parquet` - per-cloud class labels and features
- `plots/` - diagnostic plots:
  - Density profile, PCA analysis, clustering, class templates
  - `individual_profiles/` - sample of 5 random cloud profiles (raw and normalized)

**Key options:**
- `--min-timesteps N` - minimum cloud lifetime (default: 3)
- `--n-classes K` - number of cloud classes (default: 3)
- `--n-sample-clouds N` - number of individual clouds to plot (default: 5)
- `--no-plots` - skip diagnostic plotting

## Physics

Convective mass flux M(z) = \rho_0(z) * area(z) * w(z)

We integrate over cloud lifetimes to get total upward mass transport J(z). The shape j(z) = J(z)/M is independent of cloud size. Clouds with similar j(z) belong to the same class (shallow, congestus, deep).

