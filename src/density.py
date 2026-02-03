"""
Compute reference density profile rho0(z) from raw LES fields.

Uses thermodynamic formula with inputs:
- l: liquid water mixing ratio (kg/kg, despite file label saying g/kg)
- q: total water mixing ratio (kg/kg, despite file label saying g/kg)
- p: pressure (Pa)
- t: liquid-water potential temperature theta_l (K)

Returns rho0(z) in kg/m^3.
"""

from __future__ import annotations
from pathlib import Path
import numpy as np
import xarray as xr


def compute_rho0_from_raw(
    base_path: str | Path,
    time_indices: np.ndarray | None = None,
    reduce: str = 'median',
    sample_frac: float = 1.0,
    sample_seed: int | None = None,
    time_max: int | None = None,
    sample_mode: str = 'stride',
) -> xr.DataArray:
    """
    Compute domain-mean reference density profile rho0(z) from raw LES fields.

    Parameters:
        base_path: Directory containing RICO raw NetCDF files (rico.l.nc, rico.q.nc, rico.p.nc, rico.t.nc)
        time_indices: Specific time indices to use
        reduce: 'median' or 'mean' for temporal aggregation
        sample_frac: Fraction of horizontal domain to sample (for speed)
        sample_seed: Random seed for sampling
        time_max: Maximum number of time steps to use
        sample_mode: 'stride' (faster) or 'random' sampling

    Returns:
        DataArray rho0[z] in kg/m^3
    
    Requires exact variable names: 'l', 'q', 'p', 't' and dimensions: 'time', 'z'
    """
    file_map = {
        'l': 'rico.l.nc',
        'q': 'rico.q.nc',
        'p': 'rico.p.nc',
        't': 'rico.t.nc',
    }

    ds_l = xr.open_dataset(f"{str(base_path).rstrip('/')}/{file_map['l']}")
    ds_q = xr.open_dataset(f"{str(base_path).rstrip('/')}/{file_map['q']}")
    ds_p = xr.open_dataset(f"{str(base_path).rstrip('/')}/{file_map['p']}")
    ds_t = xr.open_dataset(f"{str(base_path).rstrip('/')}/{file_map['t']}")

    # Extract required variables (exact names required)
    # Note: Despite metadata labels saying 'g/kg', RICO data is actually in kg/kg
    l_data = ds_l['l']
    q_data = ds_q['q']
    p = ds_p['p']
    theta_l = ds_t['t']

    # Rename RICO dimensions ('zt', 'yt', 'xt') to standard ('z', 'y', 'x')
    def _normalize_dims(da: xr.DataArray) -> xr.DataArray:
        """Rename RICO dimension names to standard names. Requires RICO format: ('time', 'zt', 'yt', 'xt')."""
        if da.dims != ('time', 'zt', 'yt', 'xt'):
            raise ValueError(f"Expected RICO dimensions ('time', 'zt', 'yt', 'xt'), got {da.dims}")
        return da.rename({'zt': 'z', 'yt': 'y', 'xt': 'x'})

    l_data = _normalize_dims(l_data)
    q_data = _normalize_dims(q_data)
    p = _normalize_dims(p)
    theta_l = _normalize_dims(theta_l)

    # Optionally limit time indices
    if time_indices is not None:
        l_data = l_data.isel(time=time_indices)
        q_data = q_data.isel(time=time_indices)
        p = p.isel(time=time_indices)
        theta_l = theta_l.isel(time=time_indices)
    elif time_max is not None and 'time' in l_data.dims:
        l_data = l_data.isel(time=slice(0, int(time_max)))
        q_data = q_data.isel(time=slice(0, int(time_max)))
        p = p.isel(time=slice(0, int(time_max)))
        theta_l = theta_l.isel(time=slice(0, int(time_max)))

    # Subsample horizontal domain for speed if requested
    if sample_frac < 1.0:
        ref = p
        spatial = [d for d in ref.dims if d not in ('time', 'z')]
        if spatial:
            per_dim_frac = sample_frac ** (1.0 / len(spatial))
            indexers = {}
            if sample_mode == 'random':
                rng = np.random.default_rng(sample_seed)
                for d in spatial:
                    n = ref.sizes[d]
                    k = max(1, int(round(per_dim_frac * n)))
                    idx = np.sort(rng.choice(n, size=k, replace=False))
                    indexers[d] = idx
            else:
                # Stride sampling (faster for large domains)
                for d in spatial:
                    n = ref.sizes[d]
                    step = max(1, int(round(1.0 / per_dim_frac)))
                    indexers[d] = slice(0, n, step)
            
            l_data = l_data.isel(**{d: indexers[d] for d in l_data.dims if d in indexers})
            q_data = q_data.isel(**{d: indexers[d] for d in q_data.dims if d in indexers})
            p = p.isel(**{d: indexers[d] for d in p.dims if d in indexers})
            theta_l = theta_l.isel(**{d: indexers[d] for d in theta_l.dims if d in indexers})

    # Data is already in kg/kg (no conversion needed)
    q_l = l_data.astype('float64')
    q_t = q_data.astype('float64')
    q_v = q_t - q_l

    # Physical constants
    R_d = 287.04      # Gas constant for dry air (J/kg/K)
    R_v = 461.5       # Gas constant for water vapor (J/kg/K)
    c_pd = 1005.0     # Specific heat of dry air (J/kg/K)
    c_pv = 1850.0     # Specific heat of water vapor (J/kg/K)
    L_v = 2.5e6       # Latent heat of vaporization (J/kg)
    p_0 = 100000.0    # Reference pressure (Pa)
    epsilon = 0.622   # Ratio of molecular weights (Md/Mv)
    rho_l = 1000.0    # Density of liquid water (kg/m^3)

    # Compute kappa (variable due to moisture)
    kappa = (R_d / c_pd) * ((1.0 + q_v / epsilon) / (1.0 + q_v * (c_pv / c_pd)))
    
    # Compute temperature from potential temperature
    T = theta_l * (c_pd / (c_pd - L_v * q_l)) * (p_0 / p) ** (-kappa)
    
    # Compute density components
    p_v = (q_v / (q_v + epsilon)) * p
    rho = (p - p_v) / (R_d * T) + (p_v / (R_v * T)) + (q_l * rho_l)

    # Reduce to vertical profile
    axes = [d for d in rho.dims if d not in ('time', 'z')]
    if len(axes) > 0:
        rho = rho.mean(dim=tuple(axes), skipna=True)
    
    if 'time' in rho.dims:
        if reduce == 'median':
            rho0 = rho.median(dim='time', skipna=True)
        elif reduce == 'mean':
            rho0 = rho.mean(dim='time', skipna=True)
        else:
            raise ValueError(f"reduce must be 'median' or 'mean', got '{reduce}'")
    else:
        rho0 = rho

    rho0 = rho0.rename('rho0').transpose('z')
    
    if not np.isfinite(rho0).any():
        raise ValueError("Computed rho0 has no finite values. Check raw files and units.")
    
    return rho0
