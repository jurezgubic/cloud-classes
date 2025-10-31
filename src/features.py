"""
Lifetime reductions from the tracking output (cloud_results.nc).

What I compute per cloud (by height z):
- S_a(z) = sum over t of A(z,t) * dt [m^2 s]  (time integrated in-cloud area)
- S_aw(z) [m^3]: upward volume transport integrated over time
  - preferred exact: sum over (y,x,t) of w_plus(z,y,x,t) * dx * dy * dt inside mask
  - fallback: use aggregates via mass flux per level
- T_c = max(age) * dt [s]  (cloud lifetime from age variable in seconds)
- tilde_a(z) = S_a/T_c [m^2]  (lifetime mean area)
- tilde_w_a(z) = S_aw/S_a [m s^-1]  (area weighted lifetime mean w_plus)
- J_rho(z) = sum over t of max(M(z,t), 0) * dt [kg]  (time integrated mass using instantaneous density)
- J(z) [kg] = rho0(z) * S_aw(z)

Inputs from cloud_results.nc (per track, time, level):
- area_per_level[track,time,level] = A(z,t) [m^2]
- w_per_level[track,time,level] = mean w(z,t) over in-cloud points [m s^-1]
- mass_flux_per_level[track,time,level] = M(z,t) = sum over cells of rho * w * dx * dy [kg s^-1]
- age[track,time] = cloud age at each timestep [timesteps]
- Optionally 4-D: mask[track,level,y,x,time], w[track,level,y,x,time]
- height[level] = z [m]
- valid_track[track] in {0,1}. I normally use only valid (complete lifetime) tracks.

Note on signs: I use only upward transport (w_plus or M_plus). Convective mass flux is an updraft thing.
"""

from __future__ import annotations
import numpy as np
import xarray as xr

def _z_coord(ds: xr.Dataset) -> xr.DataArray:
    """Return height coord as DataArray named 'z'."""
    if ds is None:
        print("[_z_coord] got None dataset. i stop.")
        return None
    if 'height' in ds:
        return xr.DataArray(ds['height'].values, dims=('z',), name='z')
    if 'level' in ds.dims:
        return xr.DataArray(np.arange(ds.sizes['level'], dtype=float), dims=('z',), name='z')
    raise ValueError("Expected 'height' variable or 'level' dimension in dataset.")

def _track_is_valid(ds: xr.Dataset, i: int) -> bool:
    v = ds.get('valid_track')
    if v is None:
        return True
    try:
        return bool(v[i].item() == 1)
    except Exception:
        return True

def reduce_track(ds: xr.Dataset, track_index: int, dt: float,
                 rho0: xr.DataArray | None = None,
                 positive_only: bool = True,
                 require_valid: bool = True,
                 eps: float = 0.0) -> xr.Dataset:
    """
    Reduce one tracked cloud (one row in 'track') to lifetime-mean profiles.

    Physics:
    - Area A(z,t) times w_plus(z,t) integrated over time gives an upward volume transport S_aw(z).
    - Multiplying S_aw by a reference density rho0(z) gives a time integrated mass J(z) [kg].
    - Using instantaneous density inside the cloud and summing M_plus(z,t) * dt gives J_rho(z) [kg].
    - Cloud lifetime T_c is extracted from max(age) * dt
    """
    if require_valid and not _track_is_valid(ds, track_index):
        raise ValueError("Track is flagged invalid (partial lifetime). Set require_valid=False to force.")

    z = _z_coord(ds)
    # say what I am doing (simple progress)
    print(f"[reduce_track] reducing track {track_index} ...", flush=True)
    # Extract per-level, per-time arrays for this track
    A = ds['area_per_level'].isel(track=track_index)
    W = ds['w_per_level'].isel(track=track_index)
    # Prefer updraft-only mass flux per level if available; else legacy total mass flux
    if 'mass_flux_updraft_per_level' in ds:
        M = ds['mass_flux_updraft_per_level'].isel(track=track_index)
    else:
        M = ds['mass_flux_per_level'].isel(track=track_index)
    # Ensure vertical dim is named 'z' for clarity/consistency
    if 'level' in A.dims:
        A = A.rename({'level':'z'})
        W = W.rename({'level':'z'})
        M = M.rename({'level':'z'})

    # Optional 4-D inputs at full resolution (preferred when available)
    mask4d = None
    w4d = None
    if 'mask' in ds:
        mask4d = ds['mask'].isel(track=track_index)
        if 'level' in mask4d.dims:
            mask4d = mask4d.rename({'level':'z'})
    if 'w' in ds:
        w4d = ds['w'].isel(track=track_index)
        if 'level' in w4d.dims:
            w4d = w4d.rename({'level':'z'})

    # Time indices when the cloud exists (any level has finite area)
    live_t = np.isfinite(A).any(dim='z') & (xr.where(np.isfinite(A), A, 0.0).sum(dim='z') > 0)
    if live_t.any():
        A = A.where(live_t, other=0.0)
        W = W.where(live_t)
        M = M.where(live_t, other=0.0)
        nt = int(live_t.sum().item())
    else:
        nt = 0
    print(f"[reduce_track] active time steps = {nt}", flush=True)
    
    # Extract cloud lifetime from age variable (in timesteps)
    # age[track, time] gives the age of the cloud at each timestep
    age = ds['age'].isel(track=track_index)
    # Maximum age gives the total lifetime in timesteps
    max_age = float(age.max().item())
    T_c = max_age * dt  # Convert to seconds
    print(f"[reduce_track] cloud lifetime from age: T_c = {T_c:.1f} s (max_age = {max_age} timesteps)", flush=True)

    # Use upward part only if requested
    # physics: convective mass flux is about updrafts, so keep w_plus and M_plus
    if positive_only:
        Wp = xr.where(W > 0.0, W, 0.0)
        Mp = xr.where(M > 0.0, M, 0.0)
    else:
        Wp = xr.where(np.isfinite(W), W, 0.0)
        Mp = xr.where(np.isfinite(M), M, 0.0)

    # Time integrated area and volume flux
    # physics: S_a = sum A * dt  (area time).
    S_a = (xr.where(np.isfinite(A), A, 0.0) * dt).sum(dim='time')          # [z] m^2 s
    # Preferred exact S_aw from 4-D if available; else fallback using aggregates (mass-flux based)
    S_aw = None
    if (mask4d is not None) and (w4d is not None):
        # need dx, dy for cell area
        dx = ds.attrs.get('dx', None)
        dy = ds.attrs.get('dy', None)
        if (dx is None) or (dy is None):
            hr = ds.attrs.get('horizontal_resolution', None)
            if hr is not None:
                dx = dy = float(hr)
        if (dx is None) or (dy is None):
            raise ValueError("4-D S_aw requires dx,dy in attrs (or horizontal_resolution)")
        w_plus_full = xr.where(w4d > 0.0, w4d, 0.0)
        S_aw = (w_plus_full.where(mask4d) * (float(dx) * float(dy)) * dt).sum(dim=('y','x','time'))  # [z] m^3
    else:
        # fallback using per-level aggregates; prefer mass-flux-based approximation
        # S_aw ~ (sum over time of M_plus * dt) / rho0(z)
        Mp = xr.where(M > 0.0, M, 0.0)
        rho0_z = None
        if rho0 is not None:
            if 'z' not in rho0.dims:
                rho0 = rho0.rename({rho0.dims[0]: 'z'})
            rho0_z = xr.DataArray(np.asarray(rho0.values, dtype=float), coords=dict(z=z.values), dims=('z',))
        if rho0_z is None:
            raise ValueError("fallback S_aw requires rho0. Provide rho0 or 4-D mask and w in the input dataset.")
        S_aw = (Mp * dt).sum(dim='time') / rho0_z

    # Lifetime means
    # physics: divide time integrals by lifetime (T_c already computed above from age variable)
    tilde_a = xr.where(T_c > 0, S_a / T_c, np.nan)                         # [level] m^2
    tilde_w_a = xr.where(S_a > eps, S_aw / S_a, np.nan)                    # [level] m s^-1

    # Time‑integrated mass flux using instantaneous rho (from M per level)
    J_rho = (Mp * dt).sum(dim='time')                                      # [z] kg

    # If a reference density profile is given, also compute J = rho0 * S_aw
    data_vars = dict(S_a=S_a, S_aw=S_aw, tilde_a=tilde_a, tilde_w_a=tilde_w_a,
                     J_rho=J_rho)
    if rho0 is not None:
        # Align rho0 length and coordinate to current z-grid
        if 'z' not in rho0.dims:
            rho0 = rho0.rename({rho0.dims[0]: 'z'})
        if rho0.sizes.get('z', None) != z.sizes['z']:
            raise ValueError("rho0 length does not match number of levels")
        rho0_z = xr.DataArray(np.asarray(rho0.values, dtype=float), coords=dict(z=z.values), dims=('z',))
        data_vars['J'] = (rho0_z * S_aw).rename('J')                       # [level] kg

    out = xr.Dataset(
        data_vars=data_vars,
        coords=dict(z=z),
        attrs=dict(T_c=T_c, dt=float(dt), track_index=int(track_index))
    )
    # Effective radius from lifetime‑mean area (useful geometric proxy)
    out['R_eff_tilde'] = xr.where(out['tilde_a'] > 0, np.sqrt(out['tilde_a'] / np.pi), np.nan)
    print(f"[reduce_track] done track {track_index}", flush=True)
    return out

def reduce_all_tracks(ds: xr.Dataset, dt: float,
                      rho0: xr.DataArray | None = None,
                      only_valid: bool = True,
                      min_timesteps: int = 1,
                      positive_only: bool = True) -> list[xr.Dataset]:
    """Convenience: reduce all tracks in a cloud_results.nc Dataset.

    - Skips tracks with less than `min_timesteps` active entries.
    - If `only_valid`, uses only complete lifetime tracks (valid_track==1), in so skipping tainted tracks.
    """
    ntracks = ds.sizes.get('track', 0)
    out = []
    print(f"[reduce_all_tracks] start: ntracks={ntracks}, only_valid={only_valid}, min_timesteps={min_timesteps}", flush=True)

    # Fast preselection using cheap per-track signals if available
    candidates = None
    if 'size' in ds:
        s = ds['size']  # [track,time]
        live_counts = (xr.where(np.isfinite(s), s, 0.0) > 0).sum(dim='time')  # count active times
        cand_mask = live_counts >= min_timesteps
        if only_valid and 'valid_track' in ds:
            valid_mask = xr.DataArray((ds['valid_track'].values == 1), dims=('track',))
            cand_mask = cand_mask & valid_mask
        candidates = np.nonzero(cand_mask.values)[0]
        print(f"[reduce_all_tracks] candidate tracks (by size): {len(candidates)}", flush=True)

    if candidates is None:
        # Fallback: scan all tracks, but still print heartbeat
        step = max(1, ntracks // 20)
        for i in range(ntracks):
            if only_valid and not _track_is_valid(ds, i):
                if (i + 1) % step == 0:
                    print(f"[reduce_all_tracks] scanned {i+1}/{ntracks}, reduced={len(out)}", flush=True)
                continue
            A = ds['area_per_level'].isel(track=i)
            if 'level' in A.dims:
                A = A.rename({'level':'z'})
            live_t = np.isfinite(A).any(dim='z') & (xr.where(np.isfinite(A), A, 0.0).sum(dim='z') > 0)
            if int(live_t.sum().item()) < min_timesteps:
                if (i + 1) % step == 0:
                    print(f"[reduce_all_tracks] scanned {i+1}/{ntracks}, reduced={len(out)}", flush=True)
                continue
            out.append(reduce_track(ds, i, dt, rho0=rho0, positive_only=positive_only, require_valid=only_valid))
            if (i + 1) % step == 0:
                print(f"[reduce_all_tracks] scanned {i+1}/{ntracks}, reduced={len(out)}", flush=True)
    else:
        # Iterate only candidates
        n_c = len(candidates)
        if n_c == 0:
            print("[reduce_all_tracks] no candidates found.", flush=True)
            return out
        step = max(1, n_c // 20)
        for j, i in enumerate(candidates):
            out.append(reduce_track(ds, int(i), dt, rho0=rho0, positive_only=positive_only, require_valid=only_valid))
            if ((j + 1) % step) == 0:
                print(f"[reduce_all_tracks] processed candidates: {j+1}/{n_c}, reduced={len(out)}", flush=True)

    print(f"[reduce_all_tracks] done: reduced_clouds={len(out)}", flush=True)
    return out