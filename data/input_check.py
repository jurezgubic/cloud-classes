import xarray as xr
import numpy as np
from pathlib import Path

nc_path = Path("../../cloud_results.nc")

def main():
    if not nc_path.exists():
        print("file not found:", nc_path)
        return

    ds = xr.open_dataset(nc_path)

    # dims
    n_tracks    = int(ds.sizes.get("track", 0))
    n_timesteps = int(ds.sizes.get("time", 0))
    n_levels    = int(ds.sizes.get("level", 0))
    n_rings     = int(ds.sizes.get("ring", 0))
    n_coords    = int(ds.sizes.get("coordinate", 0))

    print("opened:", nc_path)
    print("dims:")
    print("  tracks:", n_tracks)
    print("  timesteps:", n_timesteps)
    print("  levels:", n_levels)
    print("  rings:", n_rings)
    print("  coordinate:", n_coords)

    # basic variables (shape only so no loading of large arrays)
    core_vars = [
        "valid_track",
        "track_id",
        "height",
        "mass_flux",
        "mass_flux_per_level",
        "env_mass_flux_per_level",
        "area_per_level",
        "w_per_level",
        "nip_per_level",
        "merges_count",
        "splits_count",
        "age",
    ]
    print("\nvariables (shape):")
    for v in core_vars:
        if v in ds:
            print(f"  {v}: {tuple(ds[v].shape)}")
        else:
            print(f"  {v}: not present")

    # number of valid tracks (check tracks with actual data, not just flag since some tracks may be all NaN or 0 or 1)
    n_valid_tracks = None
    if "mass_flux" in ds:
        # Using mass_flux to determine which tracks actually have data
        mf = ds["mass_flux"].values  # (track, time)
        # Count tracks that have at least one value
        has_data = np.any(np.isfinite(mf) & (mf != 0), axis=1)
        n_valid_tracks = int(np.sum(has_data))


    # first few track_ids
    first_track_ids = None
    if "track_id" in ds:
        first_track_ids = ds["track_id"].values[:5]

    # height range (levels)
    height_min = None
    height_max = None
    if "height" in ds:
        height_min = float(ds["height"].min().values)
        height_max = float(ds["height"].max().values)

    # number of tracks that ever have finite mass_flux over time
    n_tracks_with_massflux = None
    if "mass_flux" in ds:
        # shape (track, time): this is ~ tracks * timesteps; modest to load
        mf = ds["mass_flux"].values
        track_has_any = np.any(np.isfinite(mf), axis=1) if mf.size > 0 else np.array([])
        n_tracks_with_massflux = int(np.sum(track_has_any))

    # merges / splits total counts (all tracks and times)
    total_merges = None
    total_splits = None
    if "merges_count" in ds:
        total_merges = int(np.sum(ds["merges_count"].values > 0))
    if "splits_count" in ds:
        total_splits = int(np.sum(ds["splits_count"].values > 0))

    # print summary
    print("\nbasic overview:")
    print("  valid tracks:", n_valid_tracks)
    print("  tracks with any mass_flux:", n_tracks_with_massflux)
    print("  height min/max (level coordinate):",(height_min, height_max))
    print("  total merge events (merges_count>0):", total_merges)
    print("  total split events (splits_count>0):", total_splits )

    if first_track_ids is not None:
        print("  first track_ids:", first_track_ids)

if __name__ == "__main__":
    main()
