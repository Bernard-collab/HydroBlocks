#!/usr/bin/env python3

import os
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin

# -------------------------------------------------
# USER PATHS
# -------------------------------------------------
workspace = "/scratch/alpine/battobrah@xsede.org/snow_experiment/data/cids/1"
met_dir   = workspace   # where precip.nc, tair.nc, etc. already exist
outputdir = workspace   # write tif files here

# variable name inside each netcdf
nc_map = {
    "precip": "precip",
    "tair":   "tair",
    "lwdown": "lwdown",
    "swdown": "swdown",
    "psurf":  "psurf",
    "spfh":   "spfh",
    "wind":   "wind",
}

# output tif names
out_map = {
    "precip": "precip_latlon_coarse.tif",
    "tair":   "tair_latlon_coarse.tif",
    "lwdown": "lwdown_latlon_coarse.tif",
    "swdown": "swdown_latlon_coarse.tif",
    "psurf":  "psurf_latlon_coarse.tif",
    "spfh":   "spfh_latlon_coarse.tif",
    "wind":   "wind_latlon_coarse.tif",
}

# -------------------------------------------------
# HELPERS
# -------------------------------------------------
def find_lat_lon_names(ds):
    lat_name = None
    lon_name = None

    for cand in ["lat", "latitude", "y"]:
        if cand in ds.coords or cand in ds.dims:
            lat_name = cand
            break

    for cand in ["lon", "longitude", "x"]:
        if cand in ds.coords or cand in ds.dims:
            lon_name = cand
            break

    if lat_name is None or lon_name is None:
        raise ValueError(
            f"Could not find lat/lon names in dataset. "
            f"coords={list(ds.coords)}, dims={list(ds.dims)}"
        )

    return lat_name, lon_name


def find_time_name(ds):
    for cand in ["time", "Time", "datetime"]:
        if cand in ds.coords or cand in ds.dims:
            return cand
    return None


def build_transform(lon, lat):
    lon = np.asarray(lon)
    lat = np.asarray(lat)

    if lon.ndim != 1 or lat.ndim != 1:
        raise ValueError("This script expects 1D lon and lat coordinates.")

    xres = abs(lon[1] - lon[0]) if len(lon) > 1 else 0.125
    yres = abs(lat[1] - lat[0]) if len(lat) > 1 else 0.125

    west  = lon.min() - xres / 2.0
    north = lat.max() + yres / 2.0

    return from_origin(west, north, xres, yres)


def write_tif(arr, lon, lat, out_fp, nodata=-9999.0):
    arr = np.asarray(arr, dtype=np.float32)

    # raster row 0 must be north, so flip if lat is ascending
    if lat[0] < lat[-1]:
        arr = np.flipud(arr)
        lat_for_transform = lat[::-1]
    else:
        lat_for_transform = lat

    arr[~np.isfinite(arr)] = nodata
    transform = build_transform(lon, lat_for_transform)

    profile = {
        "driver": "GTiff",
        "height": arr.shape[0],
        "width": arr.shape[1],
        "count": 1,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": transform,
        "nodata": nodata,
        "compress": "DEFLATE"
    }

    with rasterio.open(out_fp, "w", **profile) as dst:
        dst.write(arr, 1)


def reduce_to_one_grid(da):
    """
    Convert a possibly time-varying DataArray to one 2D grid.
    Default behavior = mean over time.
    """
    dims = list(da.dims)

    # find lat/lon dims
    lat_dim = None
    lon_dim = None
    for d in dims:
        if d.lower() in ["lat", "latitude", "y"]:
            lat_dim = d
        if d.lower() in ["lon", "longitude", "x"]:
            lon_dim = d

    if lat_dim is None or lon_dim is None:
        raise ValueError(f"Could not identify lat/lon dims in {dims}")

    other_dims = [d for d in dims if d not in [lat_dim, lon_dim]]

    if len(other_dims) == 0:
        # already 2D
        return da

    # mean over all non-spatial dims, typically time
    return da.mean(dim=other_dims, skipna=True)


# -------------------------------------------------
# MAIN
# -------------------------------------------------
os.makedirs(outputdir, exist_ok=True)

for short_name, var_name in nc_map.items():
    nc_fp = os.path.join(met_dir, f"{short_name}.nc")

    if not os.path.exists(nc_fp):
        print(f"[skip] missing {nc_fp}")
        continue

    print(f"Processing {nc_fp}")

    ds = xr.open_dataset(nc_fp)

    # rename coords if needed
    lat_name, lon_name = find_lat_lon_names(ds)
    rename_dict = {}
    if lat_name != "lat":
        rename_dict[lat_name] = "lat"
    if lon_name != "lon":
        rename_dict[lon_name] = "lon"
    if rename_dict:
        ds = ds.rename(rename_dict)

    if var_name not in ds.variables:
        raise ValueError(
            f"Variable '{var_name}' not found in {nc_fp}. "
            f"Available vars: {list(ds.variables)}"
        )

    da = ds[var_name]

    # reduce to one 2D field
    da2 = reduce_to_one_grid(da)

    arr = da2.values
    lon = ds["lon"].values
    lat = ds["lat"].values

    out_fp = os.path.join(outputdir, out_map[short_name])
    write_tif(arr, lon, lat, out_fp)

    ds.close()
    print(f"  wrote {out_fp}")

print("Done.")