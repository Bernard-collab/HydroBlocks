import os
import gc
import numpy as np
import xarray as xr
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS, Transformer

workspace = "/scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/data/snow_experiment/data/cids/1"

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

# PROCESS ONE VARIABLE AT A TIME
TARGET_VARS = ["precip"]   # change to ["tair"], ["lwdown"], etc.
vars_order = TARGET_VARS

mask_fine = os.path.join(workspace, "mask_latlon.tif")
if not os.path.exists(mask_fine):
    raise FileNotFoundError(mask_fine)

with rasterio.open(mask_fine) as src:
    fine_profile = src.profile.copy()
    fine_transform = src.transform
    fine_crs = src.crs
    fine_height = src.height
    fine_width = src.width
    fine_bounds = src.bounds
    fine_mask = src.read(1)

# derive coarse grid from one nc
sample_nc = os.path.join(workspace, "precip.nc")
ds0 = xr.open_dataset(sample_nc, decode_times=False)
latc = ds0["lat"].values
lonc = ds0["lon"].values
ds0.close()

# build coarse transform assuming regular lon/lat centers
resx = float(np.abs(lonc[1] - lonc[0]))
resy = float(np.abs(latc[1] - latc[0]))
west = float(lonc.min() - resx / 2.0)
north = float(latc.max() + resy / 2.0)
coarse_transform = from_origin(west, north, resx, resy)

coarse_crs = CRS.from_epsg(4326)
fine_crs_obj = CRS.from_user_input(fine_crs)
transformer = Transformer.from_crs(fine_crs_obj, coarse_crs, always_xy=True)

def write_multiband_tif(path, data, profile, transform, crs, nodata=-9999.0):
    prof = profile.copy()
    prof.update(
        driver="GTiff",
        dtype="float32",
        count=data.shape[0],
        compress="lzw",
        nodata=nodata,
        transform=transform,
        crs=crs,
        width=data.shape[2],
        height=data.shape[1]
    )
    with rasterio.open(path, "w", **prof) as dst:
        dst.write(data.astype(np.float32))

# precompute fine-grid lon/lat once
rows = np.arange(fine_height)
cols = np.arange(fine_width)
cc, rr = np.meshgrid(cols, rows)
xs_f, ys_f = rasterio.transform.xy(fine_transform, rr, cc, offset="center")
xs_f = np.asarray(xs_f)
ys_f = np.asarray(ys_f)
lon_f, lat_f = transformer.transform(xs_f, ys_f)

nodata_mask = fine_profile.get("nodata", -9999.0)

for short in vars_order:
    print(f"\nProcessing variable: {short}")

    ncfile = os.path.join(workspace, f"{short}.nc")
    if not os.path.exists(ncfile):
        raise FileNotFoundError(ncfile)

    ds = xr.open_dataset(ncfile, decode_times=False)
    varname = nc_map[short]
    da = ds[varname]

    # netcdf dims expected: (t, lat, lon)
    arr = da.values.astype(np.float32)
    nt, nlat, nlon = arr.shape

    # write coarse multiband tif directly on native nc grid
    coarse_tif = os.path.join(workspace, f"{short}_latlon_coarse.tif")
    coarse_profile = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": nt,
        "height": nlat,
        "width": nlon,
    }
    write_multiband_tif(coarse_tif, arr, coarse_profile, coarse_transform, coarse_crs)
    print(f"Wrote coarse tif: {coarse_tif}")

    # write fine tif timestep-by-timestep (memory safe)
    fine_tif = os.path.join(workspace, f"{short}_latlon_fine.tif")
    profile = fine_profile.copy()
    profile.update(
        driver="GTiff",
        dtype="float32",
        count=nt,
        compress="lzw",
        nodata=-9999.0,
        transform=fine_transform,
        crs=fine_crs,
        width=fine_width,
        height=fine_height
    )

    with rasterio.open(fine_tif, "w", **profile) as dst:
        for i in range(nt):
            dai = da.isel(t=i)

            interp2d = dai.interp(
                lon=(("y", "x"), lon_f),
                lat=(("y", "x"), lat_f),
                method="linear"
            ).values.astype(np.float32)

            if np.isnan(interp2d).any():
                interp2d_nn = dai.interp(
                    lon=(("y", "x"), lon_f),
                    lat=(("y", "x"), lat_f),
                    method="nearest"
                ).values.astype(np.float32)
                interp2d = np.where(np.isnan(interp2d), interp2d_nn, interp2d)
                del interp2d_nn

            interp2d = np.where(
                np.isfinite(fine_mask) & (fine_mask != nodata_mask),
                interp2d,
                -9999.0
            )

            dst.write(interp2d, i + 1)

            if (i + 1) % 500 == 0 or i == nt - 1:
                print(f"{short}: wrote {i+1}/{nt} timesteps")

            del interp2d
            gc.collect()

    print(f"Wrote fine tif: {fine_tif}")

    ds.close()
    del da, arr, ds
    gc.collect()

    print(f"Done: {short}")

print("Selected meteorology TIFFs created.")