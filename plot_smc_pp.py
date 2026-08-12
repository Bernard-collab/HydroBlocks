import os
import sys
import gc
import time
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(
    "/projects/battobrah@xsede.org/software/anaconda/envs/geo_preproc/lib/python3.11/site-packages"
)

import geospatialtools.gdal_tools as gdal_tools

# PATHS

script_start = time.time()

outputdir = "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/output_plots"
os.makedirs(outputdir, exist_ok=True)

path = "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yrs_100hru_100bh_test/"
print(path, flush=True)

# SETTINGS

cid_list = range(1, 1601)
n_cids = len(cid_list)

buffer = 0

seasons = ["Winter", "Spring", "Summer", "Fall"]
season_months = [(12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11)]

print("=" * 70, flush=True)
print("Starting PP seasonal surface soil moisture mapping", flush=True)
print(f"Experiment path: {path}", flush=True)
print(f"Output directory: {outputdir}", flush=True)
print(f"Number of CIDs: {n_cids}", flush=True)
print("=" * 70, flush=True)

# READ STATIC MAPS

print("Reading HRU and CID maps...", flush=True)

t0 = time.time()
hrus = gdal_tools.read_raster(path + "postprocess/hrus.vrt")
print(f"HRU read in {(time.time() - t0) / 60:.2f} min", flush=True)

t0 = time.time()
cids = gdal_tools.read_raster(path + "postprocess/cids.vrt")
print(f"CID read in {(time.time() - t0) / 60:.2f} min", flush=True)

metad = gdal_tools.retrieve_metadata(path + "postprocess/hrus.vrt")

coord_xx0 = np.linspace(metad["minx"], metad["maxx"], metad["nx"])
coord_yy0 = np.linspace(metad["miny"], metad["maxy"], metad["ny"])

coords_x = coord_xx0[buffer:len(coord_xx0) - buffer + 1]
coords_y = coord_yy0[buffer:len(coord_yy0) - buffer + 1]

print("Finished reading static maps.", flush=True)
print(f"HRU shape: {hrus.shape}", flush=True)
print(f"CID shape: {cids.shape}", flush=True)

# READ TIME

print("Reading time variable...", flush=True)

fp = nc.Dataset(path + "1/input_file.nc")
time_var = fp.groups["meteorology"]["time"][:]
fp.close()

start = pd.Timestamp("2014-01-01 00:00")
datetime_series = start + pd.to_timedelta(time_var, unit="h")
datetime_series = datetime_series.tz_localize("UTC").tz_convert("America/Denver")

times = pd.Series(datetime_series, name="datetime")

print("Finished reading time.", flush=True)
print(f"Number of time steps: {len(times)}", flush=True)
print(f"Start time: {times.iloc[0]}", flush=True)
print(f"End time: {times.iloc[-1]}", flush=True)

# FUNCTION: READ TOP SMC

def get_top_smc(ncfile):
    with nc.Dataset(ncfile) as ds:
        smc_var = ds.groups["data"].variables["smc"]
        dims = smc_var.dimensions
        smc = np.array(smc_var[:], dtype=np.float32)

    if "soil" in dims:
        soil_axis = dims.index("soil")
    elif "nsoil" in dims:
        soil_axis = dims.index("nsoil")
    elif "soil_layers" in dims:
        soil_axis = dims.index("soil_layers")
    else:
        soil_axis = int(np.argmin(smc.shape[1:]) + 1)

    if soil_axis == 2:
        data = smc[:, :, 0]
    elif soil_axis == 1:
        data = smc[:, 0, :]
    else:
        raise ValueError(f"Unexpected smc dimensions: {dims}, shape={smc.shape}")

    return data

# FUNCTION: PLACE HRU VALUES INTO DOMAIN MAP
# ============================================================

def place_data_in_map(out_map, model_values, hrus, cids, cid):
    nvals = len(model_values)

    for h in range(1, nvals + 1):
        val = model_values[h - 1]

        if not np.isfinite(val):
            continue

        mask = (cids == cid) & (hrus == h)

        if np.any(mask):
            out_map[mask] = val

    return out_map

# FIRST PASS: COMPUTE GLOBAL MIN AND MAX

global_min = np.inf
global_max = -np.inf

print("=" * 70, flush=True)
print("FIRST PASS: Computing global SMC min/max", flush=True)
print("=" * 70, flush=True)

t0_global = time.time()

for cid in cid_list:
    if cid == 1:
        print("Beginning first pass...", flush=True)

    if cid % 50 == 0 or cid == n_cids:
        elapsed = (time.time() - t0_global) / 60
        print(
            f"First pass: CID {cid}/{n_cids} "
            f"({100 * cid / n_cids:.1f}%) "
            f"Elapsed: {elapsed:.1f} min",
            flush=True,
        )

    ncfile = path + f"output_data/{cid}/2014-01-01.nc"

    if not os.path.exists(ncfile):
        print(f"Missing file: {ncfile}", flush=True)
        continue

    try:
        data = get_top_smc(ncfile)

        for months in season_months:
            times_index = times.dt.month.isin(months).values

            data_sel = data[times_index, :]
            data_ave = np.nanmean(data_sel, axis=0)

            if np.any(np.isfinite(data_ave)):
                global_min = min(global_min, float(np.nanmin(data_ave)))
                global_max = max(global_max, float(np.nanmax(data_ave)))

        del data
        gc.collect()

    except Exception as e:
        print(f"ERROR in first pass for CID {cid}", flush=True)
        print(f"File: {ncfile}", flush=True)
        print(e, flush=True)
        continue

print("Finished first pass.", flush=True)
print(f"First pass runtime: {(time.time() - t0_global) / 60:.2f} min", flush=True)
print(f"Global SMC min: {global_min}", flush=True)
print(f"Global SMC max: {global_max}", flush=True)

# SECOND PASS: BUILD SEASONAL MAPS

print("=" * 70, flush=True)
print("SECOND PASS: Building seasonal maps", flush=True)
print("=" * 70, flush=True)

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

for iss, ax in enumerate(axes):
    my_season = seasons[iss]
    my_months = season_months[iss]

    season_start = time.time()

    print("", flush=True)
    print("=" * 70, flush=True)
    print(f"Building {my_season} map", flush=True)
    print("=" * 70, flush=True)

    final_map = np.full(hrus.shape, np.nan, dtype=np.float32)
    times_index = times.dt.month.isin(my_months).values

    for cid in cid_list:
        if cid % 50 == 0 or cid == n_cids:
            elapsed = (time.time() - season_start) / 60
            print(
                f"{my_season}: CID {cid}/{n_cids} "
                f"({100 * cid / n_cids:.1f}%) "
                f"Elapsed: {elapsed:.1f} min",
                flush=True,
            )

        ncfile = path + f"output_data/{cid}/2014-01-01.nc"

        if not os.path.exists(ncfile):
            print(f"Missing file: {ncfile}", flush=True)
            continue

        try:
            data = get_top_smc(ncfile)

            data_sel = data[times_index, :]
            data_ave = np.nanmean(data_sel, axis=0)

            final_map = place_data_in_map(
                final_map,
                data_ave,
                hrus,
                cids,
                cid
            )

            del data, data_sel, data_ave
            gc.collect()

        except Exception as e:
            print(f"ERROR while mapping CID {cid} for {my_season}", flush=True)
            print(f"File: {ncfile}", flush=True)
            print(e, flush=True)
            continue

    print(f"Finished mapping {my_season}", flush=True)
    print(f"{my_season} runtime: {(time.time() - season_start) / 60:.2f} min", flush=True)

    if buffer > 0:
        final_map = final_map[
            buffer:final_map.shape[0] - buffer + 1,
            buffer:final_map.shape[1] - buffer + 1
        ]

    print(f"Plotting {my_season}", flush=True)

    im = ax.pcolormesh(
        coords_x,
        coords_y,
        np.flipud(final_map),
        cmap="viridis",
        shading="auto",
        vmin=global_min,
        vmax=global_max
    )

    ax.set_title(f"PP Surface soil moisture (0–5 cm) - {my_season}")
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")

    fig.colorbar(im, ax=ax, label="SMC [m³/m³]")

print("Saving figure...", flush=True)

plt.tight_layout()

outfile = os.path.join(
    outputdir,
    "res_smc_surface_seasonal_PP_corrected_100hru_progress.png"
)

plt.savefig(outfile, dpi=300)
plt.close()

print("=" * 70, flush=True)
print("All seasonal plots completed successfully.", flush=True)
print(f"Saved: {outfile}", flush=True)
print(f"Total runtime: {(time.time() - script_start) / 60:.2f} min", flush=True)
print("=" * 70, flush=True)
