import os
import numpy as np
import netCDF4 as nc
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import geospatialtools.gdal_tools as gdal_tools


# PATHS

outputdir = "/scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/output_plots"
os.makedirs(outputdir, exist_ok=True)

path = "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yrs_100hru_100bh_test/"

print("Simulation path:", path, flush=True)

# READ HRU AND CID MAPS

hrus_file = path + "postprocess/hrus.vrt"
cids_file = path + "postprocess/cids.vrt"

hrus = gdal_tools.read_raster(hrus_file).astype(np.int32)
cids = gdal_tools.read_raster(cids_file).astype(np.int32)
metad = gdal_tools.retrieve_metadata(hrus_file)

extent = [metad["minx"], metad["maxx"], metad["miny"], metad["maxy"]]

# VARIABLES
var2plot = ["dem", "slope","y_aspect", "x_aspect","svf", "tvf","hor_n", "hor_w"]
#var2plot = ["dem"]

cid_list = range(1, 1601)

# PLOT
# 

fig, axes = plt.subplots(4, 2, figsize=(11, 15))
axes = axes.flatten()

for j, ax in enumerate(axes):

    var = var2plot[j]
    print(f"\nPlotting {var}", flush=True)

    final_map = np.full(hrus.shape, -9999.0, dtype=np.float32)

    for cid in cid_list:

        if cid % 50 == 0:
            print(f"  Processing CID {cid}", flush=True)

        fp_path = path + f"{cid}/input_file.nc"

        if not os.path.exists(fp_path):
            print(f"Missing: {fp_path}", flush=True)
            continue

        fp = nc.Dataset(fp_path)
        grp = fp.groups["parameters"]

        if var not in grp.variables:
            print(f"{var} missing in CID {cid}", flush=True)
            fp.close()
            continue

        data = np.array(grp.variables[var][:], dtype=np.float32)
        fp.close()

        mask = (hrus != -9999) & (cids == cid)

        hru_ids = hrus[mask].astype(np.int32)
        valid = (hru_ids >= 0) & (hru_ids < len(data))

        rows, cols = np.where(mask)
        final_map[rows[valid], cols[valid]] = data[hru_ids[valid]]

    final_map[final_map == -9999.0] = np.nan

    im = ax.imshow(
        final_map,
        origin="upper",
        extent=extent,
        cmap="terrain",
        interpolation="nearest"
    )

    ax.set_title(var)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    fig.colorbar(im, ax=ax, shrink=0.85)

plt.tight_layout()

outfile = os.path.join(outputdir, "static_variables_HRU_mapped_100hru_full_domain.png")
plt.savefig(outfile, dpi=250, bbox_inches="tight")

print("Saved:", outfile, flush=True)

plt.show()
