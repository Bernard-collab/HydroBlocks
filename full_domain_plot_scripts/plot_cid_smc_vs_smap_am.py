#!/usr/bin/env python3
"""Compare HydroBlocks surface SMC with NSIDC-0779 SMAP (6 AM), built from each CID.

Unlike the earlier version, this script does NOT use the merged postprocessed
grid (postprocess/output_dir). Everything is built from the individual CIDs:

  * HydroBlocks SMC   <edir>/output_data/<cid>/*.nc   (per-HRU time series)
  * HRU longitudes    <edir>/<cid>/input_file.nc      (parameters/lons)
  * HRU map           <edir>/<cid>/hru_mapping_latlon.tif
  * CID mask          <edir>/postprocess/cids/<cid>.tif  (optional; per-CID file)

Method
------
1. For every CID (in parallel), the 3-hourly surface-layer SMC of each HRU is
   linearly interpolated to 06:00 local solar time on each SMAP day, using the
   HRU's mean longitude (UTC = LST - lon/15).
2. Each 90-m HRU pixel is assigned to the SMAP 1-km EASE-Grid 2.0 cell that
   contains it. SMAP is then averaged over every HRU, weighting cells by the
   number of HRU pixels they contain. Only valid SMAP cells are averaged; an
   HRU is used on any day with at least some valid SMAP data (no coverage
   threshold, as in the original postprocessed comparison).
3. For every SMAP day (in parallel, by date chunks), valid HydroBlocks-SMAP
   pairs are accumulated per HRU and season.
4. Seasonal means per HRU are painted back onto a map (from each CID's own HRU
   raster) for plotting. Statistics are area-weighted over HRUs.

Outputs (two figures and one CSV):
  seasonal_06AM_HB_SMAP_maps_<years>.png
  seasonal_06AM_HB_minus_SMAP_histograms_<years>.png
  seasonal_06AM_HB_minus_SMAP_metrics_<years>.csv

Figures use matplotlib's default font (DejaVu Sans). Only the season headings
are bold. Sizes are chosen to stay readable (about 7-9 pt) when the map figure
is printed at full page width (~6.5 in).

Example
-------
python plot_cid_smc_vs_smap_am.py \
    --edir  /scratch/.../experiments/simulations/<experiment> \
    --smap-dir /scratch/.../SMAP_NSIDC0779 \
    --start-year 2016 --end-year 2023 --workers 16
"""

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timedelta

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject, transform as warp_transform
from rasterio.windows import Window


SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}

SMAP_BAND = 1           # descending overpass (~06:00 local solar time)
SOLAR_HOUR = 6.0        # local solar time, hours
PASS_DESCRIPTION = "6 AM descending"
NODATA = -9999
EPOCH = datetime(1970, 1, 1)

# ---------------------------------------------------------------------------
# Publication text sizes (matplotlib default font, DejaVu Sans).
# Only the season headings are bold.
# ---------------------------------------------------------------------------
FS_SEASON = 18      # bold season headings
FS_HEADER = 16      # column headers / histogram axis labels
FS_LABEL = 15       # axis and colorbar labels
FS_TICK = 13        # tick labels
FS_STATS = 13       # statistics box in histograms

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.weight": "normal",
    "axes.titleweight": "normal",
    "axes.labelweight": "normal",
    "font.size": FS_LABEL,
    "axes.labelsize": FS_LABEL,
    "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK,
})

WORKER = {}


# ---------------------------------------------------------------------------
# Arguments and file discovery
# ---------------------------------------------------------------------------
def arguments():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--edir", required=True, help="HydroBlocks experiment directory")
    p.add_argument("--smap-dir", required=True,
                   help="Directory containing daily NSIDC-0779 GeoTIFFs")
    p.add_argument("--start-year", type=int, default=2016)
    p.add_argument("--end-year", type=int, default=2023)
    p.add_argument("--variable", default="smc", help="HydroBlocks output variable")
    p.add_argument("--layer", type=int, default=0,
                   help="Soil layer index of the surface layer (default 0, 0-5 cm)")
    p.add_argument("--output-subdir", default="output_data",
                   help="Folder (inside --edir) holding per-CID output folders")
    p.add_argument("--dt-hours", type=float, default=3.0,
                   help="HydroBlocks output time step in hours (default 3)")
    p.add_argument("--first-offset-hours", type=float, default=0.0,
                   help="UTC hour of the first record in each output file "
                        "relative to the date in its file name (default 0)")
    p.add_argument("--cids", default=None,
                   help="Comma-separated CID subset (default: all CIDs)")
    p.add_argument("--plot-pixels", type=int, default=1600,
                   help="Map resolution for plotting (pixels along the longer side)")
    p.add_argument("--border", type=float, default=0.01,
                   help="White border around the map on all sides, as a fraction "
                        "of the domain size (default 0.01; 0 = none)")
    p.add_argument("--outdir", default=None)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--workers", type=int, default=1)
    return p.parse_args()


def discover_cids(edir, requested):
    if requested:
        return [int(v) for v in requested.split(",") if v.strip()]
    cids = sorted(int(d) for d in os.listdir(edir)
                  if d.isdigit() and os.path.isfile(os.path.join(edir, d, "input_file.nc")))
    if not cids:
        raise FileNotFoundError(f"No CID folders with input_file.nc in {edir}")
    return cids


def discover_smap_files(folder, start_year, end_year):
    pattern = re.compile(r"NSIDC-0779_EASE2_G1km_SMAP_SM_DS_(\d{8})(?:_\w+)?\.tif$")
    result = []
    for path in glob.glob(os.path.join(folder, "*.tif")):
        match = pattern.match(os.path.basename(path))
        if not match:
            continue
        date = datetime.strptime(match.group(1), "%Y%m%d")
        if start_year <= date.year <= end_year:
            result.append((date, path))
    result.sort(key=lambda item: item[0])
    if not result:
        raise FileNotFoundError(f"No NSIDC-0779 daily GeoTIFF files in {folder}")
    return result


def hours_since_epoch(value):
    return (value - EPOCH).total_seconds() / 3600.0


def date_from_filename(path):
    name = os.path.basename(path)
    for fmt, rx in (("%Y-%m-%d", r"(\d{4}-\d{2}-\d{2})"), ("%Y%m%d", r"(\d{8})")):
        m = re.search(rx, name)
        if m:
            try:
                return datetime.strptime(m.group(1), fmt)
            except ValueError:
                pass
    return None


def season_of(month):
    return next(name for name, months in SEASONS.items() if month in months)


# ---------------------------------------------------------------------------
# Phase 1 (parallel over CIDs): HB at 6 AM LST + HRU-to-SMAP-cell mapping
# ---------------------------------------------------------------------------
def read_hru_lons(edir, cid):
    with nc.Dataset(os.path.join(edir, str(cid), "input_file.nc")) as ds:
        return np.asarray(ds.groups["parameters"].variables["lons"][:], dtype=float)


def read_cid_smc(edir, cid, args, nhru):
    """Return (times [h since epoch], smc[time, hru]) for the surface layer."""
    folder = os.path.join(edir, args["output_subdir"], str(cid))
    files = sorted(glob.glob(os.path.join(folder, "*.nc")))
    if not files:
        raise FileNotFoundError(f"No output files in {folder}")

    times, blocks = [], []
    for path in files:
        start = date_from_filename(path)
        if start is None:
            continue
        with nc.Dataset(path) as ds:
            grp = ds.groups["data"] if "data" in ds.groups else ds
            var = grp.variables[args["variable"]]
            if var.ndim == 3:                       # (time, hru, soil)
                data = np.ma.asarray(var[:, :, args["layer"]])
            elif var.ndim == 2:                     # (time, hru)
                data = np.ma.asarray(var[:, :])
            else:
                raise ValueError(f"Unexpected shape {var.shape} in {path}")
        data = data.filled(np.nan).astype(np.float32)
        if data.shape[1] != nhru and data.shape[0] == nhru:
            data = data.T
        if data.shape[1] != nhru:
            raise ValueError(f"CID {cid}: {path} has {data.shape[1]} HRUs, "
                             f"input_file.nc has {nhru}")
        data[(data < 0.0) | (data > 1.0)] = np.nan
        t0 = hours_since_epoch(start) + args["first_offset_hours"]
        times.append(t0 + args["dt_hours"] * np.arange(data.shape[0]))
        blocks.append(data)

    times = np.concatenate(times)
    data = np.concatenate(blocks, axis=0)
    order = np.argsort(times, kind="stable")
    return times[order], data[order]


def interpolate_to_solar_time(times, data, lons, smap_dates, max_gap):
    """HB SMC at 06:00 local solar time for every SMAP date and HRU."""
    midnights = np.array([hours_since_epoch(d) for d in smap_dates])
    target = midnights[:, None] + SOLAR_HOUR - lons[None, :] / 15.0   # (date, hru)
    upper = np.searchsorted(times, target, side="left")
    ok = (upper > 0) & (upper < times.size)
    upper = np.clip(upper, 1, times.size - 1)
    lower = upper - 1
    interval = times[upper] - times[lower]
    ok &= (interval > 0) & (interval <= max_gap)
    weight = np.where(ok, (target - times[lower]) / np.where(interval > 0, interval, 1), 0)
    cols = np.broadcast_to(np.arange(lons.size)[None, :], target.shape)
    low = data[lower, cols]
    high = data[upper, cols]
    result = low * (1.0 - weight) + high * weight
    result[~ok] = np.nan
    return result.astype(np.float32)


def read_cid_hru_map(edir, cid):
    """Return (hru map with -9999 outside the CID, transform, crs)."""
    with rasterio.open(os.path.join(edir, str(cid), "hru_mapping_latlon.tif")) as src:
        hrus = src.read(1).astype(np.float64)
        transform, crs = src.transform, src.crs
        nodata = src.nodata
    invalid = ~np.isfinite(hrus) | (hrus < 0)
    if nodata is not None:
        invalid |= hrus == nodata
    mask_path = os.path.join(edir, "postprocess", "cids", f"{cid}.tif")
    if os.path.isfile(mask_path):
        with rasterio.open(mask_path) as src:
            mask = src.read(1)
        if mask.shape == hrus.shape:
            invalid |= mask != cid
    hrus[invalid] = NODATA
    return hrus.astype(np.int32), transform, crs


def process_cid(cid, edir, args, smap_dates, smap_grid, canvas):
    lons = read_hru_lons(edir, cid)
    nhru = lons.size

    # (a) HydroBlocks SMC at 06:00 local solar time
    times, data = read_cid_smc(edir, cid, args, nhru)
    hb_daily = interpolate_to_solar_time(times, data, lons, smap_dates,
                                         max_gap=args["dt_hours"] + 0.01)
    del data

    # (b) HRU pixel -> SMAP cell mapping (EASE-Grid 2.0 is cylindrical, so the
    #     projected x depends only on longitude and y only on latitude)
    hrus, tr, _ = read_cid_hru_map(edir, cid)
    ny, nx = hrus.shape
    lon_c = tr.c + (np.arange(nx) + 0.5) * tr.a
    lat_c = tr.f + (np.arange(ny) + 0.5) * tr.e
    xs, _ = warp_transform("EPSG:4326", smap_grid["crs"], lon_c, np.zeros(nx))
    _, ys = warp_transform("EPSG:4326", smap_grid["crs"], np.zeros(ny), lat_c)
    st = smap_grid["transform"]
    col_idx = np.floor((np.asarray(xs) - st.c) / st.a).astype(np.int64) - smap_grid["col_off"]
    row_idx = np.floor((np.asarray(ys) - st.f) / st.e).astype(np.int64) - smap_grid["row_off"]

    rows, cols = np.nonzero(hrus != NODATA)
    hru_of_pixel = hrus[rows, cols].astype(np.int64)
    r, c = row_idx[rows], col_idx[cols]
    inside = (r >= 0) & (r < smap_grid["height"]) & (c >= 0) & (c < smap_grid["width"])
    keep = inside & (hru_of_pixel < nhru)
    cell = r[keep] * smap_grid["width"] + c[keep]
    hru_keep = hru_of_pixel[keep]
    ncell = smap_grid["height"] * smap_grid["width"]
    pair, count = np.unique(hru_keep * ncell + cell, return_counts=True)
    pair_hru = (pair // ncell).astype(np.int32)
    pair_cell = (pair % ncell).astype(np.int64)
    area = np.bincount(hru_of_pixel[hru_of_pixel < nhru], minlength=nhru).astype(np.float64)

    # (c) Footprint on the plotting canvas (nearest neighbour, from this CID's raster)
    dst = np.full(canvas["shape"], NODATA, dtype=np.int32)
    reproject(source=hrus, destination=dst,
              src_transform=tr, src_crs="EPSG:4326", src_nodata=NODATA,
              dst_transform=canvas["transform"], dst_crs="EPSG:4326",
              dst_nodata=NODATA, resampling=Resampling.nearest)
    footprint = np.flatnonzero(dst >= 0)
    footprint_hru = dst.flat[footprint]

    return {
        "cid": cid, "nhru": nhru, "hb_daily": hb_daily,
        "pair_hru": pair_hru, "pair_cell": pair_cell,
        "pair_weight": count.astype(np.float32), "area": area,
        "footprint": footprint, "footprint_hru": footprint_hru,
    }


def process_cid_safe(packed):
    cid = packed[0]
    try:
        return process_cid(*packed)
    except Exception as error:                     # keep going if one CID fails
        return {"cid": cid, "error": f"{type(error).__name__}: {error}"}


# ---------------------------------------------------------------------------
# Phase 2 (parallel over date chunks): SMAP -> HRU means, seasonal accumulation
# ---------------------------------------------------------------------------
def init_smap_worker(pair_hru, pair_cell, pair_weight, area, total_hru, smap_grid):
    WORKER.update(pair_hru=pair_hru, pair_cell=pair_cell, pair_weight=pair_weight,
                  area=area, total_hru=total_hru, smap_grid=smap_grid)


def read_smap_window(path, grid):
    window = Window(grid["col_off"], grid["row_off"], grid["width"], grid["height"])
    with rasterio.open(path) as src:
        arr = src.read(SMAP_BAND, window=window, boundless=True, masked=True,
                       fill_value=src.nodata).filled(np.nan).astype(np.float32)
    arr[(~np.isfinite(arr)) | (arr <= 0.0) | (arr > 1.0)] = np.nan
    return arr.ravel()


def process_dates(task):
    items, hb_block = task                         # hb_block: (n_items, total_hru)
    n = WORKER["total_hru"]
    acc = {s: {"hb_sum": np.zeros(n), "smap_sum": np.zeros(n),
               "count": np.zeros(n, dtype=np.uint32)} for s in SEASONS}
    used = 0
    for k, (date, path) in enumerate(items):
        hb = hb_block[k]
        if not np.any(np.isfinite(hb)):
            continue
        smap_flat = read_smap_window(path, WORKER["smap_grid"])
        vals = smap_flat[WORKER["pair_cell"]]
        ok = np.isfinite(vals)
        w = WORKER["pair_weight"]
        hru = WORKER["pair_hru"]
        wsum = np.bincount(hru[ok], weights=w[ok], minlength=n)
        vsum = np.bincount(hru[ok], weights=w[ok] * vals[ok], minlength=n)
        # Average of the valid SMAP cells within each HRU (no coverage threshold,
        # as in the original script: any valid SMAP data counts)
        covered = wsum > 0
        smap_hru = np.full(n, np.nan)
        smap_hru[covered] = vsum[covered] / wsum[covered]

        valid = np.isfinite(hb) & np.isfinite(smap_hru)
        if not np.any(valid):
            continue
        a = acc[season_of(date.month)]
        a["hb_sum"][valid] += hb[valid]
        a["smap_sum"][valid] += smap_hru[valid]
        a["count"][valid] += 1
        used += 1
    return acc, used


# ---------------------------------------------------------------------------
# Statistics and plotting
# ---------------------------------------------------------------------------
def finalize(acc):
    out = {}
    for s in SEASONS:
        cnt = acc[s]["count"]
        hb = np.full(cnt.shape, np.nan)
        sm = np.full(cnt.shape, np.nan)
        np.divide(acc[s]["hb_sum"], cnt, out=hb, where=cnt > 0)
        np.divide(acc[s]["smap_sum"], cnt, out=sm, where=cnt > 0)
        out[s] = {"hb": hb, "smap": sm, "difference": hb - sm, "count": cnt}
    return out


def weighted_metrics(hb, smap, area):
    valid = np.isfinite(hb) & np.isfinite(smap) & (area > 0)
    d = hb[valid] - smap[valid]
    w = area[valid]
    n = int(valid.sum())
    if n == 0:
        return d, w, np.nan, np.nan, np.nan, np.nan, 0
    w = w / w.sum()
    md = float(np.sum(w * d))
    mad = float(np.sum(w * np.abs(d)))
    rmse = float(np.sqrt(np.sum(w * d ** 2)))
    x, y = hb[valid], smap[valid]
    mx, my = np.sum(w * x), np.sum(w * y)
    sx = np.sqrt(np.sum(w * (x - mx) ** 2))
    sy = np.sqrt(np.sum(w * (y - my) ** 2))
    r = float(np.sum(w * (x - mx) * (y - my)) / (sx * sy)) if n >= 2 and sx > 0 and sy > 0 else np.nan
    return d, area[valid], md, mad, rmse, r, n


def to_image(values, canvas_ids):
    img = np.full(canvas_ids.shape, np.nan)
    ok = canvas_ids >= 0
    img[ok] = values[canvas_ids[ok]]
    return img


def plot_maps(maps, canvas_ids, extent, output, dpi, limits=None):
    moist, diffs = [], []
    for s in SEASONS:
        for k in ("hb", "smap"):
            v = maps[s][k][np.isfinite(maps[s][k])]
            if v.size:
                moist.append(v)
        v = maps[s]["difference"][np.isfinite(maps[s]["difference"])]
        if v.size:
            diffs.append(np.abs(v))
    if not moist:
        raise RuntimeError("No paired HydroBlocks-SMAP HRUs were available")
    allm = np.concatenate(moist)
    # Full data range (no percentile clipping), as in the original script
    vmin, vmax = float(np.min(allm)), float(np.max(allm))
    dlim = max(float(np.max(np.concatenate(diffs))), 1e-6)

    mnorm = colors.Normalize(vmin=vmin, vmax=vmax)
    dnorm = colors.TwoSlopeNorm(vmin=-dlim, vcenter=0.0, vmax=dlim)
    mcmap = plt.cm.viridis.copy(); mcmap.set_bad("white")
    dcmap = plt.cm.RdBu_r.copy(); dcmap.set_bad("white")

    fig, axes = plt.subplots(4, 3, figsize=(13.5, 17), sharex=True, sharey=True,
                             constrained_layout=True)
    headers = ("HydroBlocks SMC", "SMAP SMC", "HydroBlocks − SMAP")
    im_m = im_d = None
    for r, season in enumerate(SEASONS):
        item = maps[season]
        panels = ((item["hb"], mcmap, mnorm), (item["smap"], mcmap, mnorm),
                  (item["difference"], dcmap, dnorm))
        for c, (vals, cmap, norm) in enumerate(panels):
            ax = axes[r, c]
            im = ax.imshow(to_image(vals, canvas_ids), extent=extent, origin="upper",
                           cmap=cmap, norm=norm, interpolation="nearest", aspect="equal")
            if c < 2:
                im_m = im
            else:
                im_d = im
            if r == 0:
                ax.set_title(headers[c], fontsize=FS_HEADER, fontweight="normal", pad=10)
            if r == 3:
                ax.set_xlabel("Longitude [°]", fontsize=FS_LABEL)
            if c == 0:
                ax.set_ylabel("Latitude [°]", fontsize=FS_LABEL)
            ax.tick_params(labelsize=FS_TICK)
            if limits is not None:
                ax.set_xlim(limits[0])
                ax.set_ylim(limits[1])
        # Bold season heading at the left of each row
        axes[r, 0].text(-0.30, 0.5, season, transform=axes[r, 0].transAxes,
                        rotation=90, ha="center", va="center",
                        fontsize=FS_SEASON, fontweight="bold")

    cb1 = fig.colorbar(im_m, ax=axes[:, :2], orientation="horizontal",
                       fraction=0.025, pad=0.02, aspect=50)
    cb1.set_label("Soil moisture [m³ m⁻³]", fontsize=FS_LABEL)
    cb1.ax.tick_params(labelsize=FS_TICK)
    cb2 = fig.colorbar(im_d, ax=axes[:, 2], orientation="horizontal",
                       fraction=0.025, pad=0.02, aspect=25)
    cb2.set_label("ΔSMC [m³ m⁻³]", fontsize=FS_LABEL)
    cb2.ax.tick_params(labelsize=FS_TICK)

    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def plot_histograms(maps, area, output, dpi, bins):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9), constrained_layout=True)
    rows = []
    for ax, season in zip(axes.ravel(), SEASONS):
        d, w, md, mad, rmse, r, n = weighted_metrics(maps[season]["hb"],
                                                     maps[season]["smap"], area)
        rows.append((season, md, mad, rmse, r, n))
        ax.set_title(season, fontsize=FS_SEASON, fontweight="bold")
        if n == 0:
            ax.text(0.5, 0.5, "no paired HRUs", transform=ax.transAxes,
                    ha="center", va="center", fontsize=FS_LABEL)
            ax.set_axis_off()
            continue
        ax.hist(d, bins=bins, weights=w, density=True, color="#4C72B0", edgecolor="white",
                linewidth=0.4)
        ax.axvline(md, color="darkred", linestyle="--", linewidth=2.5)
        rtxt = f"{r:.3f}" if np.isfinite(r) else "NA"
        ax.text(0.98, 0.97,
                f"MD = {md:.3f} m³ m⁻³\nMAD = {mad:.3f} m³ m⁻³\n"
                f"RMSE = {rmse:.3f} m³ m⁻³\nr = {rtxt}\nHRUs = {n:,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=FS_STATS,
                bbox={"boxstyle": "round", "facecolor": "white",
                      "edgecolor": "0.5", "alpha": 0.85})
        ax.set_xlabel("HydroBlocks − SMAP [m³ m⁻³]", fontsize=FS_LABEL)
        ax.set_ylabel("Probability density", fontsize=FS_LABEL)
        ax.tick_params(labelsize=FS_TICK)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)
    return rows


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def domain_bounds(edir, cids):
    """Extent of the valid HRU pixels over all CIDs (excludes each CID raster's buffer)."""
    west = south = np.inf
    east = north = -np.inf
    for cid in cids:
        path = os.path.join(edir, str(cid), "hru_mapping_latlon.tif")
        if not os.path.isfile(path):
            continue
        hrus, tr, _ = read_cid_hru_map(edir, cid)
        rows, cols = np.nonzero(hrus != NODATA)
        if rows.size == 0:
            continue
        x0 = tr.c + cols.min() * tr.a
        x1 = tr.c + (cols.max() + 1) * tr.a
        y0 = tr.f + (rows.max() + 1) * tr.e
        y1 = tr.f + rows.min() * tr.e
        west, east = min(west, x0, x1), max(east, x0, x1)
        south, north = min(south, y0, y1), max(north, y0, y1)
    if not np.isfinite(west):
        raise FileNotFoundError("No valid HRU pixels found in hru_mapping_latlon.tif files")
    return west, south, east, north


def smap_window(sample_smap, bounds):
    west, south, east, north = bounds
    with rasterio.open(sample_smap) as src:
        crs, tr = src.crs, src.transform
    if crs is None:
        raise ValueError(f"{sample_smap} has no coordinate reference system")
    epsg = crs.to_epsg()
    # Both supported grids are "separable": projected x depends only on
    # longitude and y only on latitude, which the fast pixel mapping relies on.
    #   EPSG:4326 - regular latitude/longitude grid (e.g. extracted/reprojected SMAP)
    #   EPSG:6933 - original NSIDC-0779 EASE-Grid 2.0 (cylindrical equal-area)
    if not (crs.is_geographic or epsg == 6933):
        raise ValueError(f"SMAP must be in a lat/lon CRS (e.g. EPSG:4326) or "
                         f"EPSG:6933 (EASE-Grid 2.0); got {crs}")
    if tr.b != 0 or tr.d != 0:
        raise ValueError("Rotated SMAP rasters are not supported")
    print(f"SMAP grid: {crs.to_string()} | pixel size {abs(tr.a):.6g} x {abs(tr.e):.6g}",
          flush=True)
    xs, ys = warp_transform("EPSG:4326", crs, [west, east, west, east],
                            [south, south, north, north])
    cols = [(x - tr.c) / tr.a for x in xs]          # works for north-up or south-up
    rows = [(y - tr.f) / tr.e for y in ys]
    c0, c1 = int(np.floor(min(cols))) - 1, int(np.ceil(max(cols))) + 1
    r0, r1 = int(np.floor(min(rows))) - 1, int(np.ceil(max(rows))) + 1
    return {"crs": crs.to_string(), "transform": tr, "col_off": c0, "row_off": r0,
            "width": c1 - c0, "height": r1 - r0}


def main():
    a = arguments()
    if a.end_year < a.start_year:
        raise ValueError("end-year must not precede start-year")
    edir = os.path.abspath(a.edir)
    outdir = a.outdir or os.path.join(edir, "validation_plots", "SMAP_comparison")
    os.makedirs(outdir, exist_ok=True)

    cids = discover_cids(edir, a.cids)
    smap_files = discover_smap_files(os.path.abspath(a.smap_dir), a.start_year, a.end_year)
    smap_dates = [d for d, _ in smap_files]
    bounds = domain_bounds(edir, cids)
    smap_grid = smap_window(smap_files[0][1], bounds)

    west, south, east, north = bounds
    # Small, even border on all four sides (fraction of the domain size)
    # Same border width (in degrees) on every side, even for non-square domains
    pad = a.border * max(east - west, north - south)
    west, east, south, north = west - pad, east + pad, south - pad, north + pad
    scale = max(east - west, north - south) / a.plot_pixels
    cw, ch = int(np.ceil((east - west) / scale)), int(np.ceil((north - south) / scale))
    canvas = {"shape": (ch, cw),
              "transform": from_bounds(west, south, west + cw * scale, south + ch * scale, cw, ch)}
    extent = (west, west + cw * scale, south, south + ch * scale)
    plot_limits = ((west, east), (south, north))   # exact limits -> identical border on all sides

    workers = max(1, a.workers)
    print(f"CIDs: {len(cids)} | SMAP days: {len(smap_files)} | workers: {workers}", flush=True)
    print(f"Domain: {west:.3f}–{east:.3f}°E, {south:.3f}–{north:.3f}°N", flush=True)

    # ---- Phase 1: per CID ------------------------------------------------
    cid_args = {"variable": a.variable, "layer": a.layer, "output_subdir": a.output_subdir,
                "dt_hours": a.dt_hours, "first_offset_hours": a.first_offset_hours}
    tasks = [(cid, edir, cid_args, smap_dates, smap_grid, canvas) for cid in cids]
    results = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for k, res in enumerate(ex.map(process_cid_safe, tasks, chunksize=1), 1):
            if "error" in res:
                print(f"  CID {res['cid']} skipped: {res['error']}", flush=True)
            else:
                results.append(res)
            if k % 10 == 0 or k == len(tasks):
                print(f"  Phase 1 (CIDs): {k}/{len(tasks)}", flush=True)
    if not results:
        raise RuntimeError("No CID could be processed")

    # Combine CIDs into one global HRU index
    offsets, total = {}, 0
    for res in results:
        offsets[res["cid"]] = total
        total += res["nhru"]
    hb_all = np.full((len(smap_dates), total), np.nan, dtype=np.float32)
    area = np.zeros(total)
    canvas_ids = np.full(canvas["shape"], -1, dtype=np.int64)
    ph, pc, pw = [], [], []
    for res in results:
        o, n = offsets[res["cid"]], res["nhru"]
        hb_all[:, o:o + n] = res["hb_daily"]
        area[o:o + n] = res["area"]
        ph.append(res["pair_hru"].astype(np.int64) + o)
        pc.append(res["pair_cell"])
        pw.append(res["pair_weight"])
        canvas_ids.flat[res["footprint"]] = res["footprint_hru"].astype(np.int64) + o
    pair_hru, pair_cell, pair_weight = np.concatenate(ph), np.concatenate(pc), np.concatenate(pw)
    del results
    print(f"Total HRUs: {total:,} | HRU–SMAP cell pairs: {pair_hru.size:,}", flush=True)

    # ---- Phase 2: per date chunk -----------------------------------------
    nchunks = min(len(smap_files), workers * 4)
    edges = np.linspace(0, len(smap_files), nchunks + 1).astype(int)
    date_tasks = [(smap_files[i:j], hb_all[i:j]) for i, j in zip(edges[:-1], edges[1:]) if j > i]
    acc = {s: {"hb_sum": np.zeros(total), "smap_sum": np.zeros(total),
               "count": np.zeros(total, dtype=np.uint32)} for s in SEASONS}
    used = 0
    with ProcessPoolExecutor(max_workers=workers, initializer=init_smap_worker,
                             initargs=(pair_hru, pair_cell, pair_weight, area, total,
                                       smap_grid)) as ex:
        for k, (part, n_used) in enumerate(ex.map(process_dates, date_tasks), 1):
            for s in SEASONS:
                for key in ("hb_sum", "smap_sum", "count"):
                    acc[s][key] += part[s][key]
            used += n_used
            print(f"  Phase 2 (dates): {k}/{len(date_tasks)}", flush=True)

    # ---- Outputs -----------------------------------------------------------
    maps = finalize(acc)
    label = f"{a.start_year}_{a.end_year}"
    plot_maps(maps, canvas_ids, extent,
              os.path.join(outdir, f"seasonal_06AM_HB_SMAP_maps_{label}.png"), a.dpi,
              limits=plot_limits)
    metrics = plot_histograms(maps, area,
                              os.path.join(outdir, f"seasonal_06AM_HB_minus_SMAP_histograms_{label}.png"),
                              a.dpi, a.bins)

    csv_path = os.path.join(outdir, f"seasonal_06AM_HB_minus_SMAP_metrics_{label}.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["pass", "description", "season", "MD_m3_m3",
                                          "MAD_m3_m3", "RMSE_m3_m3", "correlation_r",
                                          "valid_hrus_n", "paired_days_used",
                                          "weighting"])
        w.writeheader()
        for season, md, mad, rmse, r, n in metrics:
            w.writerow({"pass": "06AM", "description": PASS_DESCRIPTION, "season": season,
                        "MD_m3_m3": md, "MAD_m3_m3": mad, "RMSE_m3_m3": rmse,
                        "correlation_r": r, "valid_hrus_n": n, "paired_days_used": used,
                        "weighting": "HRU area"})
    print(f"Saved: {csv_path}", flush=True)
    print(f"Paired days used: {used}", flush=True)


if __name__ == "__main__":
    main()