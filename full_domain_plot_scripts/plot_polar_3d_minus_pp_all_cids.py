#!/usr/bin/env python3
"""Read-only raw-CID analysis: multi-year 3D downscaled minus PP downscaled. Requires numpy, netCDF4,
rasterio, matplotlib. No simulation files are modified. Environment: HB_PATH_3D, HB_PATH_PP,
HB_OUTDIR, SLURM_CPUS_PER_TASK, HB_TIME_CHUNK (default 128), HB_ALLOW_PARTIAL (default 0). Writes
four six-panel PNG figures (radiation/albedo and temperature/flux/snow). Stored model aspect is the
default; HB_OUTDIR can override the output directory. Each run maps through its own zero-based HRU
raster. All masks are required. Clocks must match file by file, and be strictly increasing across
files. Means use common valid times per geographic HRU pair and per variable. Rnet requires both
SWnet and LWnet valid. Finite -9999 and declared fills are excluded. Albedo and snow cover fraction
outside [0,1] are excluded. SWE values are assumed to be mm water equivalent (kg m-2 is numerically
identical). Temperature and flux values retain the output conventions; no sign reversal. Bins are
pixel weighted, not area weighted. Slope bins use pixel terrain in every aspect mode; limits are
[0,45) and [0,65). Aspect sectors are [0,22.5), etc. Flat pixels remain included where an aspect is
supplied; job-log coverage summaries count them. Circular resultants <=1e-6 have undefined
direction. Stored-model aspect reflects the completed run, including any arithmetic averaging error;
plotting cannot repair a completed simulation.
"""
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import netCDF4 as nc
import rasterio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.colors as colors
# model outputs (can be overridden from the job script with HB_PATH_PP / HB_PATH_3D)
path_pp = os.environ.get("HB_PATH_PP", "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test/")
path_3d = os.environ.get("HB_PATH_3D", "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/")
# output directory (override with HB_OUTDIR)
outputdir = os.environ.get("HB_OUTDIR", "/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/Polar_3D_minus_PP_model_aspect")
os.makedirs(outputdir, exist_ok=True)
# output files to average over (in output_data/<cid>/). "*.nc" = whole simulation;
# use e.g. "2016-01-01.nc" to reproduce the original single-file analysis
OUTPUT_FILES = "*.nc"
# HRU rasters use zero-based indices.
# Aspect used for the aspect bins:
#   "hru_xy" : each pixel takes its HRU's aspect, computed as the circular mean
#              atan2(x_aspect, y_aspect) from input_file.nc (correct averaging of
#              angles; this is the HRU orientation implied by the clustering)
#   "pixel"  : each pixel's own 90-m aspect from aspect_latlon.tif
ASPECT_SOURCE = "hru_model"  # hru_model, hru_xy, pixel, or both (the two HRU sources).
CHUNK = int(os.environ.get("HB_TIME_CHUNK", "128"))
ALLOW_PARTIAL = os.environ.get("HB_ALLOW_PARTIAL", "0") == "1"
# parallel workers
WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))
# variables
# Compare internal SW diagnostics in both runs. PP LW uses its unchanged forcing.
DOWN_VARS = {'3d': {'swdn': 'swdn_3d', 'lwdn': 'lwdn_3d'}, 'pp': {'swdn': 'swdn_3d', 'lwdn': 'lwdn'}}
vars_needed = ["swdn", "swnet", "lwdn", "lwnet", "salb", "tv", "trad", "sh", "lh", "fsno", "swe"]
mean_keys = ["swdn", "swnet", "lwdn", "lwnet", "rnet", "salb", "trad", "tv", "fsno", "swe", "sh", "lh"]
state_order = ["ΔTsur", "ΔTveg", "Δfsno", "ΔSWE", "ΔFsen", "ΔFlat"]
panel_order = ["ΔSWdown", "ΔSWnet", "ΔLWdown", "ΔLWnet", "ΔRnet", "ΔAlbedo"]
all_panel_order = panel_order + state_order
rad_keys = ["ΔSWdown", "ΔSWnet", "ΔLWdown", "ΔLWnet", "ΔRnet"]
# bins (computed once up to 65 deg; the main figure uses the first nine rings)
slope_edges_full = np.arange(0, 70, 5)        # 0..65
slope_edges_masked = np.arange(0, 50, 5)      # 0..45
aspect_edges = np.arange(0, 360, 22.5)
n_r_full = len(slope_edges_full) - 1
n_t = len(aspect_edges)

# ---------------------------------------------------------------------------
# Per-CID helpers
# ---------------------------------------------------------------------------
def list_cids(path):
    cids = sorted((int(d) for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))))
    if not cids:
        raise FileNotFoundError(f"No CID folders in {path}")
    return cids


def read_grid(fp):
    with rasterio.open(fp) as ds:
        arr = ds.read(1, masked=True).astype(float).filled(np.nan)
        signature = (ds.shape, ds.crs, ds.transform)
    arr[arr == -9999] = np.nan
    return arr, signature


def checked_grid(fp, signature):
    arr, got = read_grid(fp)
    if got != signature:
        raise ValueError(f"Raster alignment mismatch: {fp}")
    return arr


def read_cid_hru_map(path, cid, signature):
    arr = checked_grid(os.path.join(path, str(cid), "hru_mapping_latlon.tif"), signature)
    mask = checked_grid(os.path.join(path, "postprocess", "cids", f"{cid}.tif"), signature)
    valid = np.isfinite(arr) & (mask == cid)
    if np.any(valid & ((arr < 0) | (arr != np.floor(arr)))):
        raise ValueError("Invalid zero-based HRU IDs")
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[valid] = arr[valid].astype(np.int64)
    return out


def place_data(idx, model):
    out = np.full(idx.shape, np.nan)
    valid = (idx >= 0) & (idx < model.size)
    out[valid] = model[idx[valid]]
    return out


def clock(ds):
    v = ds.groups["metadata"].variables["date"]
    dates = nc.num2date(v[:], v.units, calendar=getattr(v, "calendar", "standard"))
    return [str(t) for t in dates]


def clean(var, start, end, albedo=False):
    a = np.ma.asarray(var[start:end], dtype=float).filled(np.nan)
    for attr in ("_FillValue", "missing_value"):
        for value in np.atleast_1d(getattr(var, attr, [])):
            a[a == value] = np.nan
    a[(a == -9999) | ~np.isfinite(a)] = np.nan
    if albedo:
        a[(a < 0) | (a > 1)] = np.nan
    return a


def paired_means(cid, pairs):
    """Mean differences on common valid timesteps for each geographic HRU pair. Require identical file
    partitioning and strictly increasing matching clocks; fail explicitly rather than silently
    averaging mismatched output periods.
    """
    paths = [path_3d, path_pp]
    files = [sorted(glob.glob(os.path.join(p, "output_data", str(cid), OUTPUT_FILES))) for p in paths]
    if not files[0] or not files[1] or len(files[0]) != len(files[1]):
        raise ValueError("Missing output files or unequal file counts")
    sums = np.zeros((len(mean_keys), len(pairs)))
    counts = np.zeros_like(sums, dtype=np.int64)
    previous = None
    time_count = 0
    for f3, fp in zip(*files):
        with nc.Dataset(f3) as d3, nc.Dataset(fp) as dp:
            times = clock(d3)
            if times != clock(dp) or not times:
                raise ValueError(f"Timestamp mismatch/empty clock: {f3}, {fp}")
            if any(x >= y for x, y in zip(times, times[1:])) or (previous is not None and times[0] <= previous):
                raise ValueError("Duplicate or non-increasing timestamps across output files")
            previous = times[-1]
            groups = [d3.groups["data"], dp.groups["data"]]
            for run, group, ids in zip(("3d", "pp"), groups, pairs.T):
                for key in vars_needed:
                    v = group.variables[DOWN_VARS[run].get(key, key)]
                    if v.ndim != 2 or v.shape[0] != len(times) or np.max(ids) >= v.shape[1]:
                        raise ValueError(f"Invalid time/HRU dimensions: {v.name} {v.shape}")
            for start in range(0, len(times), CHUNK):
                end = min(start + CHUNK, len(times))
                data = []
                for run, group, ids in zip(("3d", "pp"), groups, pairs.T):
                    values = {key: clean(group.variables[DOWN_VARS[run].get(key, key)], start, end, key in ("salb", "fsno"))[:, ids] for key in vars_needed}
                    values["rnet"] = values["swnet"] - values["lwnet"]
                    data.append(values)
                for k, key in enumerate(mean_keys):
                    diff = data[0][key] - data[1][key]
                    valid = np.isfinite(diff)
                    sums[k] += np.sum(np.where(valid, diff, 0), axis=0)
                    counts[k] += valid.sum(axis=0)
            time_count += len(times)
    result = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=result, where=counts > 0)
    return result, counts, time_count


def process_cid(cid):
    try:
        root = os.path.join(path_3d, str(cid))
        slope, signature = read_grid(os.path.join(root, "slope_latlon.tif"))
        slope_deg = np.degrees(np.arctan(slope))
        h3 = read_cid_hru_map(path_3d, cid, signature)
        hp = read_cid_hru_map(path_pp, cid, signature)
        common = (h3 >= 0) & (hp >= 0)
        if not np.any(common):
            raise ValueError("No common CID pixels")
        pairs, inverse = np.unique(np.column_stack((h3[common], hp[common])), axis=0, return_inverse=True)
        means, paired_counts, nt = paired_means(cid, pairs)
        aspects = {}
        with nc.Dataset(os.path.join(root, "input_file.nc")) as ds:
            par = ds.groups["parameters"]
            if "hru_model" in SOURCES:
                raw = np.ma.asarray(par.variables["aspect"][:], dtype=float).filled(np.nan)
                raw[raw == -9999] = np.nan
                aspects["hru_model"] = place_data(h3, (180 - np.degrees(raw)) % 360)
            if "hru_xy" in SOURCES:
                x = np.ma.asarray(par.variables["x_aspect"][:], dtype=float).filled(np.nan)
                y = np.ma.asarray(par.variables["y_aspect"][:], dtype=float).filled(np.nan)
                angle = (180 - np.degrees(np.arctan2(x, y))) % 360
                angle[(np.hypot(x, y) <= 1e-6) | (x == -9999) | (y == -9999)] = np.nan
                aspects["hru_xy"] = place_data(h3, angle)
        if "pixel" in SOURCES:
            raw = checked_grid(os.path.join(root, "aspect_latlon.tif"), signature)
            aspects["pixel"] = (180 - np.degrees(raw)) % 360
        sb = np.searchsorted(slope_edges_full, slope_deg[common], side="right") - 1
        output = {}
        for source, grid in aspects.items():
            angles = grid[common]
            ab = np.floor(np.where(np.isfinite(angles), angles, -22.5) / 22.5).astype(int)
            valid = np.isfinite(slope_deg[common]) & np.isfinite(angles) & (sb >= 0) & (sb < n_r_full)
            sums = np.zeros((len(mean_keys), n_r_full, n_t))
            counts = np.zeros_like(sums, dtype=np.int64)
            for k in range(len(mean_keys)):
                values = means[k, inverse]
                ok = valid & np.isfinite(values)
                bins = sb[ok] * n_t + ab[ok]
                sums[k] = np.bincount(bins, weights=values[ok], minlength=n_r_full*n_t).reshape(n_r_full, n_t)
                counts[k] = np.bincount(bins, minlength=n_r_full*n_t).reshape(n_r_full, n_t)
            coverage = dict(common_pixels=int(common.sum()), outside_slope_range=int(np.count_nonzero(~np.isfinite(slope_deg[common]) | (sb < 0) | (sb >= n_r_full))), undefined_aspect=int(np.count_nonzero(~np.isfinite(angles))), flat_pixels=int(np.count_nonzero(slope_deg[common] == 0)), timesteps=nt, minimum_pair_valid_times=int(paired_counts.min()))
            output[source] = (sums, counts, coverage)
        return cid, output, None
    except Exception as err:
        return cid, None, f"{type(err).__name__}: {err}"

# ---------------------------------------------------------------------------
# Plot (same format as the original notebook; only the headings are bold)
# ---------------------------------------------------------------------------
def plot_polar(polar_means, slope_edges, rmax, rtick_step, outfile):
    theta_edges = np.deg2rad(np.append(aspect_edges, 360.0))
    Theta, R = np.meshgrid(theta_edges, slope_edges)
    # Fixed ranges for each slope range; six panels retained.
    vmax_rad = 50.0 if rmax == 45 else 60.0
    vmax_alb = 0.02 if rmax == 45 else 0.02
    norm_rad = colors.TwoSlopeNorm(vmin=-vmax_rad, vcenter=0.0, vmax=vmax_rad)
    norm_alb = colors.TwoSlopeNorm(vmin=-vmax_alb, vcenter=0.0, vmax=vmax_alb)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")
    # plot
    fig = plt.figure(figsize=(13, 8), facecolor="white")
    axes = [fig.add_subplot(2, 3, i + 1, projection="polar") for i in range(6)]
    # bold headings (mathbf makes the math part bold too)
    panel_titles = [
        r"(a) $\mathbf{\Delta SW_{down}}$",
        r"(b) $\mathbf{\Delta SW_{net}}$",
        r"(c) $\mathbf{\Delta LW_{down}}$",
        r"(d) $\mathbf{\Delta LW_{net}}$",
        r"(e) $\mathbf{\Delta R_{net}}$",
        r"(f) $\mathbf{\Delta Albedo}$",
    ]
    pcm_rad = None
    pcm_alb = None
    rticks = np.arange(0, rmax + 1, rtick_step)
    for ax, key, title in zip(axes, panel_order, panel_titles):
        if key == "ΔAlbedo":
            pcm = ax.pcolormesh(Theta, R, polar_means[key], cmap=cmap, norm=norm_alb, shading='auto')
            pcm_alb = pcm
        else:
            pcm = ax.pcolormesh(Theta, R, polar_means[key], cmap=cmap, norm=norm_rad, shading='auto')
            pcm_rad = pcm
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax)
        ax.set_rticks(rticks)
        ax.set_yticklabels([f"{r}°" for r in rticks], fontsize=8)
        ax.set_rlabel_position(157.5)
        ax.set_thetagrids(np.arange(0, 360, 45), labels=[f'{a}°' for a in np.arange(0, 360, 45)], fontsize=8)
        ax.grid(True, linewidth=0.7, alpha=0.8)
        ax.set_title(title, fontsize=12, fontweight="bold", pad=16)
        ax.set_facecolor("white")
    plt.subplots_adjust(wspace=0.35, hspace=0.40, right=0.85)
    # radiation colorbar
    cax1 = fig.add_axes([0.88, 0.45, 0.02, 0.40])
    rad_ticks = np.linspace(-vmax_rad, vmax_rad, 5)
    rad_vals = np.concatenate([polar_means[k].ravel() for k in rad_keys])
    low, high = np.any(rad_vals < -vmax_rad), np.any(rad_vals > vmax_rad)
    extension = "both" if low and high else "min" if low else "max" if high else "neither"
    cbar1 = fig.colorbar(pcm_rad, cax=cax1, ticks=rad_ticks, extend=extension)
    cbar1.set_label(r"W m$^{-2}$", fontsize=11)
    cbar1.ax.tick_params(labelsize=9)
    # albedo colorbar
    cax2 = fig.add_axes([0.88, 0.12, 0.02, 0.22])
    alb_ticks = np.linspace(-vmax_alb, vmax_alb, 5)
    alb_vals = polar_means["ΔAlbedo"]
    low, high = np.any(alb_vals < -vmax_alb), np.any(alb_vals > vmax_alb)
    extension = "both" if low and high else "min" if low else "max" if high else "neither"
    cbar2 = fig.colorbar(pcm_alb, cax=cax2, ticks=alb_ticks, extend=extension)
    cbar2.set_label("Albedo", fontsize=11)
    cbar2.ax.tick_params(labelsize=9)
    plt.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", outfile, flush=True)

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def plot_states(polar_means, slope_edges, rmax, rtick_step, outfile):
    """Fixed symmetric scales supplied by the user; extensions mark exceedances.
    """
    limits = {"ΔTsur": 2.0 if rmax == 45 else 3.0, "ΔTveg": 2.0 if rmax == 45 else 3.0, "Δfsno": 0.08,
              "ΔSWE": 30.0, "ΔFsen": 50.0, "ΔFlat": 5.0}
    units = {'ΔTsur': 'K', 'ΔTveg': 'K', 'Δfsno': 'Fraction', 'ΔSWE': 'mm', 'ΔFsen': 'W m$^{-2}$', 'ΔFlat': 'W m$^{-2}$'}
    titles = [r"(a) $\mathbf{\Delta T_{sur}}$", r"(b) $\mathbf{\Delta T_{veg}}$",
              r"(c) $\mathbf{\Delta f_{sno}}$", r"(d) $\mathbf{\Delta SWE}$",
              r"(e) $\mathbf{\Delta F_{sen}}$", r"(f) $\mathbf{\Delta F_{lat}}$"]
    theta, radius = np.meshgrid(np.deg2rad(np.append(aspect_edges, 360)), slope_edges)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("white")
    fig = plt.figure(figsize=(15, 9), facecolor="white")
    axes = [fig.add_subplot(2, 3, i+1, projection="polar") for i in range(6)]
    for ax, key, title in zip(axes, state_order, titles):
        values = polar_means[key]
        limit = limits[key]
        pcm = ax.pcolormesh(theta, radius, values, cmap=cmap,
                           norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), shading="auto")
        ax.set_theta_zero_location("N")
        ax.set_theta_direction(-1)
        ax.set_ylim(0, rmax)
        ticks = np.arange(0, rmax+1, rtick_step)
        ax.set_rticks(ticks)
        ax.set_yticklabels([f"{t}°" for t in ticks], fontsize=8)
        ax.set_rlabel_position(157.5)
        ax.set_thetagrids(np.arange(0,360,45), labels=[f"{a}°" for a in np.arange(0,360,45)], fontsize=8)
        ax.grid(True, linewidth=0.7, alpha=0.8)
        ax.set_title(title, fontsize=13, fontweight="bold", pad=16)
        low, high = np.any(values < -limit), np.any(values > limit)
        extension = "both" if low and high else "min" if low else "max" if high else "neither"
        bar = fig.colorbar(pcm, ax=ax, fraction=0.046, pad=0.12, ticks=np.linspace(-limit, limit, 5), extend=extension)
        bar.set_label(units[key], fontsize=10)
        bar.ax.tick_params(labelsize=8)
        finite = values[np.isfinite(values)]
        if finite.size:
            print(f"{key}, slopes 0–{rmax}: bin range {finite.min():.4f} to {finite.max():.4f}; "
                  f"bins below/above scale: {np.count_nonzero(values < -limit)}/{np.count_nonzero(values > limit)}", flush=True)
    fig.subplots_adjust(wspace=0.48, hspace=0.40, left=0.05, right=0.95, top=0.91, bottom=0.08)
    fig.savefig(outfile, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", outfile, flush=True)


def main():
    if CHUNK < 1 or WORKERS < 1:
        raise ValueError("Chunk size and workers must be positive")
    c3, cp = set(list_cids(path_3d)), set(list_cids(path_pp))
    missing = {"missing_pp": sorted(c3-cp), "missing_3d": sorted(cp-c3)}
    if any(missing.values()) and not ALLOW_PARTIAL:
        raise ValueError(f"CID sets differ: {missing}")
    cids = sorted(c3 & cp)
    if not cids:
        raise ValueError("No common CIDs")
    totals = {source: [np.zeros((len(mean_keys),n_r_full,n_t)), np.zeros((len(mean_keys),n_r_full,n_t), dtype=np.int64)] for source in SOURCES}
    report = dict(paths={"3d": path_3d, "pp": path_pp}, missing=missing, failed={}, coverage={}, weighting="equal raster pixels; pixel slope; selected aspect", sources=SOURCES)
    print(f"3D: {path_3d}\nPP: {path_pp}\nCIDs: {len(cids)}, workers: {WORKERS}, sources: {SOURCES}", flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for n, (cid, output, err) in enumerate(ex.map(process_cid, cids, chunksize=1), 1):
            if err:
                report["failed"][str(cid)] = err
                print(f"CID {cid} failed: {err}", flush=True)
            else:
                report["coverage"][str(cid)] = {}
                for source, (sums, counts, coverage) in output.items():
                    totals[source][0] += sums
                    totals[source][1] += counts
                    report["coverage"][str(cid)][source] = coverage
            if n % 25 == 0 or n == len(cids):
                print(f"Processed {n}/{len(cids)}", flush=True)
    print("Missing CID counterparts:", report["missing"], flush=True)
    for source in SOURCES:
        rows = [entry[source] for entry in report["coverage"].values()]
        summary = {key: sum(row[key] for row in rows) for key in
                   ("common_pixels", "outside_slope_range", "undefined_aspect", "flat_pixels")}
        print("Coverage:", source, summary, flush=True)
    if not report["coverage"] or (report["failed"] and not ALLOW_PARTIAL):
        raise RuntimeError("Incomplete or empty domain: see the job log. HB_ALLOW_PARTIAL=1 explicitly permits partial figures.")
    for source, (sums, counts) in totals.items():
        means = np.full_like(sums, np.nan)
        np.divide(sums, counts, out=means, where=counts > 0)
        if not np.isfinite(means).any():
            raise RuntimeError(f"No finite binned differences for {source}")
        full = dict(zip(all_panel_order, means))
        reduced = {key: v[:len(slope_edges_masked)-1] for key,v in full.items()}
        suffix = "_PARTIAL" if report["failed"] or any(missing.values()) else ""
        for data, edges, limit, step, name in ((reduced,slope_edges_masked,45,5,"slope0_45"),(full,slope_edges_full,65,10,"slope0_65")):
            plot_polar(data, edges, limit, step, os.path.join(outputdir, f"polar_multiyear_{name}_{source}_3D_minus_PP{suffix}.png"))
            plot_states(data, edges, limit, step, os.path.join(outputdir, f"polar_multiyear_temp_flux_snow_{name}_{source}_3D_minus_PP{suffix}.png"))
    print(f"CIDs used: {len(report['coverage'])}; failed: {len(report['failed'])}", flush=True)
SOURCES = ["hru_model", "hru_xy"] if ASPECT_SOURCE == "both" else [ASPECT_SOURCE]
if any(source not in ("hru_model", "hru_xy", "pixel") for source in SOURCES):
    raise ValueError("HB_ASPECT_SOURCE must be both, hru_model, hru_xy, or pixel")
if __name__ == "__main__":
    main()
