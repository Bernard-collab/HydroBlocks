#!/usr/bin/env python3
"""Seasonal HB-NMP 3d FSNO versus MODSCAG snow-fraction comparison.

MODSCAG percent values are converted to fractions and aggregated onto the
HB-NMP grid with average resampling. Difference colors use the full retained data range, symmetric about zero;
histograms include all retained differences. No percentiles are used.

Example
-------
python plot_postprocessed_fsno_vs_modscaq.py \
  --edir /path/to/3d_experiment \
  --modscag-dir /path/to/modscag_domain_extract_old \
  --start-year 2014 --end-year 2022 \
  --snow-threshold 0.05 \
  --outdir /path/to/validation_plots/3d_vs_MODSCAG
"""

import argparse
import calendar
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


# Match the SMC--SMAP plotting font settings.
plt.rcParams.update({
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
    "figure.titlesize": 20,
    "figure.titleweight": "bold",
})


SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edir", required=True, help="HB-NMP 3d experiment")
    parser.add_argument("--modscag-dir", required=True)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2022)
    parser.add_argument("--model-variable", default="fsno")
    parser.add_argument("--modscag-variable", default="snow_fraction")
    parser.add_argument(
        "--diff-limit", type=float, default=None,
        help="Optional symmetric map color limit; default uses maximum absolute retained difference",
    )
    parser.add_argument(
        "--snow-threshold", type=float, default=0.00,
        help="Keep pixels where seasonal 3d or MODSCAG fsno >= this value; use 0 to retain all valid zeros",
    )
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument(
        "--workers", type=int, default=1,
        help="Number of monthly worker processes (use SLURM_CPUS_PER_TASK)",
    )
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def axis_of(dimensions, candidates):
    names = [name.lower() for name in dimensions]
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    return None


def time_lat_lon(variable):
    values = np.ma.asarray(variable[:]).filled(np.nan).astype(float)
    time_axis = axis_of(variable.dimensions, ("time", "t"))
    lat_axis = axis_of(variable.dimensions, ("lat", "latitude", "y"))
    lon_axis = axis_of(variable.dimensions, ("lon", "longitude", "x"))
    if None in (time_axis, lat_axis, lon_axis):
        raise ValueError(f"Cannot identify time/lat/lon axes: {variable.dimensions}")
    return np.transpose(values, (time_axis, lat_axis, lon_axis))


def edges(centers):
    centers = np.asarray(centers, dtype=float)
    result = np.empty(centers.size + 1)
    result[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    result[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    result[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return result


def modscag_records(args):
    pattern = re.compile(r"modscag_monthly_mean_(\d{4})(\d{2})\.nc$")
    records = []
    for filename in glob.glob(os.path.join(args.modscag_dir, "*.nc")):
        match = pattern.match(os.path.basename(filename))
        if match:
            year, month = map(int, match.groups())
            if args.start_year <= year <= args.end_year:
                records.append((year, month, filename))
    records.sort()
    if not records:
        raise FileNotFoundError("No MODSCAG files found for the requested years")
    return records


def find_model_grid(args, records):
    folder = os.path.join(args.edir, "postprocess", "output_dir")
    for year, month, _ in records:
        files = sorted(glob.glob(os.path.join(folder, f"{year}{month:02d}??.nc")))
        if files:
            with nc.Dataset(files[0]) as dataset:
                native_lon = np.asarray(dataset.variables["lon"][:], dtype=float)
                native_lat = np.asarray(dataset.variables["lat"][:], dtype=float)
            return (
                np.sort(native_lon), np.sort(native_lat), folder,
                np.argsort(native_lon), np.argsort(native_lat),
            )
    raise FileNotFoundError(f"No matching HydroBlocks files in {folder}")


def read_model_month(folder, year, month, variable, lon_order, lat_order):
    files = sorted(glob.glob(os.path.join(folder, f"{year}{month:02d}??.nc")))
    expected = calendar.monthrange(year, month)[1]
    if len(files) != expected:
        print(f"Warning: {year}-{month:02d} has {len(files)}/{expected} model days")
    total = count = None
    for filename in files:
        with nc.Dataset(filename) as dataset:
            if variable not in dataset.variables:
                raise KeyError(f"{variable!r} not found in {filename}")
            values = time_lat_lon(dataset.variables[variable])
        values = values[:, lat_order, :][:, :, lon_order]
        valid = np.isfinite(values) & (values >= 0.0) & (values <= 1.0)
        if total is None:
            total = np.zeros(values.shape[1:], dtype=float)
            count = np.zeros(values.shape[1:], dtype=np.int64)
        total += np.sum(np.where(valid, values, 0.0), axis=0)
        count += np.sum(valid, axis=0)
    if total is None:
        return None
    result = np.full(total.shape, np.nan)
    np.divide(total, count, out=result, where=count > 0)
    return result


def read_modscag_month(filename, variable, model_lon, model_lat):
    with nc.Dataset(filename) as dataset:
        if variable not in dataset.variables:
            raise KeyError(f"{variable!r} not found in {filename}")
        source = np.ma.asarray(dataset.variables[variable][:]).filled(np.nan).squeeze().astype(float)
        source_lon = np.asarray(dataset.variables["x"][:], dtype=float)
        source_lat = np.asarray(dataset.variables["y"][:], dtype=float)
    if source.ndim != 2:
        raise ValueError(f"Expected 2-D MODSCAG data, found {source.shape}")

    # Rasterio expects row zero to be the northernmost row.
    if source_lon[0] > source_lon[-1]:
        source_lon, source = source_lon[::-1], source[:, ::-1]
    if source_lat[0] < source_lat[-1]:
        source_lat, source = source_lat[::-1], source[::-1, :]
    source[(source < 0.0) | (source > 100.0)] = np.nan
    source /= 100.0

    source_lon_edges = edges(source_lon)
    source_lat_edges = edges(source_lat[::-1])
    model_lon_edges = edges(model_lon)
    model_lat_edges = edges(model_lat)
    source_transform = from_bounds(
        source_lon_edges[0], source_lat_edges[0], source_lon_edges[-1],
        source_lat_edges[-1], source.shape[1], source.shape[0],
    )
    destination_transform = from_bounds(
        model_lon_edges[0], model_lat_edges[0], model_lon_edges[-1],
        model_lat_edges[-1], model_lon.size, model_lat.size,
    )
    north_up = np.full((model_lat.size, model_lon.size), np.nan)
    reproject(
        source, north_up,
        src_transform=source_transform, src_crs="EPSG:4326",
        dst_transform=destination_transform, dst_crs="EPSG:4326",
        src_nodata=np.nan, dst_nodata=np.nan,
        resampling=Resampling.average,
    )
    return np.flipud(north_up)


def process_month(task):
    """Read one matching model/MODSCAG month inside a worker process."""
    (
        year, month, modscag_file, model_folder, model_variable,
        modscag_variable, model_lon, model_lat, lon_order, lat_order,
    ) = task
    model_month = read_model_month(
        model_folder, year, month, model_variable, lon_order, lat_order
    )
    if model_month is None:
        return year, month, None, None
    satellite_month = read_modscag_month(
        modscag_file, modscag_variable, model_lon, model_lat
    )
    return year, month, model_month, satellite_month


def read_seasonal_means(args):
    records = modscag_records(args)
    lon, lat, folder, lon_order, lat_order = find_model_grid(args, records)
    shape = (lat.size, lon.size)
    model_sum = {season: np.zeros(shape) for season in SEASONS}
    model_weight = {season: np.zeros(shape) for season in SEASONS}
    satellite_sum = {season: np.zeros(shape) for season in SEASONS}
    satellite_weight = {season: np.zeros(shape) for season in SEASONS}
    used = 0

    tasks = [
        (
            year, month, filename, folder, args.model_variable,
            args.modscag_variable, lon, lat, lon_order, lat_order,
        )
        for year, month, filename in records
    ]

    def accumulate_months(results):
        nonlocal used
        for number, result in enumerate(results, 1):
            year, month, model_month, satellite_month = result
            if model_month is None:
                print(f"Skipping {year}-{month:02d}: no model files", flush=True)
                continue
            season = next(name for name, months in SEASONS.items() if month in months)
            days = calendar.monthrange(year, month)[1]
            # Use the same valid months for both datasets at each grid cell.
            paired_valid = (
            np.isfinite(model_month)
            & np.isfinite(satellite_month)
            )

            model_sum[season] += np.where(
            paired_valid, model_month * days, 0.0
            )
            satellite_sum[season] += np.where(
            paired_valid, satellite_month * days, 0.0
            )

            model_weight[season] += np.where(paired_valid, days, 0.0)
            satellite_weight[season] += np.where(paired_valid, days, 0.0)
            #satellite_weight[season] += np.where(satellite_valid, days, 0.0)
            used += 1
            print(
                f"Processed {number}/{len(records)}: {year}-{month:02d}",
                flush=True,
            )

    print(
        f"Monthly readers: {args.workers}; matching months: {len(tasks)}",
        flush=True,
    )
    if args.workers == 1:
        accumulate_months(map(process_month, tasks))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            accumulate_months(executor.map(process_month, tasks, chunksize=1))

    if used == 0:
        raise RuntimeError("No matching months were processed")
    model, satellite = {}, {}
    for season in SEASONS:
        model[season] = np.full(shape, np.nan)
        satellite[season] = np.full(shape, np.nan)
        np.divide(model_sum[season], model_weight[season], out=model[season],
                  where=model_weight[season] > 0)
        np.divide(satellite_sum[season], satellite_weight[season], out=satellite[season],
                  where=satellite_weight[season] > 0)
    print(f"Matched months processed: {used}", flush=True)
    return lon, lat, model, satellite


def comparison(model, satellite, threshold):
    valid = np.isfinite(model) & np.isfinite(satellite)
    if threshold > 0:
        valid &= (model >= threshold) | (satellite >= threshold)
    model_valid = model[valid]
    satellite_valid = satellite[valid]
    difference = model_valid - satellite_valid
    if difference.size == 0:
        return valid, difference, np.nan, np.nan, np.nan, np.nan
    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    correlation = np.nan
    if difference.size >= 2 and np.std(model_valid) > 0 and np.std(satellite_valid) > 0:
        correlation = float(np.corrcoef(model_valid, satellite_valid)[0, 1])
    return valid, difference, md, mad, rmse, correlation


def plot_maps(args, lon, lat, model, satellite, output):
    fsno_norm = colors.Normalize(0.0, 1.0)
    differences = [comparison(model[season], satellite[season],
                              args.snow_threshold)[1] for season in SEASONS]
    retained = [values for values in differences if values.size]
    if not retained:
        raise RuntimeError("No paired pixels pass the snow threshold")
    actual_min = min(float(values.min()) for values in retained)
    actual_max = max(float(values.max()) for values in retained)
    limit = args.diff_limit if args.diff_limit is not None else max(
        abs(actual_min), abs(actual_max), 1.0e-6)
    print(f"Retained difference range: {actual_min:.6f} to {actual_max:.6f}; "
          f"map color limits: {-limit:.6f} to {limit:.6f}", flush=True)
    difference_norm = colors.TwoSlopeNorm(
        vmin=-limit, vcenter=0.0, vmax=limit
    )
    figure, axes = plt.subplots(4, 3, figsize=(15, 19), sharex=True, sharey=True)
    for row, season in enumerate(SEASONS):
        valid, _, _, _, _, _ = comparison(
            model[season], satellite[season], args.snow_threshold
        )
        model_map = np.where(valid, model[season], np.nan)
        satellite_map = np.where(valid, satellite[season], np.nan)
        difference_map = np.where(valid, model[season] - satellite[season], np.nan)
        panels = (
            (model_map, "HB-NMP 3d fsno", "Blues", fsno_norm, "FSNO [fraction]"),
            (satellite_map, "MODSCAG fsno", "Blues", fsno_norm, "fsno [fraction]"),
            (difference_map, "3d - MODSCAG", "RdBu_r", difference_norm, "Difference [fraction]"),
        )
        for column, (values, title, cmap, norm, label) in enumerate(panels):
            axis = axes[row, column]
            image = axis.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")
            axis.set_title(f"{season}: {title}")
            axis.set_xlabel("Longitude [°]")
            axis.set_ylabel("Latitude [°]")
            figure.colorbar(image, ax=axis, label=label, shrink=0.86)
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}", flush=True)


def plot_histograms(args, model, satellite, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))
    for axis, season in zip(axes.ravel(), SEASONS):
        _, difference, md, mad, rmse, correlation = comparison(
            model[season], satellite[season], args.snow_threshold
        )
        if difference.size == 0:
            axis.set_title(f"{season}: no paired pixels", fontweight="bold")
            axis.axis("off")
            continue
        axis.hist(difference, bins=args.bins, density=True)
        axis.axvline(md, color="darkred", linestyle="dashed", linewidth=4)
        r_text = f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
        axis.text(
            0.98, 0.98,
            f"MD = {md:.3f}\nMAD = {mad:.3f}\nRMSE = {rmse:.3f}\n"
            f"r = {r_text}\nn = {difference.size:,}",
            transform=axis.transAxes, ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightgray",
                  "edgecolor": "black", "alpha": 0.6},
        )
        axis.set_title(season, fontweight="bold")
        axis.set_xlabel("HB-NMP 3D − MODSCAG fsno")
        axis.set_ylabel("Probability density")
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}", flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("end year must not precede start year")
    if args.diff_limit is not None and args.diff_limit <= 0:
        raise ValueError("diff-limit must be positive")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")
    if not 0 <= args.snow_threshold <= 1:
        raise ValueError("snow-threshold must be between 0 and 1")
    args.edir = os.path.abspath(args.edir)
    args.modscag_dir = os.path.abspath(args.modscag_dir)
    outdir = args.outdir or os.path.join(args.edir, "validation_plots", "3d_vs_MODSCAG")
    os.makedirs(outdir, exist_ok=True)
    lon, lat, model, satellite = read_seasonal_means(args)
    stem = f"3d_fsno_vs_MODSCAG_{args.start_year}_{args.end_year}"
    plot_maps(args, lon, lat, model, satellite,
              os.path.join(outdir, f"seasonal_maps_{stem}.png"))
    plot_histograms(args, model, satellite,
                    os.path.join(outdir, f"seasonal_histograms_{stem}.png"))


if __name__ == "__main__":
    main()
