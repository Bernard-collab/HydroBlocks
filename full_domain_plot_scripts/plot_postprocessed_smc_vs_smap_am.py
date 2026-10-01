#!/usr/bin/env python3
"""Compare postprocessed HydroBlocks surface SMC with NSIDC-0779 SMAP (6 AM only).

SMAP NSIDC-0779 band 1 is the descending overpass, approximately 06:00 local
solar time. For every available SMAP day, the script linearly interpolates
the 3-hourly HydroBlocks SMC to 6 AM local-solar time at each longitude.
SMAP is spatially averaged from its 1-km EPSG:6933 grid onto the exact
HydroBlocks postprocessed grid. Seasonal means use only paired, valid daily
observations.

Outputs (two figures and one CSV):
  seasonal_06AM_HB_SMAP_maps_<years>.png
  seasonal_06AM_HB_minus_SMAP_histograms_<years>.png
  seasonal_06AM_HB_minus_SMAP_metrics_<years>.csv
"""

import argparse
import csv
import glob
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timedelta
from functools import lru_cache

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.windows import from_bounds as window_from_bounds
from rasterio.warp import Resampling, reproject, transform_bounds

# Larger, publication-readable text (matplotlib default font). Titles are bold.
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

SMAP_BAND = 1          # descending overpass
SOLAR_HOUR = 6.0        # local solar time, hours
PASS_DESCRIPTION = "6 AM descending"

EPOCH = datetime(1970, 1, 1)
WORKER_STATE = {}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edir", required=True,
                        help="HydroBlocks 3D experiment directory")
    parser.add_argument("--smap-dir", required=True,
                        help="Directory containing daily NSIDC-0779 GeoTIFFs")
    parser.add_argument("--start-year", type=int, default=2016)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--variable", default="smc")
    parser.add_argument("--outdir", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--workers", type=int, default=1)
    return parser.parse_args()


def datetime_hours(value):
    """Hours since EPOCH for a Python datetime."""
    value = datetime(
        value.year, value.month, value.day,
        value.hour, value.minute, value.second,
        getattr(value, "microsecond", 0),
    )
    return (value - EPOCH).total_seconds() / 3600.0


def axis_of(dimensions, candidates):
    names = [name.lower() for name in dimensions]
    for candidate in candidates:
        if candidate in names:
            return names.index(candidate)
    return None


def read_time_lat_lon(variable):
    """Read a NetCDF variable and return it in (time, lat, lon) order."""
    values = np.ma.asarray(variable[:]).filled(np.nan).astype(np.float32)
    dimensions = variable.dimensions
    time_axis = axis_of(dimensions, ("time", "t"))
    lat_axis = axis_of(dimensions, ("lat", "latitude", "y"))
    lon_axis = axis_of(dimensions, ("lon", "longitude", "x"))
    if None in (time_axis, lat_axis, lon_axis):
        raise ValueError(f"Cannot identify time/lat/lon axes for {dimensions}")
    return np.transpose(values, (time_axis, lat_axis, lon_axis))


def discover_hb_files(folder, start_year, end_year):
    paths = sorted(glob.glob(os.path.join(folder, "????????.nc")))
    result = {}
    for path in paths:
        try:
            date = datetime.strptime(os.path.basename(path)[:8], "%Y%m%d")
        except ValueError:
            continue
        if start_year <= date.year <= end_year:
            result[date.date()] = path
    if not result:
        raise FileNotFoundError(f"No HydroBlocks daily NetCDF files in {folder}")
    return result


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


def initialize_hb_grid(sample_file, variable_name):
    with nc.Dataset(sample_file) as dataset:
        lon = np.asarray(dataset.variables["lon"][:], dtype=float)
        lat = np.asarray(dataset.variables["lat"][:], dtype=float)
        if variable_name not in dataset.variables:
            raise KeyError(f"{variable_name!r} not found in {sample_file}")
        dimensions = dataset.variables[variable_name].dimensions
        if axis_of(dimensions, ("soil", "layer", "nsoil")) is not None:
            raise ValueError(
                "Postprocessed SMC unexpectedly contains a soil-layer axis; "
                "select the surface layer explicitly before comparison."
            )

    lon_order = np.argsort(lon)
    lat_order = np.argsort(lat)
    lon = lon[lon_order]
    lat = lat[lat_order]
    lon_edges = coordinate_edges(lon)
    lat_edges = coordinate_edges(lat)
    transform = from_bounds(
        lon_edges[0], lat_edges[0], lon_edges[-1], lat_edges[-1],
        len(lon), len(lat),
    )
    return lon, lat, lon_order, lat_order, transform


def coordinate_edges(centers):
    centers = np.asarray(centers, dtype=float)
    if centers.size < 2:
        raise ValueError("At least two coordinate centers are required")
    edges = np.empty(centers.size + 1, dtype=float)
    edges[1:-1] = 0.5 * (centers[:-1] + centers[1:])
    edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
    edges[-1] = centers[-1] + 0.5 * (centers[-1] - centers[-2])
    return edges


def build_hb_reader(files, variable_name, lon_order, lat_order):
    @lru_cache(maxsize=3)
    def read_one(date_key):
        path = files.get(date_key)
        if path is None:
            return None
        with nc.Dataset(path) as dataset:
            variable = dataset.variables[variable_name]
            data = read_time_lat_lon(variable)
            data = data[:, lat_order, :][:, :, lon_order]
            data[(data < 0.0) | (data > 1.0)] = np.nan

        # The eight postprocessed slots are the original 3-hourly model
        # outputs: 00, 03, 06, 09, 12, 15, 18 and 21 UTC. The stored t
        # coordinate incorrectly appears as consecutive hours 0 through 7.
        start = datetime(date_key.year, date_key.month, date_key.day)
        hours = np.array(
            [datetime_hours(start + timedelta(hours=3 * index))
             for index in range(data.shape[0])],
            dtype=float,
        )
        return hours, data

    return read_one


def sample_hb_local_solar(date, lon, read_hb):
    """Interpolate HB SMC to 6 AM local-solar time, independently by longitude."""
    chunks = []
    for offset in (-1, 0, 1):
        item = read_hb((date + timedelta(days=offset)).date())
        if item is not None:
            chunks.append(item)
    if not chunks:
        return None

    times = np.concatenate([item[0] for item in chunks])
    data = np.concatenate([item[1] for item in chunks], axis=0)
    order = np.argsort(times)
    times = times[order]
    data = data[order]

    midnight = datetime_hours(datetime(date.year, date.month, date.day))
    # Local solar time = UTC + longitude/15; therefore UTC = LST - lon/15.
    target = midnight + SOLAR_HOUR - lon / 15.0
    result = np.full(data.shape[1:], np.nan, dtype=np.float32)

    for column, target_hour in enumerate(target):
        upper = int(np.searchsorted(times, target_hour, side="left"))
        if upper == 0 or upper >= len(times):
            continue
        lower = upper - 1
        interval = times[upper] - times[lower]
        if interval <= 0.0 or interval > 3.01:
            continue
        weight = (target_hour - times[lower]) / interval
        low = data[lower, :, column]
        high = data[upper, :, column]
        valid = np.isfinite(low) & np.isfinite(high)
        result[valid, column] = (
            low[valid] * (1.0 - weight) + high[valid] * weight
        )
    return result


def read_smap_on_hb_grid(path, hb_transform, hb_shape):
    """Read only the HB-domain portion and area-average it onto the HB grid."""
    height, width = hb_shape
    west = hb_transform.c
    north = hb_transform.f
    east = west + hb_transform.a * width
    south = north + hb_transform.e * height

    with rasterio.open(path) as source_dataset:
        if source_dataset.count < SMAP_BAND:
            raise ValueError(f"{path} does not contain band {SMAP_BAND}")
        source_bounds = transform_bounds(
            "EPSG:4326", source_dataset.crs,
            west, south, east, north, densify_pts=21,
        )
        window = window_from_bounds(*source_bounds, transform=source_dataset.transform)
        window = window.round_offsets().round_lengths()
        source = source_dataset.read(
            SMAP_BAND, window=window, boundless=True,
            masked=True, fill_value=source_dataset.nodata,
        ).filled(np.nan).astype(np.float32)
        source[(~np.isfinite(source)) | (source <= 0.0) | (source > 1.0)] = np.nan

        destination_north_up = np.full(hb_shape, np.nan, dtype=np.float32)
        reproject(
            source=source,
            destination=destination_north_up,
            src_transform=source_dataset.window_transform(window),
            src_crs=source_dataset.crs,
            src_nodata=np.nan,
            dst_transform=hb_transform,
            dst_crs="EPSG:4326",
            dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    # NetCDF latitude is sorted south-to-north; raster destination is north-up.
    return np.flipud(destination_north_up)


def empty_accumulators(shape):
    return {
        season: {
            "hb_sum": np.zeros(shape, dtype=np.float64),
            "smap_sum": np.zeros(shape, dtype=np.float64),
            "count": np.zeros(shape, dtype=np.uint32),
        }
        for season in SEASONS
    }


def accumulate_pair(accumulator, hb, smap):
    valid = np.isfinite(hb) & np.isfinite(smap)
    accumulator["hb_sum"] += np.where(valid, hb, 0.0)
    accumulator["smap_sum"] += np.where(valid, smap, 0.0)
    accumulator["count"] += valid.astype(np.uint32)


def initialize_worker(hb_files, variable_name, lon, lon_order, lat_order,
                      hb_transform, hb_shape):
    """Initialize data shared by all date chunks handled by one worker."""
    WORKER_STATE["lon"] = lon
    WORKER_STATE["hb_transform"] = hb_transform
    WORKER_STATE["hb_shape"] = hb_shape
    WORKER_STATE["read_hb"] = build_hb_reader(
        hb_files, variable_name, lon_order, lat_order
    )


def process_chunk(items):
    """Process a chunk of SMAP dates and return seasonal partial sums."""
    accumulators = empty_accumulators(WORKER_STATE["hb_shape"])
    used = 0

    for date, smap_path in items:
        hb = sample_hb_local_solar(
            date, WORKER_STATE["lon"], WORKER_STATE["read_hb"]
        )
        if hb is None:
            continue

        smap = read_smap_on_hb_grid(
            smap_path, WORKER_STATE["hb_transform"], WORKER_STATE["hb_shape"]
        )
        if not np.any(np.isfinite(hb) & np.isfinite(smap)):
            continue

        season = next(
            name for name, months in SEASONS.items() if date.month in months
        )
        accumulate_pair(accumulators[season], hb, smap)
        used += 1

    return accumulators, used


def merge_accumulators(total, partial):
    for season in SEASONS:
        for key in ("hb_sum", "smap_sum", "count"):
            total[season][key] += partial[season][key]


def finalize_maps(accumulators):
    maps = {}
    for season in SEASONS:
        item = accumulators[season]
        hb = np.full(item["count"].shape, np.nan, dtype=float)
        smap = np.full(item["count"].shape, np.nan, dtype=float)
        np.divide(item["hb_sum"], item["count"], out=hb,
                  where=item["count"] > 0)
        np.divide(item["smap_sum"], item["count"], out=smap,
                  where=item["count"] > 0)
        maps[season] = {
            "hb": hb,
            "smap": smap,
            "difference": hb - smap,
            "daily_count": item["count"],
        }
    return maps


def paired_metrics(hb, smap):
    valid = np.isfinite(hb) & np.isfinite(smap)
    hb_values = hb[valid]
    smap_values = smap[valid]
    difference = hb_values - smap_values
    n = difference.size
    if n == 0:
        return difference, np.nan, np.nan, np.nan, np.nan, 0
    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    if n >= 2 and np.std(hb_values) > 0 and np.std(smap_values) > 0:
        correlation = float(np.corrcoef(hb_values, smap_values)[0, 1])
    else:
        correlation = np.nan
    return difference, md, mad, rmse, correlation, n


def full_limits(all_maps):
    moisture = []
    differences = []
    for season in SEASONS:
        item = all_maps[season]
        for key in ("hb", "smap"):
            values = item[key][np.isfinite(item[key])]
            if values.size:
                moisture.append(values)
        values = item["difference"][np.isfinite(item["difference"])]
        if values.size:
            differences.append(np.abs(values))
    if not moisture or not differences:
        raise RuntimeError("No paired HydroBlocks-SMAP pixels were available")
    all_moisture = np.concatenate(moisture)
    maximum_difference = max(float(np.max(np.concatenate(differences))), 1.0e-6)
    return float(np.min(all_moisture)), float(np.max(all_moisture)), maximum_difference


def plot_maps(lon, lat, all_maps, output, dpi):
    minimum, maximum, difference_limit = full_limits(all_maps)
    moisture_norm = colors.Normalize(vmin=minimum, vmax=maximum)
    difference_norm = colors.TwoSlopeNorm(
        vmin=-difference_limit, vcenter=0.0, vmax=difference_limit
    )
    moisture_cmap = plt.cm.viridis.copy()
    difference_cmap = plt.cm.RdBu_r.copy()
    moisture_cmap.set_bad("white")
    difference_cmap.set_bad("white")

    fig, axes = plt.subplots(4, 3, figsize=(15, 19), sharex=True, sharey=True)
    for row, season in enumerate(SEASONS):
        item = all_maps[season]
        panels = (
            (item["hb"], "HB-NMP 3D SMC", moisture_cmap,
             moisture_norm, "Soil moisture [m³/m³]"),
            (item["smap"], "SMAP SMC", moisture_cmap,
             moisture_norm, "Soil moisture [m³/m³]"),
            (item["difference"], "HB-NMP 3D − SMAP", difference_cmap,
             difference_norm, "ΔSMC [m³/m³]"),
        )
        for column, (values, title, cmap, norm, label) in enumerate(panels):
            axis = axes[row, column]
            image = axis.pcolormesh(
                lon, lat, values, cmap=cmap, norm=norm, shading="auto"
            )
            axis.set_title(f"{season}: {title}", fontweight="bold")
            axis.set_xlabel("Longitude")
            axis.set_ylabel("Latitude")
            fig.colorbar(image, ax=axis, label=label, shrink=0.86)

    fig.suptitle(f"Seasonal Surface Soil Moisture — {PASS_DESCRIPTION}", fontsize=20, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def plot_histograms(all_maps, output, dpi, bins):
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    axes = axes.ravel()
    metric_rows = []

    for axis, season in zip(axes, SEASONS):
        item = all_maps[season]
        difference, md, mad, rmse, correlation, n = paired_metrics(
            item["hb"], item["smap"]
        )
        metric_rows.append((season, md, mad, rmse, correlation, n))
        if n == 0:
            axis.set_title(f"{season}: no paired pixels", fontweight="bold")
            axis.axis("off")
            continue
        axis.hist(difference, bins=bins, density=True)
        axis.axvline(md, color="darkred", linestyle="dashed", linewidth=4)
        correlation_text = f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
        axis.text(
            0.98, 0.98,
            f"MD = {md:.3f} m³/m³\n"
            f"MAD = {mad:.3f} m³/m³\n"
            f"RMSE = {rmse:.3f} m³/m³\n"
            f"r = {correlation_text}\n"
            f"n = {n:,}",
            transform=axis.transAxes, ha="right", va="top",
            fontsize=10
            bbox={"boxstyle": "round", "facecolor": "lightgray",
                  "edgecolor": "black", "alpha": 0.6},
        )
        axis.set_title(season, fontweight="bold")
        axis.set_xlabel("HB-NMP 3D − SMAP SMC [m³/m³]")
        axis.set_ylabel("Probability density")

    fig.suptitle(f"Seasonal SMC Difference Distributions — {PASS_DESCRIPTION}", fontsize=19, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)
    return metric_rows


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("end-year must not precede start-year")
    if args.workers < 1:
        raise ValueError("workers must be at least 1")

    args.edir = os.path.abspath(args.edir)
    args.smap_dir = os.path.abspath(args.smap_dir)
    output_directory = args.outdir or os.path.join(
        args.edir, "validation_plots", "SMAP_comparison"
    )
    os.makedirs(output_directory, exist_ok=True)

    hb_folder = os.path.join(args.edir, "postprocess", "output_dir")
    hb_files = discover_hb_files(hb_folder, args.start_year, args.end_year)
    smap_files = discover_smap_files(args.smap_dir, args.start_year, args.end_year)
    lon, lat, lon_order, lat_order, hb_transform = initialize_hb_grid(
        next(iter(hb_files.values())), args.variable
    )
    accumulators = empty_accumulators((len(lat), len(lon)))

    print(f"HydroBlocks daily files: {len(hb_files)}", flush=True)
    print(f"SMAP daily files: {len(smap_files)}", flush=True)
    print(f"Comparison grid: {len(lat)} x {len(lon)}", flush=True)
    workers = min(args.workers, len(smap_files))
    chunk_size = (len(smap_files) + workers - 1) // workers
    chunks = [
        smap_files[start:start + chunk_size]
        for start in range(0, len(smap_files), chunk_size)
    ]
    used = 0

    print(f"Workers: {workers}", flush=True)
    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=initialize_worker,
        initargs=(
            hb_files, args.variable, lon, lon_order, lat_order,
            hb_transform, (len(lat), len(lon)),
        ),
    ) as executor:
        futures = [executor.submit(process_chunk, chunk) for chunk in chunks]
        for number, future in enumerate(as_completed(futures), 1):
            partial, partial_used = future.result()
            merge_accumulators(accumulators, partial)
            used += partial_used
            print(f"Worker chunks completed: {number}/{len(chunks)}", flush=True)

    all_maps = finalize_maps(accumulators)
    year_label = f"{args.start_year}_{args.end_year}"

    map_output = os.path.join(
        output_directory, f"seasonal_06AM_HB_SMAP_maps_{year_label}.png"
    )
    histogram_output = os.path.join(
        output_directory, f"seasonal_06AM_HB_minus_SMAP_histograms_{year_label}.png"
    )
    plot_maps(lon, lat, all_maps, map_output, args.dpi)
    metrics = plot_histograms(all_maps, histogram_output, args.dpi, args.bins)

    csv_rows = [
        {
            "pass": "06AM",
            "description": PASS_DESCRIPTION,
            "season": season,
            "MD_m3_m3": md,
            "MAD_m3_m3": mad,
            "RMSE_m3_m3": rmse,
            "correlation_r": correlation,
            "valid_spatial_pixels_n": n,
            "paired_daily_files_used": used,
        }
        for season, md, mad, rmse, correlation, n in metrics
    ]

    csv_output = os.path.join(
        output_directory, f"seasonal_06AM_HB_minus_SMAP_metrics_{year_label}.csv"
    )
    with open(csv_output, "w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    print(f"Saved: {csv_output}", flush=True)
    print(f"Paired daily files used: {used}", flush=True)


if __name__ == "__main__":
    main()