#!/usr/bin/env python3
"""Seasonal daytime/nighttime HydroBlocks TRAD versus MODIS LST.

Creates two 4x3 figures (one daytime and one nighttime). Columns are
HydroBlocks, MODIS, and HydroBlocks minus MODIS; rows are the four seasons.

MODIS GeoTIFF convention used here:
  band 1 = daytime LST, band 2 = nighttime LST.

For postprocessed HydroBlocks files without COSZ, model samples are selected
at 18 UTC (day) and 06 UTC (night), the closest 3-hourly samples to the usual
MODIS Terra local overpasses in the Upper Colorado region. If postprocessed
COSZ exists, --selection auto uses COSZ > 0.3 for day and COSZ < 0 for night.

how to run or submit;
python plot_postprocessed_trad_vs_modis_day_night.py \
  --edir "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/YOUR_EXPERIMENT" \
  --modis-dir "/scratch/alpine/battobrah@xsede.org/HB_NMP_full_domain_comparison_data/MODIS_LST_2014_2024" \
  --start-year 2014 \
  --end-year 2024 \
  --modis-units celsius \
  --selection auto \
  --outdir "/scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/validation_plots/Surface_Temp_Plots"

"""

import argparse
import calendar
import glob
import os
import re
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edir", required=True, help="HydroBlocks experiment directory")
    p.add_argument("--modis-dir", required=True, help="Monthly MODIS GeoTIFF directory")
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--variable", default="trad")
    p.add_argument(
        "--selection", choices=("auto", "cosz", "utc"), default="auto",
        help="Use COSZ when available or select fixed UTC hours",
    )
    p.add_argument("--day-cosz", type=float, default=0.3)
    p.add_argument("--day-hour-utc", type=float, default=18.0)
    p.add_argument("--night-hour-utc", type=float, default=6.0)
    p.add_argument(
        "--modis-units", choices=("celsius", "kelvin", "native", "auto"),
        default="celsius", help="native means MOD11 values with scale factor 0.02",
    )
    p.add_argument("--outdir", default=None)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def axis_of(dimensions, exact_names):
    names = [x.lower() for x in dimensions]
    for candidate in exact_names:
        if candidate in names:
            return names.index(candidate)
    return None


def time_lat_lon(variable):
    data = np.ma.asarray(variable[:]).filled(np.nan).astype(float)
    dims = variable.dimensions
    ta = axis_of(dims, ("time", "t"))
    ya = axis_of(dims, ("lat", "latitude", "y"))
    xa = axis_of(dims, ("lon", "longitude", "x"))
    if None in (ta, ya, xa):
        raise ValueError(f"Cannot identify time/lat/lon axes: {dims}")
    return np.transpose(data, (ta, ya, xa))


def accumulate(store_sum, store_count, season, data, valid):
    part_sum = np.sum(np.where(valid, data, 0.0), axis=0)
    part_count = np.sum(valid, axis=0)
    if store_sum[season] is None:
        store_sum[season] = np.zeros(part_sum.shape, dtype=float)
        store_count[season] = np.zeros(part_count.shape, dtype=np.int64)
    store_sum[season] += part_sum
    store_count[season] += part_count


def read_model(args):
    folder = os.path.join(os.path.abspath(args.edir), "postprocess", "output_dir")
    files = []
    for year in range(args.start_year, args.end_year + 1):
        files += sorted(glob.glob(os.path.join(folder, f"{year}????.nc")))
    if not files:
        raise FileNotFoundError(f"No postprocessed daily files in {folder}")

    sums = {period: {s: None for s in SEASONS} for period in ("day", "night")}
    counts = {period: {s: None for s in SEASONS} for period in ("day", "night")}
    lon = lat = None
    method = None

    for number, filename in enumerate(files, 1):
        date = datetime.strptime(os.path.basename(filename)[:8], "%Y%m%d")
        season = next(s for s, months in SEASONS.items() if date.month in months)
        with nc.Dataset(filename) as ds:
            if lon is None:
                lon = np.asarray(ds.variables["lon"][:], dtype=float)
                lat = np.asarray(ds.variables["lat"][:], dtype=float)
            trad = time_lat_lon(ds.variables[args.variable])
            has_cosz = "cosz" in ds.variables
            use_cosz = args.selection == "cosz" or (
                args.selection == "auto" and has_cosz
            )
            if args.selection == "cosz" and not has_cosz:
                raise KeyError("--selection cosz requested, but cosz is not postprocessed")

            finite = np.isfinite(trad) & (trad > 100.0) & (trad < 400.0)
            if use_cosz:
                cosz = time_lat_lon(ds.variables["cosz"])
                day_valid = finite & np.isfinite(cosz) & (cosz > args.day_cosz)
                night_valid = finite & np.isfinite(cosz) & (cosz < 0.0)
                method = f"COSZ > {args.day_cosz:g} / COSZ < 0"
            else:
                hours = np.arange(trad.shape[0], dtype=float) * 24.0 / trad.shape[0]
                day_index = int(np.argmin(np.abs(hours - args.day_hour_utc)))
                night_index = int(np.argmin(np.abs(hours - args.night_hour_utc)))
                day_valid = np.zeros_like(finite)
                night_valid = np.zeros_like(finite)
                day_valid[day_index] = finite[day_index]
                night_valid[night_index] = finite[night_index]
                method = f"{hours[day_index]:g} UTC / {hours[night_index]:g} UTC"

        accumulate(sums["day"], counts["day"], season, trad, day_valid)
        accumulate(sums["night"], counts["night"], season, trad, night_valid)
        if number % 100 == 0 or number == len(files):
            print(f"HydroBlocks {number}/{len(files)}", flush=True)

    xo, yo = np.argsort(lon), np.argsort(lat)
    maps = {period: {} for period in ("day", "night")}
    for period in maps:
        for season in SEASONS:
            result = np.full(sums[period][season].shape, np.nan)
            np.divide(
                sums[period][season], counts[period][season], out=result,
                where=counts[period][season] > 0,
            )
            maps[period][season] = result[np.ix_(yo, xo)] - 273.15
    print(f"Model day/night selection: {method}", flush=True)
    return lon[xo], lat[yo], maps


def edges(centers):
    centers = np.asarray(centers)
    result = np.empty(centers.size + 1)
    result[1:-1] = (centers[:-1] + centers[1:]) / 2
    result[0] = centers[0] - (centers[1] - centers[0]) / 2
    result[-1] = centers[-1] + (centers[-1] - centers[-2]) / 2
    return result


def to_celsius(data, mode):
    if mode == "auto":
        median = np.nanmedian(data)
        mode = "native" if median > 1000 else ("kelvin" if median > 100 else "celsius")
        print(f"Detected MODIS units: {mode}", flush=True)
    if mode == "native":
        return data * 0.02 - 273.15
    if mode == "kelvin":
        return data - 273.15
    return data


def read_modis(args, lon, lat):
    lon_e, lat_e = edges(lon), edges(lat)
    transform = from_bounds(lon_e[0], lat_e[0], lon_e[-1], lat_e[-1], len(lon), len(lat))
    sums = {
        p: {s: np.zeros((len(lat), len(lon))) for s in SEASONS}
        for p in ("day", "night")
    }
    weights = {
        p: {s: np.zeros((len(lat), len(lon))) for s in SEASONS}
        for p in ("day", "night")
    }
    pattern = re.compile(r"MOD11A1_monthly_mean_(\d{4})_(\d{2})\.tif$")
    rows = []
    for filename in glob.glob(os.path.join(args.modis_dir, "*.tif")):
        match = pattern.match(os.path.basename(filename))
        if match:
            year, month = map(int, match.groups())
            if args.start_year <= year <= args.end_year:
                rows.append((filename, year, month))
    rows.sort(key=lambda x: (x[1], x[2]))
    if not rows:
        raise FileNotFoundError("No matching MODIS YYYY_MM GeoTIFF files")

    detected_mode = args.modis_units
    for number, (filename, year, month) in enumerate(rows, 1):
        season = next(s for s, months in SEASONS.items() if month in months)
        days = calendar.monthrange(year, month)[1]
        with rasterio.open(filename) as src:
            if src.count < 2:
                raise ValueError(f"{filename} has {src.count} band(s); day and night require 2")
            for band, period in ((1, "day"), (2, "night")):
                source = src.read(band, masked=True).filled(np.nan).astype(float)
                if number == 1 and band == 1 and detected_mode == "auto":
                    med = np.nanmedian(source)
                    detected_mode = "native" if med > 1000 else (
                        "kelvin" if med > 100 else "celsius"
                    )
                    print(f"Detected MODIS units: {detected_mode}", flush=True)
                source = to_celsius(source, detected_mode)
                destination = np.full((len(lat), len(lon)), np.nan)
                reproject(
                    source, destination,
                    src_transform=src.transform, src_crs=src.crs,
                    dst_transform=transform, dst_crs="EPSG:4326",
                    src_nodata=np.nan, dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )
                good = np.isfinite(destination) & (destination > -100) & (destination < 100)
                sums[period][season] += np.where(good, destination * days, 0)
                weights[period][season] += np.where(good, days, 0)
        if number % 12 == 0 or number == len(rows):
            print(f"MODIS {number}/{len(rows)}", flush=True)

    maps = {p: {} for p in ("day", "night")}
    for period in maps:
        for season in SEASONS:
            north_up = np.full((len(lat), len(lon)), np.nan)
            np.divide(sums[period][season], weights[period][season], out=north_up,
                      where=weights[period][season] > 0)
            maps[period][season] = np.flipud(north_up)
    return maps


def plot_period(period, lon, lat, model, modis, output, dpi):
    differences = {}
    temperatures = []
    for season in SEASONS:
        common = np.isfinite(model[season]) & np.isfinite(modis[season])
        difference = np.full(model[season].shape, np.nan)
        difference[common] = model[season][common] - modis[season][common]
        differences[season] = difference
        temperatures += [model[season][common], modis[season][common]]
    t = np.concatenate([x for x in temperatures if x.size])
    d = np.concatenate([x[np.isfinite(x)] for x in differences.values()])
    tmin, tmax = np.percentile(t, (1, 99))
    dmax = max(float(np.percentile(np.abs(d), 99)), 0.1)
    tnorm = colors.Normalize(tmin, tmax)
    dnorm = colors.TwoSlopeNorm(vmin=-dmax, vcenter=0, vmax=dmax)

    fig, axes = plt.subplots(4, 3, figsize=(15, 19), sharex=True, sharey=True)
    for row, season in enumerate(SEASONS):
        panels = (
            (model[season], "HydroBlocks TRAD", "RdYlBu_r", tnorm, "LST [°C]"),
            (modis[season], "MODIS LST", "RdYlBu_r", tnorm, "LST [°C]"),
            (differences[season], "HydroBlocks − MODIS", "RdBu_r", dnorm, "ΔLST [°C]"),
        )
        for col, (values, name, cmap, norm, label) in enumerate(panels):
            ax = axes[row, col]
            im = ax.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")
            ax.set_title(f"{season}: {name}")
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            fig.colorbar(im, ax=ax, label=label, shrink=0.86)
    fig.suptitle(f"Seasonal {period.capitalize()} Land Surface Temperature", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.985))
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def comparison_metrics(model_values, modis_values):
    """Return paired-pixel differences and seasonal validation metrics."""
    valid = np.isfinite(model_values) & np.isfinite(modis_values)
    model_valid = np.asarray(model_values[valid], dtype=float)
    modis_valid = np.asarray(modis_values[valid], dtype=float)
    difference = model_valid - modis_valid
    n_valid = difference.size

    if n_valid == 0:
        return difference, np.nan, np.nan, np.nan, np.nan, 0

    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))

    # Pearson spatial correlation between the paired seasonal maps.
    if (
        n_valid >= 2
        and np.std(model_valid) > 0.0
        and np.std(modis_valid) > 0.0
    ):
        correlation = float(np.corrcoef(model_valid, modis_valid)[0, 1])
    else:
        correlation = np.nan

    return difference, md, mad, rmse, correlation, n_valid


def plot_histograms(model, modis, output, dpi):
    """Save one 2-row x 4-column day/night seasonal histogram figure."""
    results = {}
    all_differences = []

    for period in ("day", "night"):
        for season in SEASONS:
            result = comparison_metrics(
                model[period][season], modis[period][season]
            )
            results[(period, season)] = result
            if result[0].size:
                all_differences.append(result[0])

    if not all_differences:
        raise RuntimeError("No valid daytime or nighttime model-MODIS pixel pairs")

    # Four seasons across columns; daytime above nighttime.
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    for row, period in enumerate(("day", "night")):
        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            difference, md, mad, rmse, correlation, n_valid = results[
                (period, season)
            ]

            if n_valid == 0:
                axis.set_title(f"{season}: no common pixels")
                axis.axis("off")
                continue

            axis.hist(
                difference,
                bins=50,
                density=True,
            )
            axis.axvline(
                md,
                color="darkred",
                linestyle="dashed",
                linewidth=4,
            )

            correlation_text = (
                f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
            )
            axis.text(
                0.98,
                0.98,
                (
                    f"MD = {md:.2f} °C\n"
                    f"MAD = {mad:.2f} °C\n"
                    f"RMSE = {rmse:.2f} °C\n"
                    f"r = {correlation_text}\n"
                    f"n = {n_valid:,}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightgray",
                    "edgecolor": "black",
                    "alpha": 0.6,
                },
            )
            axis.set_title(f"{season} – {period.capitalize()}")
            axis.set_xlabel(
                f"HydroBlocks {period} LST - MODIS {period} LST [°C]"
            )
            if column == 0:
                axis.set_ylabel("Probability density")

    fig.suptitle(
        "Seasonal Daytime and Nighttime LST Difference Distributions",
        fontsize=15,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("end year must not precede start year")
    args.edir = os.path.abspath(args.edir)
    args.modis_dir = os.path.abspath(args.modis_dir)
    outdir = args.outdir or os.path.join(args.edir, "validation_plots")
    os.makedirs(outdir, exist_ok=True)
    lon, lat, model = read_model(args)
    modis = read_modis(args, lon, lat)
    for period in ("day", "night"):
        map_output = os.path.join(
            outdir,
            f"seasonal_{period}_TRAD_MODIS_difference_"
            f"{args.start_year}_{args.end_year}.png",
        )
        plot_period(
            period,
            lon,
            lat,
            model[period],
            modis[period],
            map_output,
            args.dpi,
        )

    histogram_output = os.path.join(
        outdir,
        f"seasonal_day_night_TRAD_MODIS_histograms_"
        f"{args.start_year}_{args.end_year}.png",
    )
    plot_histograms(model, modis, histogram_output, args.dpi)


if __name__ == "__main__":
    main()
