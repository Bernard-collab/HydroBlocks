#!/usr/bin/env python3
"""Seasonal HB-NMP TRAD versus monthly MODIS day/night LST.
Model temperatures are interpolated to nominal local solar times.
Both datasets use the same valid months and month-length weights.
Assumptions:
  * Model TRAD is in kelvin.
  * MODIS band 1 is daytime and band 2 is nighttime.
  * MODIS units are specified explicitly.
  * Monthly MODIS files have already undergone appropriate quality control.
Monthly files cannot identify the satellite's valid individual days.
"""
import argparse
import calendar
import csv
import glob
import os
import re
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
from rasterio.warp import Resampling, reproject
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})
SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}
PERIODS = ("day", "night")
def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edir", required=True)
    p.add_argument("--modis-dir", required=True)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--variable", default="trad")
    p.add_argument("--modis-units",
                   choices=("celsius", "kelvin", "native"),
                   required=True)
    p.add_argument("--day-local-hour", type=float, default=10.5)
    p.add_argument("--night-local-hour", type=float, default=22.5)
    p.add_argument(
        "--assume-model-step-hours", type=float, default=None,
        help="Override the time coordinate: assume evenly spaced "
             "records starting at 00:00 UTC, e.g. 3 for three-hourly data",
    )
    p.add_argument(
        "--minimum-day-coverage", type=float, default=0.9,
        help="Required fraction of valid daily interpolated model values "
             "at each cell within a month",
    )
    p.add_argument("--bins", type=int, default=50)
    p.add_argument("--dpi", type=int, default=400)
    p.add_argument("--outdir", required=True)
    args = p.parse_args()
    if args.end_year < args.start_year:
        p.error("end-year must not precede start-year")
    if not 0 < args.minimum_day_coverage <= 1:
        p.error("minimum-day-coverage must be in (0, 1]")
    if not all(0 <= h < 24 for h in
               (args.day_local_hour, args.night_local_hour)):
        p.error("local hours must be in [0, 24)")
    if (args.assume_model_step_hours is not None
            and args.assume_model_step_hours <= 0):
        p.error("assumed model timestep must be positive")
    if args.bins < 1:
        p.error("bins must be positive")
    return args
def read_float(variable):
    return np.ma.asarray(variable[:], dtype=float).filled(np.nan)
def read_tyx(variable):
    dims = [d.lower() for d in variable.dimensions]
    def axis(names):
        matches = [i for i, d in enumerate(dims) if d in names]
        if len(matches) != 1:
            raise ValueError(
                f"Cannot identify axes: {variable.dimensions}"
            )
        return matches[0]
    axes = (
        axis(("time", "t")),
        axis(("lat", "latitude", "y")),
        axis(("lon", "longitude", "x")),
    )
    values = read_float(variable)
    if values.ndim != 3:
        raise ValueError("Expected a three-dimensional model variable")
    return np.transpose(values, axes)
def check_grid(values, name):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size < 2:
        raise ValueError(f"{name} must be a 1-D coordinate")
    spacing = np.diff(values)
    if not np.all(np.isfinite(values)) or not np.all(spacing > 0):
        raise ValueError(f"{name} must be finite and strictly increasing")
    if not np.allclose(spacing, spacing[0], rtol=1e-4, atol=1e-7):
        raise ValueError(f"{name} must be regularly spaced")
    return values
def edges(values):
    spacing = values[1] - values[0]
    return values[0] - spacing / 2, values[-1] + spacing / 2
def satellite_records(args):
    pattern = re.compile(
        r"MOD11A1_monthly_mean_(\d{4})_(\d{2})\.tif$"
    )
    records = []
    for path in glob.glob(os.path.join(args.modis_dir, "*.tif")):
        match = pattern.fullmatch(os.path.basename(path))
        if match:
            year, month = map(int, match.groups())
            if args.start_year <= year <= args.end_year:
                if not 1 <= month <= 12:
                    raise ValueError(f"Invalid month in {path}")
                records.append((year, month, path))
    records.sort()
    if not records:
        raise FileNotFoundError("No matching monthly MODIS GeoTIFFs")
    return records
def make_model_reader(args):
    folder = os.path.join(args.edir, "postprocess", "output_dir")
    paths = {}
    for path in glob.glob(os.path.join(folder, "*.nc")):
        stem = os.path.basename(path)[:-3]
        if re.fullmatch(r"\d{8}", stem):
            date = datetime.strptime(stem, "%Y%m%d")
            paths[date] = path
    if not paths:
        raise FileNotFoundError(f"No daily model files in {folder}")
    first = paths[min(paths)]
    with nc.Dataset(first) as ds:
        original_lon = read_float(ds.variables["lon"])
        original_lat = read_float(ds.variables["lat"])
    xo, yo = np.argsort(original_lon), np.argsort(original_lat)
    lon = check_grid(original_lon[xo], "longitude")
    lat = check_grid(original_lat[yo], "latitude")
    if np.any((lon < -180) | (lon > 180)):
        raise ValueError("Convert model longitudes to -180..180 first")
    if np.any((lat < -90) | (lat > 90)):
        raise ValueError("Model latitude coordinates are invalid")
    epoch = datetime(1970, 1, 1)
    warned = False
    @lru_cache(maxsize=3)
    def read_day(date):
        nonlocal warned
        path = paths.get(date)
        if path is None:
            return None
        with nc.Dataset(path) as ds:
            if (not np.array_equal(read_float(ds.variables["lon"]),
                                   original_lon)
                    or not np.array_equal(read_float(ds.variables["lat"]),
                                          original_lat)):
                raise ValueError(f"Model grid changes in {path}")
            var = ds.variables[args.variable]
            values = read_tyx(var)
            time_var = ds.variables.get(var.dimensions[
                next(i for i, d in enumerate(var.dimensions)
                     if d.lower() in ("time", "t"))
            ])
            if (
                args.assume_model_step_hours is None
                and time_var is not None
                and "since" in str(getattr(time_var, "units", ""))
            ):
                dates = nc.num2date(
                    time_var[:], time_var.units,
                    calendar=getattr(time_var, "calendar", "standard"),
                )
                seconds = np.array([
                    (datetime(d.year, d.month, d.day,
                              d.hour, d.minute, d.second) - epoch
                     ).total_seconds()
                    for d in dates
                ])
            else:
                step = args.assume_model_step_hours
                if step is None:
                    raise ValueError(
                        f"No usable CF time coordinate in {path}. "
                        "Check its timestamps before supplying "
                        "--assume-model-step-hours."
                    )
                if not np.isclose(values.shape[0] * step, 24):
                    raise ValueError(
                        f"Record count and assumed timestep disagree: {path}"
                    )
                seconds = (
                    (date - epoch).total_seconds()
                    + np.arange(values.shape[0]) * step * 3600
                )
                if not warned:
                    print("WARNING: overriding time coordinates; daily records start at "
                          "00:00 UTC with the specified spacing.", flush=True)
                    warned = True
        if len(seconds) != values.shape[0]:
            raise ValueError(f"Time length mismatch in {path}")
        if np.any(np.diff(seconds) <= 0):
            raise ValueError(f"Non-increasing timestamps in {path}")
        values = values[:, yo, :][:, :, xo]
        values[~np.isfinite(values)
               | (values <= 100) | (values >= 400)] = np.nan
        return seconds, values
    def sample(date, local_hour):
        # Local mean solar time = UTC + longitude / 15.
        targets = (
            (date - epoch).total_seconds()
            + (local_hour - lon / 15.0) * 3600
        )
        required_days = {
            epoch + timedelta(days=int(np.floor(t / 86400)))
            for t in targets
        }
        pieces = []
        for day in sorted(required_days):
            for candidate in (day, day + timedelta(days=1)):
                item = read_day(candidate)
                if item is not None:
                    pieces.append((candidate, item))
        unique = {day: item for day, item in pieces}
        result = np.full((lat.size, lon.size), np.nan)
        if not unique:
            return result
        times = np.concatenate([item[0] for item in unique.values()])
        values = np.concatenate([item[1] for item in unique.values()])
        order = np.argsort(times)
        times, values = times[order], values[order]
        for j, target in enumerate(targets):
            k = np.searchsorted(times, target)
            if k < len(times) and abs(times[k] - target) < 0.01:
                result[:, j] = values[k, :, j]
            elif 0 < k < len(times):
                gap = times[k] - times[k - 1]
                # Reject gaps caused by missing records/days.
                cadence = np.median(np.diff(times))
                if gap > 1.5 * cadence:
                    continue
                fraction = (target - times[k - 1]) / gap
                a, b = values[k - 1, :, j], values[k, :, j]
                valid = np.isfinite(a) & np.isfinite(b)
                result[valid, j] = (
                    a[valid] * (1 - fraction) + b[valid] * fraction
                )
        return result - 273.15
    return lon, lat, sample
def model_month(args, sample, year, month, shape):
    days = calendar.monthrange(year, month)[1]
    totals = {p: np.zeros(shape) for p in PERIODS}
    counts = {p: np.zeros(shape, dtype=int) for p in PERIODS}
    for day in range(1, days + 1):
        date = datetime(year, month, day)
        for period, hour in (
            ("day", args.day_local_hour),
            ("night", args.night_local_hour),
        ):
            values = sample(date, hour)
            valid = np.isfinite(values)
            totals[period] += np.where(valid, values, 0)
            counts[period] += valid
    results = {}
    for period in PERIODS:
        good = counts[period] >= np.ceil(
            days * args.minimum_day_coverage
        )
        result = np.full(shape, np.nan)
        np.divide(totals[period], counts[period],
                  out=result, where=good)
        results[period] = result
    return results
def satellite_month(args, path, lon, lat):
    west, east = edges(lon)
    south, north = edges(lat)
    transform = from_bounds(
        west, south, east, north, len(lon), len(lat)
    )
    results = {}
    with rasterio.open(path) as src:
        if src.count < 2 or src.crs is None:
            raise ValueError(f"Need two bands and a CRS: {path}")
        for band, period in ((1, "day"), (2, "night")):
            values = src.read(band, masked=True).astype(float).filled(np.nan)
            if args.modis_units == "native":
                values[values == 0] = np.nan
                values = values * 0.02 - 273.15
            elif args.modis_units == "kelvin":
                values = values - 273.15
            values[~np.isfinite(values)
                   | (values <= -100) | (values >= 100)] = np.nan
            destination = np.full((len(lat), len(lon)), np.nan)
            reproject(
                source=values, destination=destination,
                src_transform=src.transform, src_crs=src.crs,
                dst_transform=transform, dst_crs="EPSG:4326",
                src_nodata=np.nan, dst_nodata=np.nan,
                resampling=Resampling.average,
            )
            results[period] = np.flipud(destination)
    return results
def seasonal_means(args):
    # Check satellite files before reading years of model output.
    records = satellite_records(args)
    lon, lat, sample = make_model_reader(args)
    shape = (len(lat), len(lon))
    sums = {
        dataset: {
            p: {s: np.zeros(shape) for s in SEASONS}
            for p in PERIODS
        }
        for dataset in ("model", "satellite")
    }
    weights = {
        p: {s: np.zeros(shape) for s in SEASONS}
        for p in PERIODS
    }
    for number, (year, month, path) in enumerate(records, 1):
        season = next(s for s, months in SEASONS.items()
                      if month in months)
        days = calendar.monthrange(year, month)[1]
        observed = satellite_month(args, path, lon, lat)
        simulated = model_month(args, sample, year, month, shape)
        for period in PERIODS:
            model_valid = np.isfinite(simulated[period])
            modis_valid = np.isfinite(observed[period])
            print(
                f"{year}-{month:02d} {period}: "
                f"model={int(model_valid.sum())}, "
                f"MODIS={int(modis_valid.sum())}, "
                f"paired={int((model_valid & modis_valid).sum())}",
                flush=True,
            )
            paired = (
                np.isfinite(simulated[period])
                & np.isfinite(observed[period])
            )
            sums["model"][period][season] += np.where(
                paired, simulated[period] * days, 0
            )
            sums["satellite"][period][season] += np.where(
                paired, observed[period] * days, 0
            )
            weights[period][season] += np.where(paired, days, 0)
        print(f"Processed {number}/{len(records)}: "
              f"{year}-{month:02d}", flush=True)
    maps = {}
    for dataset in sums:
        maps[dataset] = {p: {} for p in PERIODS}
        for period in PERIODS:
            for season in SEASONS:
                result = np.full(shape, np.nan)
                np.divide(
                    sums[dataset][period][season],
                    weights[period][season], out=result,
                    where=weights[period][season] > 0,
                )
                maps[dataset][period][season] = result
    return lon, lat, maps
def metrics(model, observed):
    paired = np.isfinite(model) & np.isfinite(observed)
    a, b = model[paired], observed[paired]
    difference = a - b
    if not difference.size:
        return difference, np.nan, np.nan, np.nan, np.nan
    r = np.nan
    if difference.size > 1 and np.std(a) > 0 and np.std(b) > 0:
        r = float(np.corrcoef(a, b)[0, 1])
    return (
        difference, float(difference.mean()),
        float(np.abs(difference).mean()),
        float(np.sqrt(np.mean(difference ** 2))), r,
    )
def plot_maps(args, lon, lat, maps, period):
    model = maps["model"][period]
    observed = maps["satellite"][period]
    temperature_values, difference_values = [], []
    for season in SEASONS:
        paired = np.isfinite(model[season]) & np.isfinite(observed[season])
        temperature_values.extend(
            (model[season][paired], observed[season][paired])
        )
        difference_values.append(
            model[season][paired] - observed[season][paired]
        )
    temperature_values = np.concatenate(temperature_values)
    difference_values = np.concatenate(difference_values)
    if not temperature_values.size:
        raise RuntimeError(f"No paired values for {period}")
    low, high = temperature_values.min(), temperature_values.max()
    if low == high:
        low, high = low - 0.01, high + 0.01
    limit = max(np.abs(difference_values).max(), 1e-6)
    tnorm = colors.Normalize(low, high)
    dnorm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    fig, axes = plt.subplots(
        4, 3, figsize=(15, 19), sharex=True, sharey=True
    )
    for row, season in enumerate(SEASONS):
        paired = np.isfinite(model[season]) & np.isfinite(observed[season])
        panels = (
            (model[season], "HB-NMP TRAD", "RdYlBu_r", tnorm),
            (observed[season], "MODIS LST", "RdYlBu_r", tnorm),
            (model[season] - observed[season],
             "HB-NMP − MODIS", "RdBu_r", dnorm),
        )
        for col, (values, heading, cmap, norm) in enumerate(panels):
            ax = axes[row, col]
            image = ax.pcolormesh(
                lon, lat, np.where(paired, values, np.nan),
                cmap=cmap, norm=norm, shading="auto",
            )
            ax.set_title(f"{season}: {heading}")
            ax.set_xlabel("Longitude [°]")
            ax.set_ylabel("Latitude [°]")
            fig.colorbar(
                image, ax=ax, shrink=0.86,
                label="Difference [°C]" if col == 2 else "LST [°C]",
                extend="neither",
            )
    fig.tight_layout()
    path = os.path.join(
        args.outdir,
        f"seasonal_{period}_TRAD_MODIS_difference_"
        f"{args.start_year}_{args.end_year}.png",
    )
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path, flush=True)
def plot_histograms(args, maps):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    rows = []
    for row, period in enumerate(PERIODS):
        for col, season in enumerate(SEASONS):
            ax = axes[row, col]
            diff, md, mad, rmse, r = metrics(
                maps["model"][period][season],
                maps["satellite"][period][season],
            )
            rows.append((period, season, md, mad, rmse, r, diff.size))
            if not diff.size:
                ax.set_title(f"{season}: no paired cells")
                ax.axis("off")
                continue
            ax.hist(diff, bins=args.bins, density=True)
            ax.axvline(md, color="darkred", linestyle="--", linewidth=4)
            rtext = f"{r:.3f}" if np.isfinite(r) else "NA"
            ax.text(
                0.98, 0.98,
                f"MD = {md:.2f} °C\nMAD = {mad:.2f} °C\n"
                f"RMSE = {rmse:.2f} °C\nr = {rtext}\nn = {diff.size:,}",
                transform=ax.transAxes, ha="right", va="top",
                fontsize=10,
                bbox=dict(boxstyle="round", facecolor="lightgray",
                          edgecolor="black", alpha=0.6),
            )
            ax.set_title(f"{season} – {period.capitalize()}")
            ax.set_xlabel("HB-NMP − MODIS [°C]")
            if col == 0:
                ax.set_ylabel("Probability density")
    fig.tight_layout()
    path = os.path.join(
        args.outdir,
        f"seasonal_day_night_TRAD_MODIS_histograms_"
        f"{args.start_year}_{args.end_year}.png",
    )
    fig.savefig(path, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)
    print("Saved:", path, flush=True)
    with open(os.path.join(args.outdir, "TRAD_MODIS_metrics.csv"),
              "w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(("period", "season", "MD_C", "MAD_C",
                         "RMSE_C", "r", "n_grid_cells"))
        writer.writerows(rows)
def main():
    args = arguments()
    os.makedirs(args.outdir, exist_ok=True)
    print(
        f"Nominal local solar hours: day={args.day_local_hour}, "
        f"night={args.night_local_hour}", flush=True,
    )
    lon, lat, maps = seasonal_means(args)
    for period in PERIODS:
        plot_maps(args, lon, lat, maps, period)
    plot_histograms(args, maps)
if __name__ == "__main__":
    main()
