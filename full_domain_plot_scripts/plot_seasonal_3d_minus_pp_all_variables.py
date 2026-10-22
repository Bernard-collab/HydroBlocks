#!/usr/bin/env python3
"""Seasonal paired 3D-minus-PP maps, histograms, and spatial metrics."""

import argparse
import csv
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

SEASONS = {
    "Winter": (12, 1, 2), "Spring": (3, 4, 5),
    "Summer": (6, 7, 8), "Fall": (9, 10, 11),
}

BASE = (
    "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/"
    "experiments/simulations"
)
DEFAULT_EDIR_3D = os.path.join(
    BASE, "MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test"
)
DEFAULT_EDIR_PP = os.path.join(
    BASE, "MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test"
)
DEFAULT_OUTDIR = "/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/Seasonal_3D_minus_PP"

VARIABLES = (
    ("swdn_3d", "swdn_3d", "swdn_3d", r"W m$^{-2}$"),
    ("swnet", "swnet", "swnet", r"W m$^{-2}$"),
    ("lwdn_3d", "lwdn_3d", "lwdn", r"W m$^{-2}$"),
    ("lwnet", "lwnet", "lwnet", r"W m$^{-2}$"),
    ("totsmc", "totsmc", "totsmc", r"m$^3$ m$^{-3}$"),
    ("smc", "smc", "smc", r"m$^3$ m$^{-3}$"),
    ("swe", "swe", "swe", "mm"),
    ("snowh", "snowh", "snowh", "m"),
    ("fsno", "fsno", "fsno", "fraction"),
    ("qsnow", "qsnow", "qsnow", r"mm s$^{-1}$"),
    ("snow_t", "snow_t", "snow_t", "K"),
    ("t2mb", "t2mb", "t2mb", "K"),
    ("t2mv", "t2mv", "t2mv", "K"),
    ("trad", "trad", "trad", "K"),
    ("tv", "tv", "tv", "K"),
    ("sh", "sh", "sh", r"W m$^{-2}$"),
    ("lh", "lh", "lh", r"W m$^{-2}$"),
    ("salb", "salb", "salb", "fraction"),
    ("qsurface", "qsurface", "qsurface", r"mm s$^{-1}$"),
    ("runoff", "runoff", "runoff", r"mm s$^{-1}$"),
)

PAIRS = (
    ("shortwave", "swdn_3d", "swnet"),
    ("longwave", "lwdn_3d", "lwnet"),
    ("soil_moisture", "smc", "totsmc"),
    ("snow_amount", "swe", "snowh"),
    ("snow_state", "fsno", "qsnow"),
    ("near_surface_temperature", "t2mb", "t2mv"),
    ("surface_temperature", "trad", "tv"),
    ("turbulent_fluxes", "sh", "lh"),
    ("surface_water", "qsurface", "runoff"),
    ("snow_temperature_albedo", "snow_t", "salb"),
)


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edir-3d", default=DEFAULT_EDIR_3D)
    p.add_argument("--edir-pp", default=DEFAULT_EDIR_PP)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--outdir", default=DEFAULT_OUTDIR)
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--pairs", nargs="+", choices=("all",) + tuple(x[0] for x in PAIRS), default=("all",))
    p.add_argument("--soil-layer", type=int, default=0)
    p.add_argument("--snow-layer", type=int, default=0)
    return p.parse_args()


def daily_files(root, start_year, end_year):
    folder = os.path.join(os.path.abspath(root), "postprocess", "output_dir")
    result = {}
    for year in range(start_year, end_year + 1):
        for filename in glob.glob(os.path.join(folder, f"{year}????.nc")):
            key = os.path.basename(filename)[:8]
            try:
                datetime.strptime(key, "%Y%m%d")
            except ValueError:
                continue
            if key in result:
                raise ValueError(f"Duplicate daily file: {key}")
            result[key] = filename
    if not result:
        raise FileNotFoundError(f"No daily postprocessed files in {folder}")
    return result


def axis_index(dimensions, candidates):
    lower = [name.lower() for name in dimensions]
    return next((lower.index(name) for name in candidates if name in lower), None)


def clean(variable, selection=slice(None)):
    # Retain physical values; exclude only missing and nonfinite data.
    data = np.ma.asarray(variable[selection], dtype=float).filled(np.nan)
    for attr in ("_FillValue", "missing_value"):
        for fill in np.atleast_1d(getattr(variable, attr, [])):
            data[data == fill] = np.nan
    data[(data == -9999) | ~np.isfinite(data)] = np.nan
    return data


def read_time_lat_lon(variable, args):
    dims = list(variable.dimensions)
    ta = axis_index(dims, ("time", "t"))
    ya = axis_index(dims, ("lat", "latitude", "y"))
    xa = axis_index(dims, ("lon", "longitude", "x"))
    if None in (ta, ya, xa):
        raise ValueError(f"Cannot identify time/latitude/longitude axes: {variable.name}, {dims}")

    extras = [i for i in range(len(dims)) if i not in (ta, ya, xa)]
    if len(extras) > 1:
        raise ValueError(f"More than one extra axis: {variable.name}, {dims}")

    selection = [slice(None)] * len(dims)
    if extras:
        axis = extras[0]
        layer = args.snow_layer if variable.name.split("/")[-1] == "snow_t" else args.soil_layer
        if not 0 <= layer < variable.shape[axis]:
            raise IndexError(f"Invalid layer {layer} for {variable.name}")
        selection[axis] = layer

    data = clean(variable, tuple(selection))
    retained = [d for i, d in enumerate(dims) if i not in extras]
    order = (
        axis_index(retained, ("time", "t")),
        axis_index(retained, ("lat", "latitude", "y")),
        axis_index(retained, ("lon", "longitude", "x")),
    )
    return np.transpose(data, order)


def read_coordinates(ds):
    names = []
    for candidates in (("lon", "longitude", "x"), ("lat", "latitude", "y")):
        name = next((n for n in candidates if n in ds.variables), None)
        if name is None:
            raise KeyError(f"Missing coordinate: {candidates}")
        names.append(name)

    lon, lat = (clean(ds.variables[name]) for name in names)
    for values in (lon, lat):
        if values.ndim != 1 or not np.isfinite(values).all() or np.unique(values).size != values.size:
            raise ValueError("Coordinates must be finite, unique, and one-dimensional")
    return lon, lat


def read_clock(ds, file_date):
    name = next((n for n in ("time", "t") if n in ds.variables), None)
    if name is None:
        raise KeyError("Missing time coordinate")

    t = ds.variables[name]
    dates = nc.num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
    keys = [(d.year, d.month, d.day, d.hour, d.minute, d.second) for d in dates]

    if not keys or any(a >= b for a, b in zip(keys, keys[1:])):
        raise ValueError("Empty, duplicate, or non-increasing daily timestamps")
    if any(k[:3] != (file_date.year, file_date.month, file_date.day) for k in keys):
        raise ValueError("Time coordinates do not match the daily filename")
    return keys


def coordinate_match(a, b):
    return a.shape == b.shape and np.allclose(a, b, rtol=0, atol=1e-8)


def initialise_accumulators(shape, specs):
    return {
        name: {
            season: {
                "sum_3d": np.zeros(shape),
                "sum_pp": np.zeros(shape),
                "count": np.zeros(shape, dtype=np.int64),
            }
            for season in SEASONS
        }
        for name, _, _, _ in specs
    }


def read_seasonal_means(args, specs):
    f3 = daily_files(args.edir_3d, args.start_year, args.end_year)
    fp = daily_files(args.edir_pp, args.start_year, args.end_year)
    common = sorted(set(f3) & set(fp))
    if not common:
        raise ValueError("No matching daily files")

    expected_days = (
        datetime(args.end_year + 1, 1, 1) - datetime(args.start_year, 1, 1)
    ).days
    print(
        f"Matched days: {len(common)}; missing 3D: {len(set(fp)-set(f3))}; "
        f"missing PP: {len(set(f3)-set(fp))}; "
        f"absent from both: {expected_days-len(set(f3)|set(fp))}",
        flush=True
    )

    with nc.Dataset(f3[common[0]]) as d3, nc.Dataset(fp[common[0]]) as dp:
        coords3, coordsp = read_coordinates(d3), read_coordinates(dp)

    if not all(coordinate_match(np.sort(a), np.sort(b)) for a, b in zip(coords3, coordsp)):
        raise ValueError("3D and PP coordinate grids differ")

    orders = [
        (np.argsort(lat), np.argsort(lon))
        for lon, lat in (coords3, coordsp)
    ]
    lon, lat = np.sort(coords3[0]), np.sort(coords3[1])
    accum = initialise_accumulators((lat.size, lon.size), specs)
    previous = None
    timestep = None
    seasonal_steps = {season: 0 for season in SEASONS}

    for number, key in enumerate(common, 1):
        date = datetime.strptime(key, "%Y%m%d")
        season = next(s for s, months in SEASONS.items() if date.month in months)

        with nc.Dataset(f3[key]) as d3, nc.Dataset(fp[key]) as dp:
            # Verify every day's grid, not only the first file.
            for ds, reference in ((d3, coords3), (dp, coordsp)):
                if not all(coordinate_match(a, b) for a, b in zip(read_coordinates(ds), reference)):
                    raise ValueError(f"Grid changed on {key}")

            clocks = [read_clock(ds, date) for ds in (d3, dp)]
            if clocks[0] != clocks[1]:
                raise ValueError(f"3D/PP timestamp mismatch on {key}")
            times = clocks[0]

            if previous is not None and times[0] <= previous:
                raise ValueError(f"Overlapping output timestamps on {key}")
            previous = times[-1]

            intervals = [
                (datetime(*b)-datetime(*a)).total_seconds()
                for a, b in zip(times, times[1:])
            ]
            if intervals:
                if len(set(intervals)) != 1:
                    raise ValueError(f"Unequal timestep lengths on {key}")
                if timestep is None:
                    timestep = intervals[0]
                elif timestep != intervals[0]:
                    raise ValueError(f"Output timestep changed on {key}")

            seasonal_steps[season] += len(times)

            for name, name3, namep, _ in specs:
                arrays = []
                for ds, variable, (iy, ix) in zip((d3, dp), (name3, namep), orders):
                    if variable not in ds.variables:
                        raise KeyError(f"{variable!r} missing on {key}; check postprocessing variables")
                    values = read_time_lat_lon(ds.variables[variable], args)
                    expected_shape = (len(times), lat.size, lon.size)
                    if values.shape != expected_shape:
                        raise ValueError(f"{variable} on {key}: {values.shape}, expected {expected_shape}")
                    arrays.append(values[:, iy, :][:, :, ix])

                a, b = arrays
                valid = np.isfinite(a) & np.isfinite(b)
                target = accum[name][season]
                target["sum_3d"] += np.where(valid, a, 0).sum(axis=0)
                target["sum_pp"] += np.where(valid, b, 0).sum(axis=0)
                target["count"] += valid.sum(axis=0)

        if number % 100 == 0 or number == len(common):
            print(f"Processed {number}/{len(common)} matched days", flush=True)

    print("Paired seasonal timesteps:", seasonal_steps, flush=True)
    seasonal = {}
    for name, _, _, _ in specs:
        seasonal[name] = {}
        for season in SEASONS:
            source = accum[name][season]
            a, b = np.full(source["count"].shape, np.nan), np.full(source["count"].shape, np.nan)
            np.divide(source["sum_3d"], source["count"], out=a, where=source["count"] > 0)
            np.divide(source["sum_pp"], source["count"], out=b, where=source["count"] > 0)
            seasonal[name][season] = {"three_d": a, "pp": b, "difference": a-b}
    return lon, lat, seasonal


def metrics(a, b):
    valid = np.isfinite(a) & np.isfinite(b)
    a, b = a[valid], b[valid]
    diff = a-b
    if not diff.size:
        return diff, np.nan, np.nan, np.nan, np.nan, 0

    md = float(diff.mean())
    mad = float(np.abs(diff).mean())
    rmse = float(np.sqrt(np.mean(diff**2)))
    r = float(np.corrcoef(a, b)[0, 1]) if diff.size > 1 and a.std() > 0 and b.std() > 0 else np.nan
    return diff, md, mad, rmse, r, diff.size


def display_limit(arrays):
    maxima = [np.max(np.abs(a[np.isfinite(a)])) for a in arrays if np.isfinite(a).any()]
    return max(float(max(maxima)), 1e-12) if maxima else 0.1


def variable_label(name):
    return "lwnet (+ upward)" if name == "lwnet" else name


def plot_maps(lon, lat, seasonal, specs, output, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad("0.75")
    mappables = []

    # Reserve colorbar space explicitly to avoid later layout overlap.
    fig.subplots_adjust(left=0.065, right=0.89, bottom=0.07, top=0.88, hspace=0.20, wspace=0.15)

    for row, (name, _, _, unit) in enumerate(specs):
        limit = display_limit([seasonal[name][s]["difference"] for s in SEASONS])
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)

        for column, season in enumerate(SEASONS):
            ax = axes[row, column]
            image = ax.pcolormesh(
                lon, lat, seasonal[name][season]["difference"],
                cmap=cmap, norm=norm, shading="auto", rasterized=True
            )
            if row == 0:
                ax.set_title(season, fontsize=13)
            if column == 0:
                ax.set_ylabel(f"{variable_label(name)}\nLatitude")
            if row == 1:
                ax.set_xlabel("Longitude")
        mappables.append(image)

    for row, ((name, _, _, unit), image) in enumerate(zip(specs, mappables)):
        bounds = axes[row, 0].get_position()
        cax = fig.add_axes([0.915, bounds.y0, 0.012, bounds.height])
        fig.colorbar(image, cax=cax).set_label(f"{variable_label(name)}: 3D − PP [{unit}]")

    fig.suptitle(
        f"Seasonal 3D − PP: {specs[0][0]} and {specs[1][0]}\n"
        "Shared symmetric scale across seasons within each row",
        fontsize=16
    )
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", output, flush=True)


def plot_histograms(seasonal, specs, output, dpi):
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for row, (name, _, _, unit) in enumerate(specs):
        limit = display_limit([seasonal[name][s]["difference"] for s in SEASONS])
        edges = np.linspace(-limit, limit, 51)

        for column, season in enumerate(SEASONS):
            ax = axes[row, column]
            values = seasonal[name][season]
            diff, md, mad, rmse, r, n = metrics(values["three_d"], values["pp"])
            if not n:
                ax.set_title(f"{season}: no valid pixels")
                ax.axis("off")
                continue

            ax.hist(diff, bins=edges, density=True, histtype="stepfilled", color="#4285a4", alpha=0.35)
            ax.axvline(0, linewidth=1, color="black")
            ax.axvline(md, linestyle="--", linewidth=2, color="darkred")
            ax.set_xlim(-limit, limit)
            ax.text(
                0.98, 0.98, f"MD = {md:.3g}\nMAD = {mad:.3g}\nRMSE = {rmse:.3g}\nr = {r:.3f}\nn = {n:,}",
                transform=ax.transAxes, ha="right", va="top", fontsize=10,
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85)
            )
            if row == 0:
                ax.set_title(season, fontsize=13)
            if column == 0:
                ax.set_ylabel(f"{variable_label(name)}\nProbability density")
            ax.set_xlabel(f"3D − PP [{unit}]")

    fig.suptitle(f"Seasonal difference distributions: {specs[0][0]} and {specs[1][0]}", fontsize=16)
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.91, hspace=0.32, wspace=0.24)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("Saved:", output, flush=True)


def write_metrics(seasonal, specs, output):
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variable", "season", "unit", "MD", "MAD", "RMSE", "r", "n"))
        for name, _, _, unit in specs:
            for season in SEASONS:
                values = seasonal[name][season]
                _, md, mad, rmse, r, n = metrics(values["three_d"], values["pp"])
                writer.writerow((name, season, unit, md, mad, rmse, r, n))
    print("Saved:", output, flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("Invalid year range")
    if min(args.soil_layer, args.snow_layer) < 0 or args.dpi < 1:
        raise ValueError("Layer indices must be nonnegative and DPI positive")

    pairs = PAIRS if "all" in args.pairs else tuple(p for p in PAIRS if p[0] in args.pairs)
    requested = {name for _, a, b in pairs for name in (a, b)}
    specs = tuple(v for v in VARIABLES if v[0] in requested)
    lookup = {v[0]: v for v in specs}

    os.makedirs(args.outdir, exist_ok=True)
    print(f"3D: {args.edir_3d}\nPP: {args.edir_pp}\nOutput: {args.outdir}", flush=True)
    lon, lat, seasonal = read_seasonal_means(args, specs)

    for pair_name, first, second in pairs:
        pair_specs = (lookup[first], lookup[second])
        suffix = f"{pair_name}_{args.start_year}_{args.end_year}"
        plot_maps(
            lon, lat, seasonal, pair_specs,
            os.path.join(args.outdir, f"seasonal_3D_minus_PP_maps_{suffix}.png"), args.dpi
        )
        plot_histograms(
            seasonal, pair_specs,
            os.path.join(args.outdir, f"seasonal_3D_minus_PP_histograms_{suffix}.png"), args.dpi
        )

    write_metrics(
        seasonal, specs,
        os.path.join(args.outdir, f"seasonal_3D_minus_PP_metrics_{args.start_year}_{args.end_year}.csv")
    )


if __name__ == "__main__":
    main()