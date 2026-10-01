#!/usr/bin/env python3
"""Plot paired HydroBlocks variables' seasonal 3D-minus-PP differences.

The script reads matching daily postprocessed NetCDF files from
<experiment>/postprocess/output_dir, calculates paired seasonal means, and
creates:

1. A 2-by-4 map figure for each variable pair (two variables by four seasons).
2. A matching 2-by-4 histogram figure for each variable pair.
3. One CSV table containing MD, MAD, RMSE, Pearson r, and valid-pixel count.

For 3D radiation, swdn_3d is compared with PP swdn and lwdn_3d is
compared with PP lwdn. Other variables use the same name in both simulations.

Example:
  python plot_seasonal_3d_minus_pp_paired.py \
    --edir-3d "/path/to/3D_EXPERIMENT" \
    --edir-pp "/path/to/PP_EXPERIMENT" \
    --start-year 2014 --end-year 2024 \
    --outdir "/path/to/validation_plots" --dpi 400
"""

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
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}

DEFAULT_EDIR_3D = (
    "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/"
    "experiments/simulations/"
    "old_MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_"
    "full_domain_downscaled_test"
)

DEFAULT_EDIR_PP = (
    "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/"
    "experiments/simulations/"
    "old_MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_"
    "full_domain_downscaled_test"
)

DEFAULT_OUTDIR = (
    "/scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/"
    "validation_plots/3D_minus_PP"
)

VARIABLES = (
    {
        "name": "swdn_3d",
        "three_d": ("swdn_3d",),
        "pp": ("swdn",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "swnet",
        "three_d": ("swnet",),
        "pp": ("swnet",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lwdn_3d",
        "three_d": ("lwdn_3d",),
        "pp": ("lwdn",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lwnet",
        "three_d": ("lwnet",),
        "pp": ("lwnet",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "totsmc",
        "three_d": ("totsmc",),
        "pp": ("totsmc",),
        "unit": "m$^3$ m$^{-3}$",
    },
    {
        "name": "smc",
        "three_d": ("smc",),
        "pp": ("smc",),
        "unit": "m$^3$ m$^{-3}$",
    },
    {
        "name": "swe",
        "three_d": ("swe",),
        "pp": ("swe",),
        "unit": "mm",
    },
    {
        "name": "snowh",
        "three_d": ("snowh",),
        "pp": ("snowh",),
        "unit": "m",
    },
    {
        "name": "fsno",
        "three_d": ("fsno",),
        "pp": ("fsno",),
        "unit": "unitless",
    },
    {
        "name": "qsnow",
        "three_d": ("qsnow",),
        "pp": ("qsnow",),
        "unit": "mm s$^{-1}$",
    },
    {
        "name": "snow_t",
        "three_d": ("snow_t",),
        "pp": ("snow_t",),
        "unit": "K",
    },
    {
        "name": "t2mb",
        "three_d": ("t2mb",),
        "pp": ("t2mb",),
        "unit": "K",
    },
    {
        "name": "t2mv",
        "three_d": ("t2mv",),
        "pp": ("t2mv",),
        "unit": "K",
    },
    {
        "name": "trad",
        "three_d": ("trad",),
        "pp": ("trad",),
        "unit": "K",
    },
    {
        "name": "tv",
        "three_d": ("tv",),
        "pp": ("tv",),
        "unit": "K",
    },
    {
        "name": "sh",
        "three_d": ("sh",),
        "pp": ("sh",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lh",
        "three_d": ("lh",),
        "pp": ("lh",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "salb",
        "three_d": ("salb",),
        "pp": ("salb",),
        "unit": "unitless",
    },
    {
        "name": "qsurface",
        "three_d": ("qsurface",),
        "pp": ("qsurface",),
        "unit": "mm s$^{-1}$",
    },
    {
        "name": "runoff",
        "three_d": ("runoff",),
        "pp": ("runoff",),
        "unit": "mm s$^{-1}$",
    },
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edir-3d", default=DEFAULT_EDIR_3D, help="3D experiment directory"
    )
    parser.add_argument(
        "--edir-pp", default=DEFAULT_EDIR_PP, help="PP experiment directory"
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--pairs",
        nargs="+",
        choices=("all",) + tuple(pair[0] for pair in PAIRS),
        default=("all",),
        help="Variable pair(s) to plot; default: all",
    )
    parser.add_argument(
        "--soil-layer",
        type=int,
        default=0,
        help="Layer selected if a postprocessed variable still has a soil axis",
    )
    return parser.parse_args()


def daily_files(edir, start_year, end_year):
    folder = os.path.join(os.path.abspath(edir), "postprocess", "output_dir")
    files = {}
    for year in range(start_year, end_year + 1):
        for filename in glob.glob(os.path.join(folder, f"{year}????.nc")):
            key = os.path.basename(filename)[:8]
            try:
                datetime.strptime(key, "%Y%m%d")
            except ValueError:
                continue
            files[key] = filename
    if not files:
        raise FileNotFoundError(f"No postprocessed daily NetCDF files found in {folder}")
    return files


def find_variable(dataset, candidates, experiment_name):
    for candidate in candidates:
        if candidate in dataset.variables:
            return dataset.variables[candidate], candidate
    available = ", ".join(sorted(dataset.variables))
    raise KeyError(
        f"None of {candidates} found in {experiment_name}. "
        f"Available root variables: {available}"
    )


def axis_index(dimensions, candidates):
    lowered = [name.lower() for name in dimensions]
    for candidate in candidates:
        if candidate in lowered:
            return lowered.index(candidate)
    return None


def read_time_lat_lon(variable, soil_layer):
    """Read a variable as (time, lat, lon), selecting extra axes if present."""
    dimensions = list(variable.dimensions)
    time_axis = axis_index(dimensions, ("time", "t"))
    lat_axis = axis_index(dimensions, ("lat", "latitude", "y"))
    lon_axis = axis_index(dimensions, ("lon", "longitude", "x"))
    if None in (time_axis, lat_axis, lon_axis):
        raise ValueError(
            f"Cannot identify time/latitude/longitude axes for "
            f"{variable.name}: {variable.dimensions}"
        )

    selection = []
    retained_dimensions = []
    for axis, name in enumerate(dimensions):
        if axis in (time_axis, lat_axis, lon_axis):
            selection.append(slice(None))
            retained_dimensions.append(name)
        else:
            if soil_layer >= variable.shape[axis]:
                raise IndexError(
                    f"Requested layer {soil_layer} exceeds axis {name} of "
                    f"{variable.name}, whose size is {variable.shape[axis]}"
                )
            selection.append(soil_layer)

    data = np.ma.asarray(variable[tuple(selection)]).filled(np.nan).astype(float)
    time_axis = axis_index(retained_dimensions, ("time", "t"))
    lat_axis = axis_index(retained_dimensions, ("lat", "latitude", "y"))
    lon_axis = axis_index(retained_dimensions, ("lon", "longitude", "x"))
    return np.transpose(data, (time_axis, lat_axis, lon_axis))


def read_coordinates(dataset):
    lon_name = next(
        (name for name in ("lon", "longitude", "x") if name in dataset.variables),
        None,
    )
    lat_name = next(
        (name for name in ("lat", "latitude", "y") if name in dataset.variables),
        None,
    )
    if lon_name is None or lat_name is None:
        raise KeyError("Postprocessed files must contain lon and lat coordinates")
    return (
        np.asarray(dataset.variables[lon_name][:], dtype=float),
        np.asarray(dataset.variables[lat_name][:], dtype=float),
    )


def initialise_accumulators(shape, specifications):
    return {
        variable["name"]: {
            season: {
                "sum_3d": np.zeros(shape, dtype=np.float64),
                "sum_pp": np.zeros(shape, dtype=np.float64),
                "count": np.zeros(shape, dtype=np.int64),
            }
            for season in SEASONS
        }
        for variable in specifications
    }


def read_seasonal_means(args, specifications):
    files_3d = daily_files(args.edir_3d, args.start_year, args.end_year)
    files_pp = daily_files(args.edir_pp, args.start_year, args.end_year)
    common_dates = sorted(set(files_3d) & set(files_pp))
    missing_3d = sorted(set(files_pp) - set(files_3d))
    missing_pp = sorted(set(files_3d) - set(files_pp))
    if not common_dates:
        raise RuntimeError("The 3D and PP experiments have no matching daily files")
    print(
        f"Matched days: {len(common_dates)}; "
        f"missing from 3D: {len(missing_3d)}; missing from PP: {len(missing_pp)}",
        flush=True,
    )

    with nc.Dataset(files_3d[common_dates[0]]) as ds_3d, nc.Dataset(
        files_pp[common_dates[0]]
    ) as ds_pp:
        lon_3d, lat_3d = read_coordinates(ds_3d)
        lon_pp, lat_pp = read_coordinates(ds_pp)

    if lon_3d.shape != lon_pp.shape or lat_3d.shape != lat_pp.shape:
        raise ValueError("3D and PP postprocessed grids have different dimensions")
    if not np.allclose(np.sort(lon_3d), np.sort(lon_pp)) or not np.allclose(
        np.sort(lat_3d), np.sort(lat_pp)
    ):
        raise ValueError("3D and PP postprocessed coordinate values do not match")

    lon_order_3d = np.argsort(lon_3d)
    lat_order_3d = np.argsort(lat_3d)
    lon_order_pp = np.argsort(lon_pp)
    lat_order_pp = np.argsort(lat_pp)
    lon = lon_3d[lon_order_3d]
    lat = lat_3d[lat_order_3d]
    accumulators = initialise_accumulators((lat.size, lon.size), specifications)
    resolved_names = {}

    for number, key in enumerate(common_dates, 1):
        date = datetime.strptime(key, "%Y%m%d")
        season = next(name for name, months in SEASONS.items() if date.month in months)

        with nc.Dataset(files_3d[key]) as ds_3d, nc.Dataset(files_pp[key]) as ds_pp:
            for specification in specifications:
                name = specification["name"]
                var_3d, actual_3d = find_variable(
                    ds_3d, specification["three_d"], "3D output"
                )
                var_pp, actual_pp = find_variable(
                    ds_pp, specification["pp"], "PP output"
                )
                if name not in resolved_names:
                    resolved_names[name] = (actual_3d, actual_pp)
                    print(
                        f"{name}: 3D variable={actual_3d}; PP variable={actual_pp}",
                        flush=True,
                    )

                data_3d = read_time_lat_lon(var_3d, args.soil_layer)
                data_pp = read_time_lat_lon(var_pp, args.soil_layer)
                if data_3d.shape != data_pp.shape:
                    raise ValueError(
                        f"Shape mismatch for {name} on {key}: "
                        f"3D={data_3d.shape}, PP={data_pp.shape}"
                    )

                data_3d = data_3d[:, lat_order_3d, :][:, :, lon_order_3d]
                data_pp = data_pp[:, lat_order_pp, :][:, :, lon_order_pp]
                valid = (
                    np.isfinite(data_3d)
                    & np.isfinite(data_pp)
                    & (data_3d > -9000.0)
                    & (data_pp > -9000.0)
                )
                target = accumulators[name][season]
                target["sum_3d"] += np.sum(
                    np.where(valid, data_3d, 0.0), axis=0
                )
                target["sum_pp"] += np.sum(
                    np.where(valid, data_pp, 0.0), axis=0
                )
                target["count"] += np.sum(valid, axis=0)

        if number % 100 == 0 or number == len(common_dates):
            print(f"Processed {number}/{len(common_dates)} matched days", flush=True)

    seasonal = {}
    for specification in specifications:
        name = specification["name"]
        seasonal[name] = {}
        for season in SEASONS:
            source = accumulators[name][season]
            mean_3d = np.full(source["count"].shape, np.nan)
            mean_pp = np.full(source["count"].shape, np.nan)
            np.divide(
                source["sum_3d"],
                source["count"],
                out=mean_3d,
                where=source["count"] > 0,
            )
            np.divide(
                source["sum_pp"],
                source["count"],
                out=mean_pp,
                where=source["count"] > 0,
            )
            seasonal[name][season] = {
                "three_d": mean_3d,
                "pp": mean_pp,
                "difference": mean_3d - mean_pp,
            }
    return lon, lat, seasonal


def metrics(three_d, pp):
    valid = np.isfinite(three_d) & np.isfinite(pp)
    values_3d = np.asarray(three_d[valid], dtype=float)
    values_pp = np.asarray(pp[valid], dtype=float)
    difference = values_3d - values_pp
    n_valid = difference.size
    if n_valid == 0:
        return difference, np.nan, np.nan, np.nan, np.nan, 0
    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    if n_valid >= 2 and np.std(values_3d) > 0 and np.std(values_pp) > 0:
        correlation = float(np.corrcoef(values_3d, values_pp)[0, 1])
    else:
        correlation = np.nan
    return difference, md, mad, rmse, correlation, n_valid


def display_limit(arrays):
    """Return the full maximum absolute value across all supplied maps."""
    values = [np.abs(array[np.isfinite(array)]) for array in arrays]
    values = [value for value in values if value.size]
    if not values:
        return 0.1
    return max(float(np.max(np.concatenate(values))), 1.0e-12)


def plot_maps(lon, lat, seasonal, specifications, output, dpi):
    """Plot two variables (rows) across four seasons (columns)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharex=True, sharey=True)
    cmap = plt.get_cmap("RdBu_r").copy()
    cmap.set_bad("0.75")

    for row, specification in enumerate(specifications):
        name = specification["name"]
        unit = specification["unit"]
        differences = [seasonal[name][season]["difference"] for season in SEASONS]
        # One full min/max symmetric scale for this variable across all seasons.
        limit = display_limit(differences)
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)

        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            image = axis.pcolormesh(
                lon,
                lat,
                seasonal[name][season]["difference"],
                cmap=cmap,
                norm=norm,
                shading="auto",
                rasterized=True,
            )
            if row == 0:
                axis.set_title(season, fontsize=13)
            if column == 0:
                axis.set_ylabel(f"{name}\nLatitude")
            if row == 1:
                axis.set_xlabel("Longitude")

        colorbar = fig.colorbar(
            image, ax=axes[row, :].tolist(), fraction=0.018, pad=0.018
        )
        colorbar.set_label(f"{name}: 3D − PP [{unit}]")

    pair_title = " and ".join(item["name"] for item in specifications)
    fig.suptitle(
        f"Seasonal 3D − PP Differences: {pair_title}\n"
        "Each row uses its own common full min/max scale",
        fontsize=16,
    )
    fig.subplots_adjust(left=0.065, right=0.91, bottom=0.07, top=0.88,
                        hspace=0.18, wspace=0.15)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def plot_histograms(seasonal, specifications, output, dpi):
    """Plot two variables (rows) across four seasonal histograms (columns)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))

    for row, specification in enumerate(specifications):
        name = specification["name"]
        unit = specification["unit"]
        seasonal_differences = [
            seasonal[name][season]["difference"] for season in SEASONS
        ]
        # Match the four histogram x-axes to the map scale for this variable.
        limit = display_limit(seasonal_differences)
        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            difference, md, mad, rmse, correlation, n_valid = metrics(
                seasonal[name][season]["three_d"], seasonal[name][season]["pp"]
            )
            if n_valid == 0:
                axis.set_title(f"{season}: no valid pixels")
                axis.axis("off")
                continue

            # Histograms and metrics use every valid paired pixel.
            axis.hist(
                difference, bins=50, range=(-limit, limit), density=True
            )
            axis.set_xlim(-limit, limit)
            axis.axvline(0.0, linewidth=1, color="black")
            axis.axvline(md, linestyle="dashed", linewidth=3, color="darkred")
            correlation_text = (
                f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
            )
            axis.text(
                0.98,
                0.98,
                (
                    f"MD = {md:.3g}\n"
                    f"MAD = {mad:.3g}\n"
                    f"RMSE = {rmse:.3g}\n"
                    f"r = {correlation_text}\n"
                    f"n = {n_valid:,}"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8.5,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightgray",
                    "edgecolor": "black",
                    "alpha": 0.65,
                },
            )
            if row == 0:
                axis.set_title(season, fontsize=13)
            if column == 0:
                axis.set_ylabel(f"{name}\nProbability density")
            axis.set_xlabel(f"3D − PP [{unit}]")

    pair_title = " and ".join(item["name"] for item in specifications)
    fig.suptitle(
        f"Seasonal 3D − PP Difference Distributions: {pair_title}", fontsize=16
    )
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.91,
                        hspace=0.32, wspace=0.24)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def write_metrics(seasonal, specifications, output):
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["variable", "season", "unit", "MD", "MAD", "RMSE", "r", "n"]
        )
        for specification in specifications:
            name = specification["name"]
            for season in SEASONS:
                _, md, mad, rmse, correlation, n_valid = metrics(
                    seasonal[name][season]["three_d"],
                    seasonal[name][season]["pp"],
                )
                writer.writerow(
                    [
                        name,
                        season,
                        specification["unit"],
                        md,
                        mad,
                        rmse,
                        correlation,
                        n_valid,
                    ]
                )
    print(f"Saved: {output}", flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("--end-year must not precede --start-year")
    selected_pairs = PAIRS if "all" in args.pairs else tuple(
        pair for pair in PAIRS if pair[0] in args.pairs
    )
    variable_names = {
        variable_name
        for _, first_name, second_name in selected_pairs
        for variable_name in (first_name, second_name)
    }
    specifications = tuple(
        item for item in VARIABLES if item["name"] in variable_names
    )
    specification_by_name = {item["name"]: item for item in specifications}
    os.makedirs(args.outdir, exist_ok=True)
    lon, lat, seasonal = read_seasonal_means(args, specifications)

    for pair_name, first_name, second_name in selected_pairs:
        pair_specifications = (
            specification_by_name[first_name],
            specification_by_name[second_name],
        )
        suffix = f"{pair_name}_{args.start_year}_{args.end_year}"
        map_output = os.path.join(
            args.outdir, f"seasonal_3D_minus_PP_maps_{suffix}.png"
        )
        histogram_output = os.path.join(
            args.outdir, f"seasonal_3D_minus_PP_histograms_{suffix}.png"
        )
        plot_maps(
            lon,
            lat,
            seasonal,
            pair_specifications,
            map_output,
            args.dpi,
        )
        plot_histograms(
            seasonal,
            pair_specifications,
            histogram_output,
            args.dpi,
        )

    metrics_output = os.path.join(
        args.outdir,
        f"seasonal_3D_minus_PP_metrics_{args.start_year}_{args.end_year}.csv",
    )
    write_metrics(seasonal, specifications, metrics_output)


if __name__ == "__main__":
    main()
