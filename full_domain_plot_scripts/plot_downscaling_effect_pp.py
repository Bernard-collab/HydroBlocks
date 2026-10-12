#!/usr/bin/env python3
"""Compare PP downscaled minus PP without downscaling (whole domain).

Requires numpy, matplotlib, netCDF4. Inputs are matched daily NetCDF files
under <experiment>/postprocess/output_dir, containing time/lat/lon variables.
Radiation inputs use PP swdn and lwdn in BOTH experiments.

Outputs: seasonal paired maps, density histograms, and all-pixel metrics CSV.
Seasonal means pool all matched valid timesteps within each calendar season
across the selected years; winter pools December, January and February.
No snow-cover screening is applied. Each valid grid pixel has equal weight.
Histogram densities integrate to one within the visible range; annotations
and CSV report the omitted fraction. Statistics use the complete arrays.
Correlation is spatial correlation of seasonal means, not a significance test.
State variables retain units; rates are averaged, never integrated to totals.

The experiment settings must match except for meteorological downscaling.
This script measures model-output response, not accuracy against observations
and not whether the input-preparation downscaling flag was enabled.

Example:
  python plot_downscaling_effect_pp.py --workers 8 --dpi 300
"""
import argparse
import csv
import glob
import multiprocessing as mp
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
# Plot figure text specifications
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 15,
    "axes.titlesize": 17,
    "axes.titleweight": "bold",
    "axes.labelsize": 15,
    "xtick.labelsize": 13,
    "ytick.labelsize": 13,
})
import netCDF4 as nc
import numpy as np
SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}
BASE_DIR = "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations"
DEFAULT_EDIR_DOWNSCALED = os.path.join(BASE_DIR, "MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test")
DEFAULT_EDIR_BASELINE = os.path.join(BASE_DIR, "MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_no_downscaled_test")
DEFAULT_OUTDIR = "./validation_plots/downscaled_minus_no_downscaling"

VARIABLES = (
    {
        "name": "swdn",
        "downscaled": ("swdn",),
        "baseline": ("swdn",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "swnet",
        "downscaled": ("swnet",),
        "baseline": ("swnet",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lwdn",
        "downscaled": ("lwdn",),
        "baseline": ("lwdn",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lwnet",
        "downscaled": ("lwnet",),
        "baseline": ("lwnet",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "totsmc",
        "downscaled": ("totsmc",),
        "baseline": ("totsmc",),
        "unit": "m$^3$ m$^{-3}$",
    },
    {
        "name": "smc",
        "downscaled": ("smc",),
        "baseline": ("smc",),
        "unit": "m$^3$ m$^{-3}$",
    },
    {
        "name": "swe",
        "downscaled": ("swe",),
        "baseline": ("swe",),
        "unit": "mm",
    },
    {
        "name": "snowh",
        "downscaled": ("snowh",),
        "baseline": ("snowh",),
        "unit": "m",
    },
    {
        "name": "fsno",
        "downscaled": ("fsno",),
        "baseline": ("fsno",),
        "unit": "unitless",
    },
    {
        "name": "qsnow",
        "downscaled": ("qsnow",),
        "baseline": ("qsnow",),
        "unit": "mm s$^{-1}$",
    },
    {
        "name": "snow_t",
        "downscaled": ("snow_t",),
        "baseline": ("snow_t",),
        "unit": "K",
    },
    {
        "name": "t2mb",
        "downscaled": ("t2mb",),
        "baseline": ("t2mb",),
        "unit": "K",
    },
    {
        "name": "t2mv",
        "downscaled": ("t2mv",),
        "baseline": ("t2mv",),
        "unit": "K",
    },
    {
        "name": "trad",
        "downscaled": ("trad",),
        "baseline": ("trad",),
        "unit": "K",
    },
    {
        "name": "tv",
        "downscaled": ("tv",),
        "baseline": ("tv",),
        "unit": "K",
    },
    {
        "name": "sh",
        "downscaled": ("sh",),
        "baseline": ("sh",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "lh",
        "downscaled": ("lh",),
        "baseline": ("lh",),
        "unit": "W m$^{-2}$",
    },
    {
        "name": "salb",
        "downscaled": ("salb",),
        "baseline": ("salb",),
        "unit": "unitless",
    },
    {
        "name": "qsurface",
        "downscaled": ("qsurface",),
        "baseline": ("qsurface",),
        "unit": "mm s$^{-1}$",
    },
    {
        "name": "runoff",
        "downscaled": ("runoff",),
        "baseline": ("runoff",),
        "unit": "mm s$^{-1}$",
    },
)
PAIRS = (
    ("shortwave", "swdn", "swnet"),
    ("longwave", "lwdn", "lwnet"),
    ("soil_moisture", "smc", "totsmc"),
    ("snow_amount", "swe", "snowh"),
    ("snow_state", "fsno", "qsnow"),
    ("near_surface_temperature", "t2mb", "t2mv"),
    ("surface_temperature", "trad", "tv"),
    ("turbulent_fluxes", "sh", "lh"),
    ("surface_water", "qsurface", "runoff"),
    ("surface_temperature_albedo", "trad", "salb"),
)
# Fixed symmetric display limits; adjust for this experiment as needed.
# These limits affect map colors only; values outside them saturate at the
# endpoint color and remain available to the histogram and statistics.
FIXED_MAP_LIMITS = {
    "swdn": 30.0,
    "swnet": 30.0,
    "lwdn": 30.0,
    "lwnet": 30.0,
    "t2mb": 0.15,
    "t2mv": 1.0,
    "trad": 1.0,
    "tv": 1.0,
    "sh": 30.0,
    "lh": 3.0,
    "fsno": 0.03,
    "qsnow": 2.0e-7,
    "snowh": 0.05,
    "smc": 0.03,
    "totsmc": 0.01,
    "swe": 10.0,
    "salb": 0.10,
    "qsurface": 0.002,
    "runoff": 0.005,
}
# Fixed symmetric histogram limits. Each variable uses the same bin edges in
# all four seasons, which makes the seasonal distributions directly comparable.
# These limits affect the displayed histogram only; metrics use all valid data.
FIXED_HISTOGRAM_LIMITS = {
    "swdn": 30.0,
    "swnet": 30.0,
    "lwdn": 30.0,
    "lwnet": 30.0,
    "totsmc": 0.01,
    "smc": 0.04,
    "swe": 10.0,
    "snowh": 0.10,
    "fsno": 0.06,
    "qsnow": 2.0e-7,
    "snow_t": 3.0,
    "t2mb": 0.2,
    "t2mv": 1.0,
    "trad": 1.0,
    "tv": 4.0,
    "sh": 30.0,
    "lh": 3.0,
    "salb": 0.10,
    "qsurface": 0.002,
    "runoff": 0.005,
}


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--edir-downscaled", default=DEFAULT_EDIR_DOWNSCALED, help="PP downscaled experiment directory"
    )
    parser.add_argument(
        "--edir-baseline", default=DEFAULT_EDIR_BASELINE, help="PP experiment without downscaling"
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--outdir", default=DEFAULT_OUTDIR)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--border",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        help=(
            "Number of parallel daily-file readers. Defaults to "
            "SLURM_CPUS_PER_TASK, or 1 outside Slurm."
        ),
    )
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
# Find a requested PP variable in each NetCDF file.
# Returns (variable, name); raises KeyError listing available variables if none are found.


def find_variable(dataset, candidates, experiment_name):
    for candidate in candidates:
        if candidate in dataset.variables:
            return dataset.variables[candidate], candidate
    available = ", ".join(sorted(dataset.variables))
    raise KeyError(
        f"None of {candidates} found in {experiment_name}. "
        f"Available root variables: {available}"
    )
# Return the position of the first matching dimension name (case-insensitive), or None if absent.
# e.g. axis_index(("time", "hru", "soil"), ("soil",)) -> 2


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
            if soil_layer < 0 or soil_layer >= variable.shape[axis]:
                raise IndexError(
                    f"Requested layer {soil_layer} exceeds axis {name} of "
                    f"{variable.name}, whose size is {variable.shape[axis]}"
                )
            selection.append(soil_layer)
    data = np.ma.asarray(variable[tuple(selection)], dtype=float).filled(np.nan)
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
                "sum_downscaled": np.zeros(shape, dtype=np.float64),
                "sum_pp": np.zeros(shape, dtype=np.float64),
                "count": np.zeros(shape, dtype=np.int64),
            }
            for season in SEASONS
        }
        for variable in specifications
    }


def process_date_chunk(
    date_records,
    specifications,
    lat_order_downscaled,
    lon_order_downscaled,
    lat_order_pp,
    lon_order_pp,
    soil_layer,
):
    """Read one chunk of matched days and return partial seasonal sums."""
    partial = {specification["name"]: {} for specification in specifications}
    resolved_names = {}
    for key, file_downscaled, file_pp in date_records:
        date = datetime.strptime(key, "%Y%m%d")
        season = next(
            name for name, months in SEASONS.items() if date.month in months
        )
        with nc.Dataset(file_downscaled) as ds_downscaled, nc.Dataset(file_pp) as ds_pp:
            for specification in specifications:
                name = specification["name"]
                var_downscaled, actual_downscaled = find_variable(
                    ds_downscaled, specification["downscaled"], "Downscaled output"
                )
                var_pp, actual_pp = find_variable(
                    ds_pp, specification["baseline"], "No downscaling output"
                )
                resolved_names.setdefault(name, (actual_downscaled, actual_pp))
                data_downscaled = read_time_lat_lon(var_downscaled, soil_layer)
                data_pp = read_time_lat_lon(var_pp, soil_layer)
                if data_downscaled.shape != data_pp.shape:
                    raise ValueError(
                        f"Shape mismatch for {name} on {key}: "
                        f"Downscaled={data_downscaled.shape}, No downscaling={data_pp.shape}"
                    )
                data_downscaled = data_downscaled[:, lat_order_downscaled, :][:, :, lon_order_downscaled]
                data_pp = data_pp[:, lat_order_pp, :][:, :, lon_order_pp]
                valid = (
                    np.isfinite(data_downscaled)
                    & np.isfinite(data_pp)
                    & (data_downscaled > -9000.0)
                    & (data_pp > -9000.0)
                )
                if season not in partial[name]:
                    shape = data_downscaled.shape[1:]
                    partial[name][season] = {
                        "sum_downscaled": np.zeros(shape, dtype=np.float64),
                        "sum_pp": np.zeros(shape, dtype=np.float64),
                        "count": np.zeros(shape, dtype=np.int64),
                    }
                target = partial[name][season]
                target["sum_downscaled"] += np.sum(
                    np.where(valid, data_downscaled, 0.0), axis=0
                )
                target["sum_pp"] += np.sum(
                    np.where(valid, data_pp, 0.0), axis=0
                )
                target["count"] += np.sum(valid, axis=0)
    return partial, resolved_names, len(date_records)


def read_seasonal_means(args, specifications):
    files_downscaled = daily_files(args.edir_downscaled, args.start_year, args.end_year)
    files_pp = daily_files(args.edir_baseline, args.start_year, args.end_year)
    common_dates = sorted(set(files_downscaled) & set(files_pp))
    missing_downscaled = sorted(set(files_pp) - set(files_downscaled))
    missing_pp = sorted(set(files_downscaled) - set(files_pp))
    if not common_dates:
        raise RuntimeError("The Downscaled and No downscaling experiments have no matching daily files")
    print(
        f"Matched days: {len(common_dates)}; "
        f"missing from Downscaled: {len(missing_downscaled)}; missing from No downscaling: {len(missing_pp)}",
        flush=True,
    )
    with nc.Dataset(files_downscaled[common_dates[0]]) as ds_downscaled, nc.Dataset(
        files_pp[common_dates[0]]
    ) as ds_pp:
        lon_downscaled, lat_downscaled = read_coordinates(ds_downscaled)
        lon_pp, lat_pp = read_coordinates(ds_pp)
    if lon_downscaled.shape != lon_pp.shape or lat_downscaled.shape != lat_pp.shape:
        raise ValueError("Downscaled and No downscaling postprocessed grids have different dimensions")
    if not np.allclose(np.sort(lon_downscaled), np.sort(lon_pp)) or not np.allclose(
        np.sort(lat_downscaled), np.sort(lat_pp)
    ):
        raise ValueError("Downscaled and No downscaling postprocessed coordinate values do not match")
    lon_order_downscaled = np.argsort(lon_downscaled)
    lat_order_downscaled = np.argsort(lat_downscaled)
    lon_order_pp = np.argsort(lon_pp)
    lat_order_pp = np.argsort(lat_pp)
    lon = lon_downscaled[lon_order_downscaled]
    lat = lat_downscaled[lat_order_downscaled]
    accumulators = initialise_accumulators((lat.size, lon.size), specifications)
    resolved_names = {}
    workers = max(1, min(args.workers, len(common_dates)))
    records = [(key, files_downscaled[key], files_pp[key]) for key in common_dates]
    # Several chunks per worker balance slow files while limiting IPC overhead.
    chunk_size = max(1, (len(records) + workers * 4 - 1) // (workers * 4))
    chunks = [
        records[start : start + chunk_size]
        for start in range(0, len(records), chunk_size)
    ]
    print(
        f"Daily-file readers: {workers}; chunks: {len(chunks)}; "
        f"approximately {chunk_size} days/chunk",
        flush=True,
    )
    def merge_partial(partial, names):
        for name, resolved in names.items():
            if name not in resolved_names:
                resolved_names[name] = resolved
                print(
                    f"{name}: Downscaled variable={resolved[0]}; "
                    f"No downscaling variable={resolved[1]}",
                    flush=True,
                )
        for name, seasons in partial.items():
            for season, source in seasons.items():
                target = accumulators[name][season]
                target["sum_downscaled"] += source["sum_downscaled"]
                target["sum_pp"] += source["sum_pp"]
                target["count"] += source["count"]
    completed_days = 0
    if workers == 1:
        for chunk in chunks:
            partial, names, count = process_date_chunk(
                chunk,
                specifications,
                lat_order_downscaled,
                lon_order_downscaled,
                lat_order_pp,
                lon_order_pp,
                args.soil_layer,
            )
            merge_partial(partial, names)
            completed_days += count
            print(
                f"Processed {completed_days}/{len(common_dates)} matched days",
                flush=True,
            )
    else:
        context = mp.get_context("spawn")
        with ProcessPoolExecutor(
            max_workers=workers, mp_context=context
        ) as executor:
            futures = [
                executor.submit(
                    process_date_chunk,
                    chunk,
                    specifications,
                    lat_order_downscaled,
                    lon_order_downscaled,
                    lat_order_pp,
                    lon_order_pp,
                    args.soil_layer,
                )
                for chunk in chunks
            ]
            for future in as_completed(futures):
                partial, names, count = future.result()
                merge_partial(partial, names)
                completed_days += count
                print(
                    f"Processed {completed_days}/{len(common_dates)} matched days",
                    flush=True,
                )
    seasonal = {}
    for specification in specifications:
        name = specification["name"]
        seasonal[name] = {}
        for season in SEASONS:
            source = accumulators[name][season]
            mean_downscaled = np.full(source["count"].shape, np.nan)
            mean_pp = np.full(source["count"].shape, np.nan)
            np.divide(
                source["sum_downscaled"],
                source["count"],
                out=mean_downscaled,
                where=source["count"] > 0,
            )
            np.divide(
                source["sum_pp"],
                source["count"],
                out=mean_pp,
                where=source["count"] > 0,
            )
            seasonal[name][season] = {
                "downscaled": mean_downscaled,
                "baseline": mean_pp,
                "difference": mean_downscaled - mean_pp,
            }
    return lon, lat, seasonal


def metrics(downscaled, baseline):
    valid = np.isfinite(downscaled) & np.isfinite(baseline)
    values_downscaled = np.asarray(downscaled[valid], dtype=float)
    values_pp = np.asarray(baseline[valid], dtype=float)
    difference = values_downscaled - values_pp
    n_valid = difference.size
    if n_valid == 0:
        return difference, np.nan, np.nan, np.nan, np.nan, 0
    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    if n_valid >= 2 and np.std(values_downscaled) > 0 and np.std(values_pp) > 0:
        correlation = float(np.corrcoef(values_downscaled, values_pp)[0, 1])
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


def plot_maps(lon, lat, seasonal, specifications, output, dpi, border):
    """Plot paired downscaling differences with shared seasonal color limits."""
    if border < 0:
        raise ValueError("--border must be zero or greater")
    extent = [
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    ]
    # Frame the figure around the union of valid pixels, with a small border.
    valid_domain = None
    for specification in specifications:
        name = specification["name"]
        for season in SEASONS:
            finite = np.isfinite(seasonal[name][season]["difference"])
            valid_domain = finite if valid_domain is None else (valid_domain | finite)
    if valid_domain is None or not np.any(valid_domain):
        print(f"Skipping map: no valid pixels for {output}", flush=True)
        return
    valid_rows, valid_columns = np.where(valid_domain)
    valid_x = lon[valid_columns]
    valid_y = lat[valid_rows]
    xmin, xmax = float(np.nanmin(valid_x)), float(np.nanmax(valid_x))
    ymin, ymax = float(np.nanmin(valid_y)), float(np.nanmax(valid_y))
    pad_x = border * max(xmax - xmin, np.finfo(float).eps)
    pad_y = border * max(ymax - ymin, np.finfo(float).eps)
    plot_xlim = (xmin - pad_x, xmax + pad_x)
    plot_ylim = (ymin - pad_y, ymax + pad_y)
    fig, axes = plt.subplots(
        2, 4, figsize=(20, 9), sharex=False, sharey=False
    )
    cmap = plt.get_cmap("RdBu_r").copy()
    # Draw missing-data pixels in light gray.
    cmap.set_bad("0.94")
    fig.subplots_adjust(
        left=0.045,
        right=0.94,
        bottom=0.07,
        top=0.97,
        wspace=0.10,
        hspace=0.30,
    )
    for row, specification in enumerate(specifications):
        name = specification["name"]
        display_name = name
        unit = specification["unit"]
        differences = [seasonal[name][season]["difference"] for season in SEASONS]
        # Use a fixed paper-style range when defined. Values outside the range
        # are color-saturated, not removed from the data or calculations.
        limit = FIXED_MAP_LIMITS[name] if name in FIXED_MAP_LIMITS else display_limit(differences)
        norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            image = axis.imshow(
                np.ma.masked_invalid(seasonal[name][season]["difference"]),
                origin="lower",
                extent=extent,
                cmap=cmap,
                norm=norm,
                interpolation="nearest",
                resample=False,
                aspect="equal",
            )
            axis.set_title(f"{display_name} – {season}",fontweight="bold",)
            axis.set_xlabel("Longitude [°]")
            axis.set_ylabel("Latitude [°]")
            axis.set_xlim(plot_xlim)
            axis.set_ylim(plot_ylim)
        # Match the SMC/FSNO layout: one full-height colorbar beside each row.
        fall_position = axes[row, -1].get_position()
        colorbar_axis = fig.add_axes(
            [
                fall_position.x1 + 0.006,
                fall_position.y0,
                0.008,
                fall_position.height,
            ]
        )
        colorbar = fig.colorbar(
            image, cax=colorbar_axis, orientation="vertical",
            extend="both" if name in FIXED_MAP_LIMITS else "neither",
            extendrect=False,
        )
        colorbar.set_label(f"{display_name}: Downscaled − Baseline [{unit}]")
    # Intentionally omit an overall title to match the SMC/FSNO map figure.
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def plot_histograms(seasonal, specifications, output, dpi):
    """Plot two variables (rows) across four seasonal histograms (columns)."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 9))
    for row, specification in enumerate(specifications):
        name = specification["name"]
        display_name = name
        unit = specification["unit"]
        seasonal_differences = [
            seasonal[name][season]["difference"] for season in SEASONS
        ]
        # Use one fixed linear range and the same 50 bins for every season.
        # Fall back to the complete range only for variables not listed above.
        limit = (FIXED_HISTOGRAM_LIMITS[name] if name in FIXED_HISTOGRAM_LIMITS
                 else display_limit(seasonal_differences))
        bin_edges = np.linspace(-limit, limit, 51)
        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            difference, md, mad, rmse, correlation, n_valid = metrics(
                seasonal[name][season]["downscaled"], seasonal[name][season]["baseline"]
            )
            if n_valid == 0:
                axis.set_title(f"{season}: no valid pixels")
                axis.axis("off")
                continue
            # The plotted distribution is restricted to the fixed visible
            # range. Metrics above are calculated from the complete array.
            visible = difference[
                (difference >= -limit) & (difference <= limit)
            ]
            if visible.size:
                axis.hist(visible, bins=bin_edges, density=True)
            else:
                axis.text(
                    0.5, 0.5, "No values within display range",
                    transform=axis.transAxes, ha="center", va="center",
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
                    f"n = {n_valid:,}\n"
                    f"Outside range = {100 * (1 - visible.size / n_valid):.2f}%"
                ),
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=10,
                bbox={
                    "boxstyle": "round",
                    "facecolor": "lightgray",
                    "edgecolor": "black",
                    "alpha": 0.6,
                },
            )
            if row == 0:
                axis.set_title(season)
            if column == 0:
                axis.set_ylabel(f"{display_name}\nProbability density")
            axis.set_xlabel(f"Downscaled − Baseline [{unit}]")
    fig.subplots_adjust(left=0.07, right=0.99, bottom=0.08, top=0.91,
                        hspace=0.32, wspace=0.24)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {output}", flush=True)


def histogram_limit(seasonal, name):
    return (FIXED_HISTOGRAM_LIMITS[name] if name in FIXED_HISTOGRAM_LIMITS
            else display_limit([seasonal[name][season]["difference"] for season in SEASONS]))


def distribution_summary(difference, limit):
    if difference.size == 0:
        return [np.nan] * 5 + [limit, np.nan]
    percentiles = np.percentile(difference, [1, 50, 99])
    outside = 100.0 * np.count_nonzero(np.abs(difference) > limit) / difference.size
    return [float(difference.min()), float(difference.max()), *percentiles, limit, outside]


def write_metrics(seasonal, specifications, output):
    with open(output, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["variable", "season", "unit", "MD", "MAD", "RMSE", "r", "n", "min_difference", "max_difference", "p01", "p50", "p99", "histogram_limit", "outside_range_pct"]
        )
        for specification in specifications:
            name = specification["name"]
            for season in SEASONS:
                difference, md, mad, rmse, correlation, n_valid = metrics(
                    seasonal[name][season]["downscaled"],
                    seasonal[name][season]["baseline"],
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
                        *distribution_summary(difference, histogram_limit(seasonal, name)),
                    ]
                )
    print(f"Saved: {output}", flush=True)


def main():
    args = arguments()
    if args.workers < 1 or args.dpi < 1:
        raise ValueError("--workers and --dpi must be positive")
    if os.path.abspath(args.edir_downscaled) == os.path.abspath(args.edir_baseline):
        raise ValueError("The downscaled and baseline directories must differ")
    print("Difference: PP downscaled minus PP without downscaling", flush=True)
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
            args.outdir, f"seasonal_downscaled_minus_baseline_maps_{suffix}.png"
        )
        histogram_output = os.path.join(
            args.outdir, f"seasonal_downscaled_minus_baseline_histograms_{suffix}.png"
        )
        plot_maps(
            lon,
            lat,
            seasonal,
            pair_specifications,
            map_output,
            args.dpi,
            args.border,
        )
        plot_histograms(
            seasonal,
            pair_specifications,
            histogram_output,
            args.dpi,
        )
    metrics_output = os.path.join(
        args.outdir,
        f"seasonal_downscaled_minus_baseline_metrics_{args.start_year}_{args.end_year}.csv",
    )
    write_metrics(seasonal, specifications, metrics_output)


if __name__ == "__main__":
    main()
