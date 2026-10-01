#!/usr/bin/env python3
"""Plot seasonal HRU-level 3D-minus-PP differences against terrain orientation.

Terrain parameters and outputs are paired directly by CID and HRU. No terrain
VRT, reprojection, or spatial resampling is used. CIDs run in parallel.
"""

import argparse
import csv
import glob
import os
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np

SEASONS = {
    "Winter": (12, 1, 2), "Spring": (3, 4, 5),
    "Summer": (6, 7, 8), "Fall": (9, 10, 11),
}
PP_NAME = {"swdn_3d": "swdn", "lwdn_3d": "lwdn"}
UNITS = {
    "trad": "K", "t2mb": "K", "t2mv": "K", "tv": "K",
    "swdn": "W m⁻²", "swdn_3d": "W m⁻²", "swnet": "W m⁻²",
    "lwdn": "W m⁻²", "lwdn_3d": "W m⁻²", "lwnet": "W m⁻²",
    "sh": "W m⁻²", "lh": "W m⁻²", "smc": "m³ m⁻³",
    "totsmc": "m³ m⁻³", "swe": "mm", "snowh": "m",
    "fsno": "unitless", "salb": "unitless", "snow_t": "K",
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edir-3d", required=True)
    p.add_argument("--edir-pp", required=True)
    p.add_argument("--variables", nargs="+", required=True)
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--seasons", nargs="+", choices=tuple(SEASONS),
                   default=list(SEASONS))
    p.add_argument("--cids", nargs="+", type=int, default=None,
                   help="Optional subset; default is every common CID")
    p.add_argument("--soil-layer", type=int, default=0)
    p.add_argument("--snow-layer", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--outdir", required=True)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def discover_cids(args):
    def available(root):
        return {
            int(e.name) for e in os.scandir(root)
            if e.is_dir() and e.name.isdigit()
            and os.path.isfile(os.path.join(e.path, "input_file.nc"))
        }

    common = available(args.edir_3d) & available(args.edir_pp)
    if args.cids is not None:
        missing = sorted(set(args.cids) - common)
        if missing:
            raise FileNotFoundError(f"CIDs unavailable in both runs: {missing}")
        common = set(args.cids)
    if not common:
        raise FileNotFoundError("No common CIDs with input_file.nc")
    return sorted(common)


def axis_index(dimensions, candidates):
    lower = [x.lower() for x in dimensions]
    return next((lower.index(x) for x in candidates if x in lower), None)


def read_output(group, name, soil_layer, snow_layer):
    if name not in group.variables:
        raise KeyError(f"{name!r} missing; available: {list(group.variables)}")
    variable = group.variables[name]
    data = np.ma.asarray(variable[:]).filled(np.nan).astype(float)
    dims = list(variable.dimensions)
    ta = axis_index(dims, ("time", "t"))
    ha = axis_index(dims, ("hru",))
    if ta is None or ha is None:
        raise ValueError(f"Cannot identify time/hru axes for {name}: {dims}")
    extras = [i for i in range(data.ndim) if i not in (ta, ha)]
    if len(extras) > 1:
        raise ValueError(f"More than one extra dimension for {name}: {dims}")
    if extras:
        layer_axis = extras[0]
        layer = soil_layer if name == "smc" else (
            snow_layer if name == "snow_t" else 0
        )
        if not 0 <= layer < data.shape[layer_axis]:
            raise IndexError(f"Invalid layer {layer} for {name}: {data.shape}")
        data = np.take(data, layer, axis=layer_axis)
        dims.pop(layer_axis)
    return np.transpose(data, (
        axis_index(dims, ("time", "t")), axis_index(dims, ("hru",)),
    ))


def output_file(root, cid, start_year):
    directory = os.path.join(root, "output_data", str(cid))
    preferred = os.path.join(directory, f"{start_year}-01-01.nc")
    if os.path.isfile(preferred):
        return preferred
    files = sorted(glob.glob(os.path.join(directory, "*.nc")))
    if len(files) == 1:
        return files[0]
    raise FileNotFoundError(
        f"CID {cid}: expected {preferred} or one NetCDF file in {directory}"
    )


def time_components(time_variable):
    if not hasattr(time_variable, "units"):
        raise ValueError("The meteorology time variable has no units attribute")
    dates = nc.num2date(
        time_variable[:],
        units=time_variable.units,
        calendar=getattr(time_variable, "calendar", "standard"),
        only_use_cftime_datetimes=True,
    )
    years = np.fromiter((date.year for date in dates), dtype=np.int16)
    months = np.fromiter((date.month for date in dates), dtype=np.int8)
    return years, months


def process_cid(task):
    cid, cfg = task
    variables, seasons = cfg["variables"], cfg["seasons"]
    input_3d = os.path.join(cfg["edir_3d"], str(cid), "input_file.nc")
    input_pp = os.path.join(cfg["edir_pp"], str(cid), "input_file.nc")

    with nc.Dataset(input_3d) as ds:
        p = ds.groups["parameters"]
        slope = np.ma.asarray(p.variables["slope"][:]).filled(np.nan).astype(float)
        x_aspect = np.ma.asarray(p.variables["x_aspect"][:]).filled(np.nan).astype(float)
        y_aspect = np.ma.asarray(p.variables["y_aspect"][:]).filled(np.nan).astype(float)
        hru_3d = np.asarray(p.variables["hru"][:])
        years, months = time_components(ds.groups["meteorology"].variables["time"])
    with nc.Dataset(input_pp) as ds:
        hru_pp = np.asarray(ds.groups["parameters"].variables["hru"][:])
    if not np.array_equal(hru_3d, hru_pp):
        raise ValueError(f"CID {cid}: 3D and PP HRU identifiers/order differ")

    predictors = {
        "east_west": np.sin(slope) * x_aspect,
        "north_south": np.sin(slope) * y_aspect,
    }

    file_3d = output_file(cfg["edir_3d"], cid, cfg["start_year"])
    file_pp = output_file(cfg["edir_pp"], cid, cfg["start_year"])
    period = (years >= cfg["start_year"]) & (years <= cfg["end_year"])
    if not np.any(period):
        raise ValueError(
            f"CID {cid}: input time axis contains no dates from "
            f"{cfg['start_year']} through {cfg['end_year']}"
        )

    with nc.Dataset(file_3d) as d3, nc.Dataset(file_pp) as dp:
        g3, gp = d3.groups["data"], dp.groups["data"]
        differences = {}
        for variable in variables:
            a = read_output(g3, variable, cfg["soil_layer"], cfg["snow_layer"])
            b = read_output(
                gp, PP_NAME.get(variable, variable),
                cfg["soil_layer"], cfg["snow_layer"],
            )
            if a.shape != b.shape or a.shape[1] != hru_3d.size:
                raise ValueError(
                    f"CID {cid}, {variable}: incompatible shapes "
                    f"3D={a.shape}, PP={b.shape}, terrain={hru_3d.size}"
                )
            if a.shape[0] != years.size:
                raise ValueError(
                    f"CID {cid}, {variable}: output has {a.shape[0]} times "
                    f"but input time axis has {years.size}"
                )
            difference = a - b
            finite = np.isfinite(a) & np.isfinite(b)
            differences[variable] = {}
            for season in seasons:
                chosen = period & np.isin(months, SEASONS[season])
                valid = finite & chosen[:, None]
                total = np.sum(np.where(valid, difference, 0.0), axis=0)
                count = np.sum(valid, axis=0)
                mean = np.full(hru_3d.size, np.nan)
                np.divide(total, count, out=mean, where=count > 0)
                differences[variable][season] = mean
    return predictors, differences


def regression(x, y):
    good = np.isfinite(x) & np.isfinite(y)
    x, y = x[good], y[good]
    if x.size < 2 or np.std(x) == 0:
        return x, y, np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    return x, y, slope, intercept, np.corrcoef(x, y)[0, 1]


def combine(results, args):
    return {
        "predictors": {
            name: np.concatenate([r[0][name] for r in results])
            for name in ("east_west", "north_south")
        },
        "differences": {
            variable: {
                season: np.concatenate([r[1][variable][season] for r in results])
                for season in args.seasons
            } for variable in args.variables
        },
        "cid_count": len(results),
    }


def plot_variable(variable, data, args):
    unit = UNITS.get(variable, "model units")
    tag = f"{args.start_year}_{args.end_year}"
    records = []
    specs = (
        ("east_west", r"sin(slope) × x$_{aspect}$", "East–West Terrain Component"),
        ("north_south", r"sin(slope) × y$_{aspect}$", "North–South Terrain Component"),
    )
    finite_y = np.concatenate([
        data["differences"][variable][s][
            np.isfinite(data["differences"][variable][s])
        ] for s in args.seasons
    ])
    ymin, ymax = finite_y.min(), finite_y.max()
    pad = 0.05 * (ymax - ymin)
    ylim = ymin - pad, ymax + pad

    for predictor_name, xlabel, component_title in specs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for ax, season in zip(axes.flat, args.seasons):
            x, y, slope, intercept, r = regression(
                data["predictors"][predictor_name],
                data["differences"][variable][season],
            )
            ax.scatter(x, y, s=4, alpha=0.20,
                       edgecolors="none", rasterized=True)
            ax.axhline(0, color="black", linestyle="--", linewidth=1.5)
            if np.isfinite(slope):
                xmin, xmax = x.min(), x.max()
                line_x = np.array([xmin, xmax])
                ax.plot(line_x, intercept + slope * line_x,
                        color="darkred", linewidth=2.5)
                xpad = 0.05 * (xmax - xmin)
                ax.set_xlim(xmin - xpad, xmax + xpad)
            ax.set_ylim(ylim)
            ax.set_title(season, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(f"{variable}: 3D − PP [{unit}]")
            ax.text(
                0.98, 0.98,
                f"slope = {slope:.3g}\nr = {r:.3f}\nn = {x.size:,}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(boxstyle="round", facecolor="lightgray",
                          edgecolor="black", alpha=0.6),
            )
            records.append((variable, season, predictor_name,
                            slope, intercept, r, x.size))
        for ax in axes.flat[len(args.seasons):]:
            ax.set_visible(False)
        fig.suptitle(
            f"Seasonal {variable} 3D − PP vs. {component_title}\n"
            f"{args.start_year}–{args.end_year}; {data['cid_count']:,} CIDs"
        )
        fig.subplots_adjust(left=0.10, right=0.98, bottom=0.09, top=0.90,
                            hspace=0.30, wspace=0.24)
        image = os.path.join(
            args.outdir,
            f"seasonal_{variable}_3D_minus_PP_{predictor_name}_scatter_{tag}.png",
        )
        fig.savefig(image, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print(f"Saved: {image}", flush=True)

    table = os.path.join(
        args.outdir, f"seasonal_{variable}_3D_minus_PP_terrain_metrics_{tag}.csv"
    )
    with open(table, "w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variable", "season", "predictor",
                         "regression_slope", "intercept", "r", "n"))
        writer.writerows(records)
    print(f"Saved: {table}", flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("end-year must not precede start-year")
    if args.workers < 1:
        raise ValueError("workers must be positive")
    args.edir_3d = os.path.abspath(args.edir_3d)
    args.edir_pp = os.path.abspath(args.edir_pp)
    args.outdir = os.path.abspath(args.outdir)
    os.makedirs(args.outdir, exist_ok=True)
    cids = discover_cids(args)
    print(
        f"Processing {len(cids):,} CIDs, {args.start_year}–{args.end_year}, "
        f"{len(args.variables)} variable(s), {args.workers} workers",
        flush=True,
    )
    tasks = [(cid, vars(args).copy()) for cid in cids]
    if args.workers == 1:
        iterator, pool = map(process_cid, tasks), None
    else:
        pool = ProcessPoolExecutor(max_workers=args.workers)
        iterator = pool.map(process_cid, tasks, chunksize=1)
    results = []
    try:
        for number, result in enumerate(iterator, 1):
            results.append(result)
            if number % 25 == 0 or number == len(cids):
                print(f"Completed {number:,}/{len(cids):,} CIDs", flush=True)
    finally:
        if pool is not None:
            pool.shutdown()

    data = combine(results, args)
    print(f"Combined {data['predictors']['east_west'].size:,} HRUs", flush=True)
    for variable in args.variables:
        plot_variable(variable, data, args)


if __name__ == "__main__":
    main()
