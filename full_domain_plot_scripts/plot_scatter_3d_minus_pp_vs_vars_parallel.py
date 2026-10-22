#!/usr/bin/env python3
"""Plot seasonal HRU-level 3D-minus-PP differences against terrain orientation."""

import argparse
import csv
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio

SEASONS = {
    "Winter": (12, 1, 2), "Spring": (3, 4, 5),
    "Summer": (6, 7, 8), "Fall": (9, 10, 11),
}

UNITS = {
    "trad": "K", "t2mb": "K", "t2mv": "K", "tv": "K",
    "swdn": "W m⁻²", "swdn_3d": "W m⁻²", "swnet": "W m⁻²",
    "lwdn": "W m⁻²", "lwdn_3d": "W m⁻²", "lwnet": "W m⁻²",
    "rnet": "W m⁻²", "sh": "W m⁻²", "lh": "W m⁻²",
    "smc": "m³ m⁻³", "totsmc": "m³ m⁻³", "swe": "mm",
    "snowh": "m", "fsno": "fraction", "salb": "fraction", "snow_t": "K",
}


def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--edir-3d", required=True, type=Path)
    p.add_argument("--edir-pp", required=True, type=Path)
    p.add_argument("--variables", nargs="+", required=True, choices=tuple(UNITS))
    p.add_argument("--start-year", type=int, default=2014)
    p.add_argument("--end-year", type=int, default=2024)
    p.add_argument("--seasons", nargs="+", choices=tuple(SEASONS), default=list(SEASONS))
    p.add_argument("--cids", nargs="+", type=int)
    p.add_argument("--soil-layer", type=int, default=0)
    p.add_argument("--snow-layer", type=int, default=0)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--time-chunk", type=int, default=128)
    p.add_argument("--outdir", required=True, type=Path)
    p.add_argument("--dpi", type=int, default=300)
    return p.parse_args()


def clean(variable, selection=slice(None)):
    data = np.ma.asarray(variable[selection], dtype=float).filled(np.nan)
    for attr in ("_FillValue", "missing_value"):
        for fill in np.atleast_1d(getattr(variable, attr, [])):
            data[data == fill] = np.nan
    data[(data == -9999) | ~np.isfinite(data)] = np.nan
    return data


def available_cids(root):
    return {
        int(p.name) for p in root.iterdir()
        if p.is_dir() and p.name.isdigit() and (p / "input_file.nc").is_file()
    }


def discover_cids(args):
    c3, cp = available_cids(args.edir_3d), available_cids(args.edir_pp)
    if args.cids is not None:
        requested = set(args.cids)
        missing = requested - (c3 & cp)
        if missing:
            raise ValueError(f"Requested CIDs unavailable: {sorted(missing)}")
        if not requested:
            raise ValueError("No CIDs selected")
        return sorted(requested)

    if c3 != cp:
        raise ValueError(f"CID sets differ: missing PP={sorted(c3-cp)}, missing 3D={sorted(cp-c3)}")
    if not c3:
        raise ValueError("No CIDs found")
    return sorted(c3)


def read_terrain(root, cid):
    with nc.Dataset(root / str(cid) / "input_file.nc") as ds:
        p = ds.groups["parameters"]
        return {name: clean(p.variables[name]) for name in ("slope", "x_aspect", "y_aspect")}


def read_map(root, cid):
    with rasterio.open(root / str(cid) / "hru_mapping_latlon.tif") as ds:
        data = ds.read(1, masked=True).astype(float).filled(np.nan)
        signature = (ds.shape, ds.crs, ds.transform)
    data[data == -9999] = np.nan
    return data, signature


def validate_pair(root3, rootp, cid):
    # Matching HRU labels alone do not establish identical geographic clusters.
    p3, pp = read_terrain(root3, cid), read_terrain(rootp, cid)
    for name in p3:
        if p3[name].shape != pp[name].shape or not np.allclose(
            p3[name], pp[name], rtol=0, atol=1e-6, equal_nan=True
        ):
            raise ValueError(f"CID {cid}: differing terrain parameter {name}")

    m3, s3 = read_map(root3, cid)
    mp, sp = read_map(rootp, cid)
    if s3 != sp or not np.array_equal(m3, mp, equal_nan=True):
        raise ValueError(f"CID {cid}: HRU maps differ; geographic overlap mapping is required")

    nhru = p3["slope"].size
    for values in (m3, mp):
        valid = np.isfinite(values)
        invalid = valid & ((values < 0) | (values >= nhru) | (values != np.floor(values)))
        if invalid.any():
            raise ValueError(f"CID {cid}: invalid zero-based HRU map")
    return p3


def output_index(root, cid, start_year, end_year):
    # Read actual output clocks rather than assigning the input meteorology clock.
    records = {}
    for filename in sorted((root / "output_data" / str(cid)).glob("*.nc")):
        with nc.Dataset(filename) as ds:
            t = ds.groups["metadata"].variables["date"]
            dates = nc.num2date(t[:], t.units, calendar=getattr(t, "calendar", "standard"))
            for i, d in enumerate(dates):
                if start_year <= d.year <= end_year:
                    stamp = (d.year, d.month, d.day, d.hour, d.minute, d.second)
                    if stamp in records:
                        raise ValueError(f"CID {cid}: duplicate output date {stamp}")
                    records[stamp] = (filename, i)

    if not records:
        raise ValueError(f"CID {cid}: no requested output in {root}")
    return records


def axis_index(dimensions, candidates):
    lower = [name.lower() for name in dimensions]
    return next((lower.index(name) for name in candidates if name in lower), None)


def read_field(group, name, rows, cfg):
    if name not in group.variables:
        raise KeyError(f"Missing output variable {name!r}")

    variable = group.variables[name]
    dims = list(variable.dimensions)
    ta, ha = axis_index(dims, ("time", "t")), axis_index(dims, ("hru",))
    if ta is None or ha is None:
        raise ValueError(f"Cannot identify time/HRU axes: {name}, {dims}")

    extras = [i for i in range(len(dims)) if i not in (ta, ha)]
    if len(extras) > 1:
        raise ValueError(f"Too many extra dimensions: {name}, {dims}")

    selection = [slice(None)] * len(dims)
    selection[ta] = rows
    if extras:
        axis = extras[0]
        layer = cfg["soil_layer"] if name == "smc" else cfg["snow_layer"] if name == "snow_t" else 0
        if not 0 <= layer < variable.shape[axis]:
            raise IndexError(f"Invalid layer {layer} for {name}")
        selection[axis] = layer

    data = clean(variable, tuple(selection))
    retained = [d for i, d in enumerate(dims) if i not in extras]
    data = np.transpose(data, (axis_index(retained, ("time", "t")), axis_index(retained, ("hru",))))

    if name in ("salb", "fsno"):
        data[(data < 0) | (data > 1)] = np.nan
    return data


def read_quantity(group, requested, rows, cfg, is_pp):
    # Saved LWnet is positive upward, so surface net radiation uses subtraction.
    if requested == "rnet":
        return read_field(group, "swnet", rows, cfg) - read_field(group, "lwnet", rows, cfg)

    # swdn_3d is read from BOTH runs: PP has filtering but no terrain correction.
    actual = "lwdn" if is_pp and requested == "lwdn_3d" else requested
    return read_field(group, actual, rows, cfg)


def process_cid(task):
    cid, cfg = task
    root3, rootp = cfg["edir_3d"], cfg["edir_pp"]
    terrain = validate_pair(root3, rootp, cid)
    nhru = terrain["slope"].size

    # Stored slope is a gradient; x/y aspects are mean sine/cosine components.
    slope_angle = np.arctan(terrain["slope"])
    predictors = {
        "east_west": np.sin(slope_angle) * terrain["x_aspect"],
        "south_north": np.sin(slope_angle) * terrain["y_aspect"],
    }
    invalid = (
        ~np.isfinite(terrain["slope"]) | (terrain["slope"] < 0)
        | ~np.isfinite(terrain["x_aspect"]) | ~np.isfinite(terrain["y_aspect"])
    )
    for values in predictors.values():
        values[invalid] = np.nan

    i3 = output_index(root3, cid, cfg["start_year"], cfg["end_year"])
    ip = output_index(rootp, cid, cfg["start_year"], cfg["end_year"])
    common = sorted(set(i3) & set(ip))
    if not common:
        raise ValueError(f"CID {cid}: no common output timestamps")

    dates = [datetime(*stamp) for stamp in common]
    if len(dates) > 1:
        gaps = [(b-a).total_seconds() for a, b in zip(dates, dates[1:])]
        if len(set(gaps)) != 1:
            raise ValueError(f"CID {cid}: common timestamps contain gaps or unequal steps")

    groups = {}
    for stamp in common:
        f3, r3 = i3[stamp]
        fp, rp = ip[stamp]
        groups.setdefault((f3, fp), []).append((stamp[1], r3, rp))

    variables, seasons = cfg["variables"], cfg["seasons"]
    shape = (len(variables), len(seasons), nhru)
    sums, counts = np.zeros(shape), np.zeros(shape, dtype=np.int64)

    for (f3, fp), records in groups.items():
        with nc.Dataset(f3) as d3, nc.Dataset(fp) as dp:
            for first in range(0, len(records), cfg["time_chunk"]):
                block = records[first:first + cfg["time_chunk"]]
                months = np.array([r[0] for r in block])
                rows3 = np.array([r[1] for r in block], dtype=int)
                rowsp = np.array([r[2] for r in block], dtype=int)

                for vi, variable in enumerate(variables):
                    a = read_quantity(d3.groups["data"], variable, rows3, cfg, False)
                    b = read_quantity(dp.groups["data"], variable, rowsp, cfg, True)
                    if a.shape != (len(block), nhru) or b.shape != a.shape:
                        raise ValueError(f"CID {cid}, {variable}: incompatible output dimensions")

                    diff = a - b
                    valid = np.isfinite(a) & np.isfinite(b)
                    for si, season in enumerate(seasons):
                        ok = valid & np.isin(months, SEASONS[season])[:, None]
                        sums[vi, si] += np.where(ok, diff, 0).sum(axis=0)
                        counts[vi, si] += ok.sum(axis=0)

    means = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)

    report = {
        "cid": cid, "common": len(common), "first": common[0], "last": common[-1],
        "unpaired3": len(set(i3)-set(ip)), "unpairedp": len(set(ip)-set(i3)),
    }
    return predictors, means, report


def regression(x, y):
    valid = np.isfinite(x) & np.isfinite(y)
    x, y = x[valid], y[valid]
    if x.size < 2 or np.ptp(x) == 0:
        return x, y, np.nan, np.nan, np.nan
    slope, intercept = np.polyfit(x, y, 1)
    r = np.corrcoef(x, y)[0, 1] if np.ptp(y) > 0 else np.nan
    return x, y, slope, intercept, r


def plot_variable(variable, vi, predictors, means, args, ncids):
    values = means[vi]
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError(f"No finite seasonal differences for {variable}")

    ymin, ymax = float(finite.min()), float(finite.max())
    pad = max(0.05 * (ymax-ymin), 1e-6)
    unit, tag = UNITS[variable], f"{args.start_year}_{args.end_year}"
    records = []

    specs = (
        ("east_west", r"$\sin(\theta_s)\,\overline{\sin(a)}$", "East–West component (+ East)"),
        ("south_north", r"$\sin(\theta_s)\,\overline{\cos(a)}$", "South–North component (+ South)"),
    )
    ylabel = f"{variable}: 3D − PP [{unit}]"
    if variable == "lwnet":
        ylabel += "\nPositive upward"

    for name, xlabel, title in specs:
        fig, axes = plt.subplots(2, 2, figsize=(11, 9))
        for si, season in enumerate(args.seasons):
            ax = axes.flat[si]
            x, y, slope, intercept, r = regression(predictors[name], values[si])
            ax.scatter(x, y, s=4, alpha=0.20, edgecolors="none", rasterized=True)
            ax.axhline(0, color="black", linestyle="--", linewidth=1)

            if np.isfinite(slope):
                xx = np.array([x.min(), x.max()])
                ax.plot(xx, intercept+slope*xx, color="darkred", linewidth=2)
                xpad = max(0.05*np.ptp(x), 1e-6)
                ax.set_xlim(x.min()-xpad, x.max()+xpad)

            ax.set_ylim(ymin-pad, ymax+pad)
            ax.set_title(season, fontweight="bold")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.grid(alpha=0.15)
            ax.text(
                0.98, 0.98, f"slope = {slope:.3g}\nr = {r:.3f}\nn = {len(x):,}",
                transform=ax.transAxes, ha="right", va="top",
                bbox=dict(facecolor="white", edgecolor="0.7", alpha=0.85)
            )
            records.append((variable, season, name, slope, intercept, r, len(x)))

        for ax in axes.flat[len(args.seasons):]:
            ax.set_visible(False)

        fig.suptitle(
            f"Seasonal {variable}: 3D − PP versus {title}\n"
            f"{args.start_year}–{args.end_year}; {ncids:,} CIDs; equally weighted HRUs",
            fontsize=14
        )
        fig.subplots_adjust(left=0.11, right=0.98, bottom=0.09, top=0.89, hspace=0.35, wspace=0.28)
        output = args.outdir / f"seasonal_{variable}_3D_minus_PP_{name}_scatter_{tag}.png"
        fig.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("Saved:", output, flush=True)

    table = args.outdir / f"seasonal_{variable}_3D_minus_PP_terrain_metrics_{tag}.csv"
    with table.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("variable", "season", "predictor", "regression_slope", "intercept", "r", "n"))
        writer.writerows(records)
    print("Saved:", table, flush=True)


def main():
    args = arguments()
    if args.end_year < args.start_year:
        raise ValueError("Invalid year range")
    if min(args.workers, args.time_chunk, args.dpi) < 1:
        raise ValueError("Workers, chunk size and DPI must be positive")
    if min(args.soil_layer, args.snow_layer) < 0:
        raise ValueError("Layer indices must be nonnegative")
    if len(set(args.seasons)) != len(args.seasons) or len(set(args.variables)) != len(args.variables):
        raise ValueError("Repeated season or variable names")

    args.outdir.mkdir(parents=True, exist_ok=True)
    cids = discover_cids(args)
    tasks = [(cid, vars(args).copy()) for cid in cids]
    results = []

    if args.workers == 1:
        for result in map(process_cid, tasks):
            results.append(result)
            print(result[2], flush=True)
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            for result in pool.map(process_cid, tasks, chunksize=1):
                results.append(result)
                print(result[2], flush=True)

    # Each CID must cover the same regular common timestamp sequence.
    coverage = {(r[2]["common"], r[2]["first"], r[2]["last"]) for r in results}
    if len(coverage) != 1:
        raise ValueError("Common output coverage differs across CIDs")

    predictors = {
        name: np.concatenate([r[0][name] for r in results])
        for name in ("east_west", "south_north")
    }
    means = np.concatenate([r[1] for r in results], axis=2)
    print(f"Combined {len(cids):,} CIDs and {len(predictors['east_west']):,} HRUs", flush=True)

    if "swdn" in args.variables:
        print("Note: swdn compares raw forcing. Use swdn_3d for internally calculated shortwave.", flush=True)

    for vi, variable in enumerate(args.variables):
        plot_variable(variable, vi, predictors, means, args, len(cids))


if __name__ == "__main__":
    main()