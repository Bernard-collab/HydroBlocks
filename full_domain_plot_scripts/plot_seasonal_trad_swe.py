#!/usr/bin/env python3
"""Plot multi-year seasonal mean HydroBlocks TRAD and SWE.

The script reads daily NetCDF files from ``<experiment>/postprocess/output_dir``
and produces a 2-row by 4-column figure. Columns are Winter (DJF), Spring
(MAM), Summer (JJA), and Fall (SON). Row 1 is surface radiative temperature
(TRAD), and row 2 is snow water equivalent (SWE).

All valid 3-hourly records from 2014-2016 are averaged. Thus, the panels are
multi-year seasonal climatologies, not separate annual means. Masked values,
NaN, infinity, values at or below -9990, and values at or above 1e9 are
excluded cell by cell.

One common color scale is used for all four seasons of each variable. TRAD uses
the combined 1st and 99th percentiles. SWE uses 0 mm as its lower bound and the
combined 99th percentile as its upper bound to limit isolated extremes.

The noninteractive Matplotlib Agg backend makes the script suitable for a
headless SLURM compute node. It only reads existing postprocessed results and
does not modify model output, VRT files, mapping.tif, or input_file.nc.

Output:
    /scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/output_plots/
    seasonal_climatology_trad_swe_2014_2016_4x2.png
"""

import argparse
import glob
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np


# -----------------------------------------------------------------------------
# USER SETTINGS
# Change PATH when plotting a different HydroBlocks experiment. The experiment
# must already contain postprocess/output_dir/YYYYMMDD.nc files.
# -----------------------------------------------------------------------------

OUTPUTDIR = (
    "/scratch/alpine/battobrah@xsede.org/"
    "HydroBlocks_Enrico_dev/output_plots"
)

DEFAULT_PATH = (
    "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/"
    "upper_colorado/experiments/simulations/"
    "MSWX_3h_Snow_Project3_pp_upper_colorado_3yr_2hru_"
    "full_domain_no_downscale"
)

YEARS = [2014, 2015, 2016]
VARIABLES = ["trad", "swe"]

SEASONS = {
    "Winter": (12, 1, 2),
    "Spring": (3, 4, 5),
    "Summer": (6, 7, 8),
    "Fall": (9, 10, 11),
}

PLOT_SETTINGS = {
    "trad": {
        "row": 0,
        "cmap": "RdYlBu_r",
        "label": "TRAD [K]",
        "title": "Surface Radiative Temperature",
    },
    "swe": {
        "row": 1,
        "cmap": "PuBu",
        "label": "SWE [mm]",
        "title": "Snow Water Equivalent",
    },
}


def parse_args():
    """Read paths and plot controls supplied by the SLURM script."""
    parser = argparse.ArgumentParser(
        description="Plot 2014-2016 seasonal mean HydroBlocks TRAD and SWE."
    )
    parser.add_argument(
        "--edir",
        default=DEFAULT_PATH,
        help="Experiment directory containing postprocess/output_dir",
    )
    parser.add_argument(
        "--out",
        default=os.path.join(
            OUTPUTDIR, "seasonal_climatology_trad_swe_2014_2016_4x2.png"
        ),
        help="Output PNG path",
    )
    parser.add_argument(
        "--cols",
        type=int,
        default=4,
        choices=[4],
        help="Number of seasonal columns; this figure requires 4",
    )
    parser.add_argument(
        "--buffer",
        type=int,
        default=0,
        help="Pixels removed from each outer edge before plotting",
    )
    parser.add_argument(
        "--border",
        type=float,
        default=0.02,
        help="Fractional geographic padding around valid data",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output PNG resolution",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    experiment_path = os.path.abspath(args.edir)
    output_path = os.path.abspath(args.out)

    if args.buffer < 0:
        raise ValueError("--buffer must be zero or greater")
    if args.border < 0:
        raise ValueError("--border must be zero or greater")

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    postprocess_dir = os.path.join(
        experiment_path, "postprocess", "output_dir"
    )

    print("=" * 72, flush=True)
    print("HydroBlocks seasonal climatology plotting", flush=True)
    print(f"Experiment: {experiment_path}", flush=True)
    print(f"Output: {output_path}", flush=True)
    print(f"Years: {YEARS}", flush=True)
    print("Variables: TRAD and SWE", flush=True)
    print("Seasons: Winter, Spring, Summer, and Fall", flush=True)
    print("=" * 72, flush=True)

    all_files = []
    for year in YEARS:
        year_files = sorted(
            glob.glob(os.path.join(postprocess_dir, f"{year}????.nc"))
        )
        print(f"{year}: found {len(year_files)} daily files", flush=True)
        all_files.extend(year_files)

    if not all_files:
        raise FileNotFoundError(
            f"No postprocessed daily files found in {postprocess_dir}"
        )

    seasonal_sums = {
        variable: {season: None for season in SEASONS}
        for variable in VARIABLES
    }
    seasonal_counts = {
        variable: {season: None for season in SEASONS}
        for variable in VARIABLES
    }
    seasonal_file_counts = {season: 0 for season in SEASONS}
    seasonal_record_counts = {
        variable: {season: 0 for season in SEASONS}
        for variable in VARIABLES
    }

    lon = None
    lat = None

    for file_number, filename in enumerate(all_files, start=1):
        basename = os.path.basename(filename)
        try:
            file_date = datetime.strptime(basename[:8], "%Y%m%d")
        except ValueError:
            print(f"WARNING: skipping unexpected filename: {basename}", flush=True)
            continue

        season = next(
            (name for name, months in SEASONS.items() if file_date.month in months),
            None,
        )
        if season is None:
            continue

        seasonal_file_counts[season] += 1

        if file_number % 50 == 0 or file_number == len(all_files):
            print(
                f"Processing {file_number}/{len(all_files)}: "
                f"{basename} ({season})",
                flush=True,
            )

        with nc.Dataset(filename, "r") as dataset:
            if lon is None:
                lon = np.asarray(dataset.variables["lon"][:], dtype=np.float64)
                lat = np.asarray(dataset.variables["lat"][:], dtype=np.float64)

            for variable in VARIABLES:
                if variable not in dataset.variables:
                    print(
                        f"WARNING: {variable} missing from {filename}", flush=True
                    )
                    continue

                data = np.ma.asarray(dataset.variables[variable][:]).filled(np.nan)
                data = np.asarray(data, dtype=np.float64)

                if data.ndim != 3:
                    raise ValueError(
                        f"Unexpected {variable} shape in {filename}: {data.shape}; "
                        "expected (time, lat, lon)"
                    )

                valid = (
                    np.isfinite(data)
                    & (data > -9990.0)
                    & (data < 1.0e9)
                )
                file_sum = np.sum(np.where(valid, data, 0.0), axis=0)
                file_count = np.sum(valid, axis=0)

                if seasonal_sums[variable][season] is None:
                    seasonal_sums[variable][season] = np.zeros(
                        file_sum.shape, dtype=np.float64
                    )
                    seasonal_counts[variable][season] = np.zeros(
                        file_count.shape, dtype=np.int64
                    )

                seasonal_sums[variable][season] += file_sum
                seasonal_counts[variable][season] += file_count
                seasonal_record_counts[variable][season] += data.shape[0]

    seasonal_maps = {variable: {} for variable in VARIABLES}

    for variable in VARIABLES:
        for season in SEASONS:
            total = seasonal_sums[variable][season]
            count = seasonal_counts[variable][season]
            if total is None or count is None:
                print(f"WARNING: no data for {variable} {season}", flush=True)
                continue

            mean_map = np.full(total.shape, np.nan, dtype=np.float64)
            np.divide(total, count, out=mean_map, where=count > 0)
            seasonal_maps[variable][season] = mean_map

            print(
                f"{variable.upper()} {season}: "
                f"files={seasonal_file_counts[season]}, "
                f"records={seasonal_record_counts[variable][season]}, "
                f"min={np.nanmin(mean_map):.6g}, "
                f"max={np.nanmax(mean_map):.6g}, "
                f"spatial mean={np.nanmean(mean_map):.6g}",
                flush=True,
            )

    if lon is None or lat is None:
        raise RuntimeError("Longitude and latitude were not read")

    color_limits = {}
    for variable in VARIABLES:
        pieces = []
        for season in SEASONS:
            if season in seasonal_maps[variable]:
                values = seasonal_maps[variable][season]
                values = values[np.isfinite(values)]
                if values.size:
                    pieces.append(values)

        if not pieces:
            continue

        combined = np.concatenate(pieces)
        if variable == "swe":
            vmax = float(np.percentile(combined, 99))
            if vmax <= 0.0:
                vmax = 1.0
            color_limits[variable] = (0.0, vmax)
        else:
            vmin = float(np.percentile(combined, 1))
            vmax = float(np.percentile(combined, 99))
            if np.isclose(vmin, vmax):
                adjustment = max(abs(vmin) * 0.01, 1.0e-6)
                vmin -= adjustment
                vmax += adjustment
            color_limits[variable] = (vmin, vmax)

    # Optionally remove a fixed number of pixels from all four edges.
    if args.buffer > 0:
        buffer = args.buffer
        sample_map = next(
            array
            for variable_maps in seasonal_maps.values()
            for array in variable_maps.values()
        )
        if 2 * buffer >= min(sample_map.shape):
            raise ValueError(
                f"--buffer {buffer} is too large for map shape {sample_map.shape}"
            )
        lon = lon[buffer:-buffer]
        lat = lat[buffer:-buffer]
        for variable in VARIABLES:
            for season in list(seasonal_maps[variable]):
                seasonal_maps[variable][season] = (
                    seasonal_maps[variable][season][buffer:-buffer, buffer:-buffer]
                )

    extent = [
        float(np.nanmin(lon)),
        float(np.nanmax(lon)),
        float(np.nanmin(lat)),
        float(np.nanmax(lat)),
    ]

    # Find the valid-data footprint and add a small geographic border.
    valid_domain = None
    for variable in VARIABLES:
        for array in seasonal_maps[variable].values():
            finite = np.isfinite(array)
            valid_domain = finite if valid_domain is None else (valid_domain | finite)

    if valid_domain is None or not np.any(valid_domain):
        raise RuntimeError("No valid seasonal pixels are available for plotting")

    valid_rows, valid_columns = np.where(valid_domain)
    valid_x = lon[valid_columns]
    valid_y = lat[valid_rows]
    xmin, xmax = float(np.nanmin(valid_x)), float(np.nanmax(valid_x))
    ymin, ymax = float(np.nanmin(valid_y)), float(np.nanmax(valid_y))
    pad_x = args.border * max(xmax - xmin, np.finfo(float).eps)
    pad_y = args.border * max(ymax - ymin, np.finfo(float).eps)
    plot_xlim = (xmin - pad_x, xmax + pad_x)
    plot_ylim = (ymin - pad_y, ymax + pad_y)

    fig, axes = plt.subplots(
        nrows=2,
        ncols=args.cols,
        figsize=(20, 9),
        sharex=False,
        sharey=False,
    )

    # Leave a narrow strip at the right for one full-height colorbar per row.
    # The slightly larger hspace leaves room for longitude labels on both rows.
    fig.subplots_adjust(
        left=0.045,
        right=0.94,
        bottom=0.07,
        top=0.97,
        wspace=0.10,
        hspace=0.18,
    )

    for variable in VARIABLES:
        settings = PLOT_SETTINGS[variable]
        row = settings["row"]
        row_images = []

        for column, season in enumerate(SEASONS):
            axis = axes[row, column]
            if season not in seasonal_maps[variable]:
                axis.set_title(f"{season}\nNo data")
                axis.axis("off")
                continue

            vmin, vmax = color_limits[variable]
            image = axis.imshow(
                np.ma.masked_invalid(seasonal_maps[variable][season]),
                origin="lower",
                extent=extent,
                cmap=settings["cmap"],
                interpolation="nearest",
                resample=False,
                aspect="equal",
                vmin=vmin,
                vmax=vmax,
            )
            row_images.append(image)
            axis.set_title(
                f"{settings['title']} – {season}", fontsize=12, fontweight="bold", pad=5
            )
            # Display both geographic axes on every seasonal panel.
            axis.set_xlabel("Longitude [°]", fontsize=10)
            axis.set_ylabel("Latitude [°]", fontsize=10)
            axis.tick_params(
                axis="both",
                labelsize=9,
                labelbottom=True,
                labelleft=True,
            )
            axis.set_xlim(plot_xlim)
            axis.set_ylim(plot_ylim)

        if row_images:
            # Give each row a dedicated colorbar axis immediately outside its
            # Fall panel. Its y-position and height exactly match that map.
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
                row_images[0],
                cax=colorbar_axis,
                orientation="vertical",
            )
            colorbar.set_label(settings["label"], fontsize=11)
            colorbar.ax.tick_params(labelsize=9)

    # No overall figure title, as requested.
    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Saved: {output_path}", flush=True)


if __name__ == "__main__":
    main()
