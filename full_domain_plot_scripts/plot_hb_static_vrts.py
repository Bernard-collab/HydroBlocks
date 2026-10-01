#!/usr/bin/env python3
"""Plot HydroBlocks static variables directly from postprocess VRT mosaics.

This is the VRT-based alternative to the HRU-painting plotting script. It
does not read ``<cid>/input_file.nc`` and does not use ``hrus.vrt`` or
``cids.vrt`` to reconstruct parameter values. Instead, each variable is read
directly from:

    <edir>/postprocess/<variable>.vrt

Expected default VRTs
---------------------
    dem.vrt       Elevation
    slope.vrt     Terrain slope
    y_aspect.vrt  North-south aspect component
    x_aspect.vrt  East-west aspect component
    svf.vrt       Sky-view factor
    tvf.vrt       Terrain-view factor
    hor_n.vrt     Northern horizon angle
    hor_w.vrt     Western horizon angle

Unit handling
-------------
``slope``, ``hor_n``, and ``hor_w`` are converted from radians to degrees.
The signed ``x_aspect`` and ``y_aspect`` components are dimensionless and are
not converted.

Example
-------
python plot_hb_static_vrts.py \
    --edir /scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_3yr_2hru_full_domain_no_downscale \
    --out /scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/full_domain_output_plot/static_parameters_from_vrts.png \
    --cols 4 \
    --buffer 0 \
    --border 0.02 \
    --dpi 400

``--border`` adds plotting space around valid pixels without changing the
raster. ``--buffer`` physically removes pixels from all four raster edges and
should normally remain zero.
"""

import argparse
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


DEFAULT_VARS = [
    "dem", "slope", "y_aspect", "x_aspect",
    "svf", "tvf", "hor_n", "hor_w",
]

RADIAN_VARIABLES = {"slope", "hor_n", "hor_w"}

VARIABLE_TITLES = {
    "dem": "Elevation",
    "slope": "Terrain Slope",
    "y_aspect": "North–South Aspect Component",
    "x_aspect": "East–West Aspect Component",
    "svf": "Sky-View Factor",
    "tvf": "Terrain-View Factor",
    "hor_n": "Northern Horizon Angle",
    "hor_w": "Western Horizon Angle",
}

COLORBAR_LABELS = {
    "dem": "Elevation [m]",
    "slope": "Slope [°]",
    "y_aspect": "North–south aspect component [−]",
    "x_aspect": "East–west aspect component [−]",
    "svf": "Sky-view factor [−]",
    "tvf": "Terrain-view factor [−]",
    "hor_n": "Northern horizon angle [°]",
    "hor_w": "Western horizon angle [°]",
}

# Keep the same palette used by the current HRU-based static plot.
VARIABLE_CMAPS = {variable: "terrain" for variable in DEFAULT_VARS}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot HydroBlocks static variables directly from VRTs."
    )
    parser.add_argument(
        "--edir",
        required=True,
        help="Experiment directory containing postprocess/<variable>.vrt",
    )
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument(
        "--vars", nargs="+", default=DEFAULT_VARS,
        help="Static VRT variables to plot",
    )
    parser.add_argument(
        "--buffer", type=int, default=0,
        help="Pixels removed from each outer raster edge; normally 0",
    )
    parser.add_argument(
        "--border", type=float, default=0.02,
        help="Fractional map border around valid pixels; default 0.02",
    )
    parser.add_argument("--cols", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument(
        "--save-arrays", action="store_true",
        help="Save every plotted array as a compressed NPZ file",
    )
    return parser.parse_args()


def read_vrt(vrt_path, buffer_pixels):
    """Read one VRT, mask invalid data, and return its spatial metadata."""
    if buffer_pixels < 0:
        raise ValueError("--buffer must be zero or greater")

    with rasterio.open(vrt_path) as source:
        if 2 * buffer_pixels >= min(source.width, source.height):
            raise ValueError(
                f"--buffer {buffer_pixels} is too large for {vrt_path}"
            )

        window = Window(
            buffer_pixels,
            buffer_pixels,
            source.width - 2 * buffer_pixels,
            source.height - 2 * buffer_pixels,
        )
        masked = source.read(1, window=window, masked=True)
        transform = window_transform(window, source.transform)
        bounds = rasterio.transform.array_bounds(
            masked.shape[0], masked.shape[1], transform
        )
        crs = source.crs

    array = np.ma.filled(masked, np.nan).astype(np.float64, copy=False)
    array[~np.isfinite(array)] = np.nan
    return array, transform, bounds, crs


def convert_units(variable, array):
    """Convert angular fields stored in radians to degrees."""
    if variable in RADIAN_VARIABLES:
        return np.degrees(array)
    return array


def get_plot_options(variable, array):
    """Return colormap and limits for one variable."""
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        raise ValueError(f"{variable}.vrt contains no finite values")

    minimum = float(np.min(finite))
    maximum = float(np.max(finite))
    cmap = VARIABLE_CMAPS.get(variable, "terrain")
    norm = None
    vmin = minimum
    vmax = maximum

    if variable in {"x_aspect", "y_aspect"}:
        limit = max(abs(minimum), abs(maximum), np.finfo(float).eps)
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        vmin = None
        vmax = None
    elif variable in {"svf", "tvf"}:
        vmin, vmax = 0.0, 1.0
    elif np.isclose(vmin, vmax):
        adjustment = max(abs(vmin) * 0.01, 1.0e-6)
        vmin -= adjustment
        vmax += adjustment

    return cmap, norm, vmin, vmax


def valid_plot_limits(array, bounds, border_fraction):
    """Calculate valid-pixel bounds with a small geographic border."""
    if border_fraction < 0:
        raise ValueError("--border must be zero or greater")

    rows, columns = np.where(np.isfinite(array))
    if rows.size == 0:
        raise ValueError("Raster contains no valid pixels")

    left, bottom, right, top = bounds
    height, width = array.shape
    pixel_width = (right - left) / float(width)
    pixel_height = (top - bottom) / float(height)

    valid_left = left + columns.min() * pixel_width
    valid_right = left + (columns.max() + 1) * pixel_width
    valid_top = top - rows.min() * pixel_height
    valid_bottom = top - (rows.max() + 1) * pixel_height

    pad_x = border_fraction * (valid_right - valid_left)
    pad_y = border_fraction * (valid_top - valid_bottom)
    return (
        (valid_left - pad_x, valid_right + pad_x),
        (valid_bottom - pad_y, valid_top + pad_y),
    )


def main():
    args = parse_args()
    experiment = os.path.abspath(args.edir)
    postprocess = os.path.join(experiment, "postprocess")
    output_path = os.path.abspath(args.out)

    if not os.path.isdir(postprocess):
        sys.exit(f"Not found: {postprocess}")

    output_parent = os.path.dirname(output_path)
    if output_parent:
        os.makedirs(output_parent, exist_ok=True)

    variables = args.vars
    ncols = max(1, min(args.cols, len(variables)))
    nrows = int(math.ceil(len(variables) / float(ncols)))

    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(4.7 * ncols, 4.1 * nrows),
        squeeze=False,
        sharex=False,
        sharey=False,
    )
    axes = axes.ravel()

    # Space accommodates labels on every panel and each adjacent colorbar.
    fig.subplots_adjust(
        left=0.045,
        right=0.985,
        bottom=0.065,
        top=0.97,
        wspace=0.16,
        hspace=0.18,
    )

    reference_shape = None
    reference_transform = None
    reference_crs = None
    output_base = os.path.splitext(output_path)[0]

    for index, variable in enumerate(variables):
        vrt_path = os.path.join(postprocess, f"{variable}.vrt")
        if not os.path.isfile(vrt_path):
            plt.close(fig)
            sys.exit(f"Not found: {vrt_path}")

        print(f"Reading {variable}: {vrt_path}", flush=True)
        try:
            array, transform, bounds, crs = read_vrt(vrt_path, args.buffer)
            array = convert_units(variable, array)
            plot_xlim, plot_ylim = valid_plot_limits(
                array, bounds, args.border
            )
            cmap, norm, vmin, vmax = get_plot_options(variable, array)
        except (OSError, ValueError) as error:
            plt.close(fig)
            sys.exit(str(error))

        if reference_shape is None:
            reference_shape = array.shape
            reference_transform = transform
            reference_crs = crs
        elif (
            array.shape != reference_shape
            or not np.allclose(tuple(transform), tuple(reference_transform))
            or crs != reference_crs
        ):
            plt.close(fig)
            sys.exit(
                f"{variable}.vrt does not match the first VRT grid: "
                f"shape={array.shape}, CRS={crs}"
            )

        left, bottom, right, top = bounds
        extent = (left, right, bottom, top)
        axis = axes[index]
        image = axis.imshow(
            np.ma.masked_invalid(array),
            extent=extent,
            origin="upper",
            cmap=cmap,
            norm=norm,
            vmin=vmin if norm is None else None,
            vmax=vmax if norm is None else None,
            interpolation="nearest",
            resample=False,
            aspect="equal",
        )
        axis.set_xlim(plot_xlim)
        axis.set_ylim(plot_ylim)
        axis.set_title(
            VARIABLE_TITLES.get(variable, variable),
            fontsize=11,
            fontweight="bold",
            pad=4,
        )
        axis.set_xlabel("Longitude [°]", fontsize=9)
        axis.set_ylabel("Latitude [°]", fontsize=9)
        axis.tick_params(
            axis="both",
            labelsize=8,
            labelbottom=True,
            labelleft=True,
        )

        # The appended colorbar matches the map height exactly.
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="4%", pad=0.08)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(
            COLORBAR_LABELS.get(variable, variable), fontsize=9
        )
        colorbar.ax.tick_params(labelsize=8)

        finite = array[np.isfinite(array)]
        print(
            f"  shape={array.shape}; range={finite.min():.6g} "
            f"to {finite.max():.6g}",
            flush=True,
        )

        if args.save_arrays:
            np.savez_compressed(
                f"{output_base}_{variable}.npz",
                data=array,
                extent=np.asarray(extent),
                plot_xlim=np.asarray(plot_xlim),
                plot_ylim=np.asarray(plot_ylim),
                units=COLORBAR_LABELS.get(variable, variable),
            )

    for axis in axes[len(variables):]:
        axis.axis("off")

    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()
