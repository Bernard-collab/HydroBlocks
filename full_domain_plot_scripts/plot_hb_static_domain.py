#!/usr/bin/env python3
"""Plot HydroBlocks HRU-representative static parameters.

Static parameter values are read from the ``parameters`` group in:

    <edir>/<cid>/input_file.nc

The values are painted back onto their native spatial locations using:

    <edir>/postprocess/hrus.vrt
    <edir>/postprocess/cids.vrt

Every raster pixel belonging to an HRU receives that HRU's representative
parameter value.

Default variables
-----------------
    dem       Elevation
    slope     Terrain slope
    y_aspect  North-south aspect component
    x_aspect  East-west aspect component
    svf       Sky-view factor
    tvf       Terrain-view factor
    hor_n     Northern horizon angle
    hor_w     Western horizon angle

Unit handling
-------------
The following variables are converted from radians to degrees:

    slope, hor_n, hor_w

The ``x_aspect`` and ``y_aspect`` variables are signed, dimensionless
directional components and are not converted.

Styling
-------
* One font for all text: DejaVu Sans. Only the panel titles are bold.
* 2 columns by default, with the column gap reduced. Latitude labels and
  ticks are shown only on the left column (the y-axis is shared), which
  removes the repeated labels that widened the gap between columns.

Domain border
-------------
The script determines the spatial extent containing valid HRU and CID pixels.
It then adds a small border around that extent. The default border is 2% of
the valid domain width and height.

This border does not crop or resample the raster. Use ``--border 0`` to remove
the border completely.

The ``--buffer`` option is different: it physically removes raster pixels
from all four outer edges before mapping. Normally use ``--buffer 0``.

Example
-------
python plot_hb_static_domain_grided.py \
    --edir /scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_1yr_100hru_100bh_bug_checking_new_ben_prep \
    --out /scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/output_plots/static_HRU_parameters_full_domain.png \
    --cols 2 \
    --buffer 0 \
    --border 0.02 \
    --dpi 300

Optional examples
-----------------
Plot only CIDs 1-4:

    --cids 1,2,3,4

Select a layer from a layered HRU parameter:

    --layer 0

Save each reconstructed map as a compressed NPZ file:

    --save-arrays
"""

import argparse
import math
import os
import sys

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


# ---------------------------------------------------------------------
# Fonts: the same font (DejaVu Sans) for every piece of text.
# Only the panel titles are bold; everything else is regular weight.
# ---------------------------------------------------------------------
FONT_TITLE = 18      # panel titles (bold)
FONT_LABEL = 15      # axis and colorbar labels
FONT_TICK = 13       # tick labels (map and colorbar)

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "mathtext.fontset": "dejavusans",   # keeps any math/symbol text in the same font
    "font.size": FONT_LABEL,
    "font.weight": "normal",
    "axes.titlesize": FONT_TITLE,
    "axes.titleweight": "bold",
    "axes.labelsize": FONT_LABEL,
    "axes.labelweight": "normal",
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "figure.titlesize": 20,
})


DEFAULT_VARS = [
    "dem",
    "slope",
    "y_aspect",
    "x_aspect",
    "svf",
    "tvf",
    "hor_n",
    "hor_w",
]


# Variables stored in radians in HydroBlocks
RADIAN_VARIABLES = {
    "slope",
    "hor_n",
    "hor_w",
}


VARIABLE_TITLES = {
    "dem": "Elevation",
    "slope": "Terrain Slope",
    "y_aspect": "y_aspect",
    "x_aspect": "x_aspect",
    "svf": "Sky-View Factor",
    "tvf": "Terrain-View Factor",
    "hor_n": "hor_n",
    "hor_w": "hor_w",
}


COLORBAR_LABELS = {
    "dem": "Elevation [m]",
    "slope": "Slope [°]",
    "y_aspect": "y_aspect [−]",
    "x_aspect": "x_aspect [−]",
    "svf": "Sky-view factor [−]",
    "tvf": "Terrain-view factor [−]",
    "hor_n": "hor_n [°]",
    "hor_w": "hor_w [°]",
}


VARIABLE_CMAPS = {
    "dem": "terrain",
    "slope": "terrain",
    "y_aspect": "terrain",
    "x_aspect": "terrain",
    "svf": "terrain",
    "tvf": "terrain",
    "hor_n": "terrain",
    "hor_w": "terrain",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paint HydroBlocks HRU static parameters onto the model grid "
            "and produce a multi-panel figure."
        )
    )

    parser.add_argument(
        "--edir",
        required=True,
        help=(
            "Simulation directory containing <cid>/input_file.nc "
            "and postprocess/"
        ),
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output figure path, normally a PNG file",
    )

    parser.add_argument(
        "--vars",
        nargs="+",
        default=DEFAULT_VARS,
        help="Parameter variables to plot",
    )

    parser.add_argument(
        "--cids",
        default=None,
        help=(
            "Comma-separated CID subset. By default, all numeric CID "
            "directories are used."
        ),
    )

    parser.add_argument(
        "--buffer",
        type=int,
        default=0,
        help=(
            "Number of raster pixels physically removed from every outer "
            "edge before mapping. Normally use 0."
        ),
    )

    parser.add_argument(
        "--border",
        type=float,
        default=0.02,
        help=(
            "Fractional plotting border around the valid domain. "
            "Default: 0.02, meaning 2 percent."
        ),
    )

    parser.add_argument(
        "--cols",
        type=int,
        default=2,
        help="Number of subplot columns (default 2)",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
        help="Output figure resolution",
    )

    parser.add_argument(
        "--layer",
        type=int,
        default=None,
        help="Layer index for a two-dimensional HRU parameter",
    )

    parser.add_argument(
        "--save-arrays",
        action="store_true",
        help="Save each reconstructed map as a compressed NPZ file",
    )

    return parser.parse_args()


def available_cids(edir, requested):
    """Return the requested CIDs or all numeric CID directories."""

    if requested:
        cids = [
            int(value.strip())
            for value in requested.split(",")
            if value.strip()
        ]
    else:
        cids = sorted(
            int(name)
            for name in os.listdir(edir)
            if name.isdigit()
            and os.path.isdir(os.path.join(edir, name))
        )

    if not cids:
        raise RuntimeError(
            f"No numeric CID directories found under {edir}"
        )

    return cids


def read_domain_raster(path, buffer_pixels):
    """Read a domain raster, optionally removing outer pixels."""

    if buffer_pixels < 0:
        raise ValueError("--buffer must be zero or greater")

    with rasterio.open(path) as source:
        if 2 * buffer_pixels >= min(source.width, source.height):
            raise ValueError(
                f"--buffer {buffer_pixels} is too large for {path}"
            )

        window = Window(
            buffer_pixels,
            buffer_pixels,
            source.width - 2 * buffer_pixels,
            source.height - 2 * buffer_pixels,
        )

        array = source.read(
            1,
            window=window,
            masked=False,
        )

        transform = window_transform(
            window,
            source.transform,
        )

        bounds = rasterio.transform.array_bounds(
            array.shape[0],
            array.shape[1],
            transform,
        )

        nodata = source.nodata

    return array, nodata, bounds


def read_parameter(ncpath, variable, layer):
    """Read one HRU parameter from input_file.nc."""

    with nc.Dataset(ncpath, "r") as dataset:
        if "parameters" not in dataset.groups:
            raise KeyError("missing group 'parameters'")

        group = dataset.groups["parameters"]

        if variable not in group.variables:
            raise KeyError(
                f"missing parameter '{variable}'"
            )

        values = np.ma.asarray(
            group.variables[variable][:]
        )

    values = np.ma.filled(
        values,
        np.nan,
    )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = np.squeeze(values)

    if values.ndim == 1:
        return values.astype(
            np.float32,
            copy=False,
        )

    if values.ndim == 2 and layer is not None:
        if layer < 0 or layer >= values.shape[1]:
            raise IndexError(
                f"layer {layer} is outside parameter shape {values.shape}"
            )

        return values[:, layer].astype(
            np.float32,
            copy=False,
        )

    raise ValueError(
        f"{variable} has shape {values.shape}; "
        "use --layer for a layered parameter"
    )


def paint_variable(
    edir,
    variable,
    cids_to_use,
    hrus,
    cids_map,
    hru_nodata,
    layer,
):
    """Paint HRU parameter values back onto the spatial raster."""

    result = np.full(
        hrus.shape,
        np.nan,
        dtype=np.float32,
    )

    number_mapped = 0
    skipped = []

    for cid in cids_to_use:
        ncpath = os.path.join(
            edir,
            str(cid),
            "input_file.nc",
        )

        if not os.path.isfile(ncpath):
            skipped.append(
                (cid, "missing input_file.nc")
            )
            continue

        try:
            values = read_parameter(
                ncpath,
                variable,
                layer,
            )
        except (
            KeyError,
            ValueError,
            IndexError,
            OSError,
        ) as error:
            skipped.append(
                (cid, str(error))
            )
            continue

        cid_mask = cids_map == cid

        if not np.any(cid_mask):
            skipped.append(
                (cid, "CID is absent from cids.vrt")
            )
            continue

        raw_hru_ids = hrus[cid_mask]

        valid = np.isfinite(raw_hru_ids)

        if hru_nodata is not None:
            valid &= raw_hru_ids != hru_nodata

        hru_ids = np.zeros(
            raw_hru_ids.shape,
            dtype=np.int64,
        )

        hru_ids[valid] = raw_hru_ids[valid].astype(
            np.int64
        )

        valid &= hru_ids >= 0
        valid &= hru_ids < values.size

        if np.any(valid):
            flat_positions = np.flatnonzero(
                cid_mask
            )

            result.flat[
                flat_positions[valid]
            ] = values[hru_ids[valid]]

            number_mapped += 1
        else:
            skipped.append(
                (cid, "no valid HRU indices")
            )

    return result, number_mapped, skipped


def convert_units(variable, mapped):
    """Convert applicable angular parameters to degrees."""

    mapped = mapped.astype(
        np.float64,
        copy=False,
    )

    if variable in RADIAN_VARIABLES:
        mapped = np.degrees(mapped)

    return mapped


def get_plot_options(variable, mapped):
    """Return the colormap and color normalization."""

    finite_values = mapped[np.isfinite(mapped)]

    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    cmap = VARIABLE_CMAPS.get(
        variable,
        "viridis",
    )

    norm = None
    vmin = minimum
    vmax = maximum

    # Aspect components are signed and should be centered on zero.
    if variable in {"x_aspect", "y_aspect"}:
        limit = max(
            abs(minimum),
            abs(maximum),
        )

        if limit == 0:
            limit = 1.0

        norm = TwoSlopeNorm(
            vmin=-limit,
            vcenter=0.0,
            vmax=limit,
        )

        vmin = None
        vmax = None

    # SVF and TVF are dimensionless fractions.
    elif variable in {"svf", "tvf"}:
        vmin = 0.0
        vmax = 1.0

    # Avoid identical limits for constant fields.
    elif np.isclose(vmin, vmax):
        difference = max(
            abs(vmin) * 0.01,
            1.0e-6,
        )

        vmin -= difference
        vmax += difference

    return cmap, norm, vmin, vmax


def valid_domain_limits(
    hrus,
    cids_map,
    hru_nodata,
    bounds,
    border_fraction,
):
    """Calculate the valid-data bounds plus a small plotting border."""

    if border_fraction < 0:
        raise ValueError(
            "--border must be zero or greater"
        )

    valid = (
        np.isfinite(hrus)
        & np.isfinite(cids_map)
        & (cids_map > 0)
    )

    if hru_nodata is not None:
        valid &= hrus != hru_nodata

    rows, columns = np.where(valid)

    if rows.size == 0:
        raise RuntimeError(
            "No valid HRU/CID pixels were found in the domain rasters"
        )

    left, bottom, right, top = bounds

    raster_height, raster_width = hrus.shape

    pixel_width = (
        right - left
    ) / float(raster_width)

    pixel_height = (
        top - bottom
    ) / float(raster_height)

    valid_left = (
        left
        + columns.min() * pixel_width
    )

    valid_right = (
        left
        + (columns.max() + 1) * pixel_width
    )

    # Raster rows begin at the northern/top edge.
    valid_top = (
        top
        - rows.min() * pixel_height
    )

    valid_bottom = (
        top
        - (rows.max() + 1) * pixel_height
    )

    domain_width = valid_right - valid_left
    domain_height = valid_top - valid_bottom

    pad_x = border_fraction * domain_width
    pad_y = border_fraction * domain_height

    plot_xlim = (
        valid_left - pad_x,
        valid_right + pad_x,
    )

    plot_ylim = (
        valid_bottom - pad_y,
        valid_top + pad_y,
    )

    return plot_xlim, plot_ylim


def main():
    args = parse_args()

    edir = os.path.abspath(
        args.edir
    )

    output_path = os.path.abspath(
        args.out
    )

    hrupath = os.path.join(
        edir,
        "postprocess",
        "hrus.vrt",
    )

    cidpath = os.path.join(
        edir,
        "postprocess",
        "cids.vrt",
    )

    for required_path in (
        edir,
        hrupath,
        cidpath,
    ):
        if not os.path.exists(required_path):
            sys.exit(
                f"Not found: {required_path}"
            )

    try:
        cids_to_use = available_cids(
            edir,
            args.cids,
        )

        hrus, hru_nodata, bounds = read_domain_raster(
            hrupath,
            args.buffer,
        )

        cids_map, _, cid_bounds = read_domain_raster(
            cidpath,
            args.buffer,
        )

    except (
        RuntimeError,
        ValueError,
        OSError,
    ) as error:
        sys.exit(str(error))

    if (
        hrus.shape != cids_map.shape
        or not np.allclose(bounds, cid_bounds)
    ):
        sys.exit(
            "hrus.vrt and cids.vrt do not have the same cropped grid"
        )

    try:
        plot_xlim, plot_ylim = valid_domain_limits(
            hrus,
            cids_map,
            hru_nodata,
            bounds,
            args.border,
        )
    except (
        RuntimeError,
        ValueError,
    ) as error:
        sys.exit(str(error))

    print(
        "HRU raster shape:",
        hrus.shape,
        flush=True,
    )

    print(
        "Number of CIDs:",
        len(cids_to_use),
        flush=True,
    )

    print(
        "Plot longitude limits:",
        plot_xlim,
        flush=True,
    )

    print(
        "Plot latitude limits:",
        plot_ylim,
        flush=True,
    )

    ncols = max(
        1,
        min(args.cols, len(args.vars)),
    )

    nrows = int(
        math.ceil(
            len(args.vars) / float(ncols)
        )
    )

    # Narrower columns (6.4 in instead of 7.8 in) so the square maps fill
    # their columns and the gap between columns shrinks.
    fig, axes = plt.subplots(
        nrows,
        ncols,
        figsize=(
            6.8 * ncols,
            6.3 * nrows,
        ),
        squeeze=False,
        sharex=True,
        sharey=True,
    )

    axes = axes.ravel()

    left, bottom, right, top = bounds

    extent = (
        left,
        right,
        bottom,
        top,
    )

    output_directory = os.path.dirname(
        output_path
    )

    if output_directory:
        os.makedirs(
            output_directory,
            exist_ok=True,
        )

    output_base = os.path.splitext(
        output_path
    )[0]

    for index, variable in enumerate(args.vars):
        print(
            f"Mapping {variable} ...",
            flush=True,
        )

        mapped, number_mapped, skipped = paint_variable(
            edir,
            variable,
            cids_to_use,
            hrus,
            cids_map,
            hru_nodata,
            args.layer,
        )

        mapped = convert_units(
            variable,
            mapped,
        )

        finite = np.isfinite(mapped)

        if not np.any(finite):
            plt.close(fig)

            reasons = "; ".join(
                f"CID {cid}: {reason}"
                for cid, reason in skipped[:5]
            )

            sys.exit(
                f"No values mapped for {variable}. {reasons}"
            )

        axis = axes[index]
        column = index % ncols
        is_left_column = column == 0

        cmap, norm, vmin, vmax = get_plot_options(
            variable,
            mapped,
        )

        image = axis.imshow(
            np.ma.masked_invalid(mapped),
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

        # Show only the valid domain plus the small requested border.
        axis.set_xlim(plot_xlim)
        axis.set_ylim(plot_ylim)

        # Bold titles (the only bold text in the figure)
        axis.set_title(
            VARIABLE_TITLES.get(
                variable,
                variable,
            ),
            fontsize=FONT_TITLE,
            fontweight="bold",
            pad=8,
        )

        axis.set_xlabel(
            "Longitude [°]",
            fontsize=FONT_LABEL,
            fontweight="normal",
        )

        # Latitude label only on the left column (shared y-axis)
        axis.set_ylabel(
            "Latitude [°]" if is_left_column else "",
            fontsize=FONT_LABEL,
            fontweight="normal",
        )

        axis.tick_params(
            axis="both",
            labelsize=FONT_TICK,
            labelbottom=True,
            labelleft=is_left_column,
        )

        # This creates a colorbar with exactly the same height as the map.
        divider = make_axes_locatable(
            axis
        )

        colorbar_axis = divider.append_axes(
            "right",
            size="4%",
            pad=0.08,
        )

        colorbar = fig.colorbar(
            image,
            cax=colorbar_axis,
        )

        colorbar.set_label(
            COLORBAR_LABELS.get(
                variable,
                variable,
            ),
            fontsize=FONT_LABEL,
            fontweight="normal",
        )

        colorbar.ax.tick_params(
            labelsize=FONT_TICK,
        )

        print(
            (
                f"  CIDs mapped: {number_mapped}/{len(cids_to_use)}; "
                f"range: {np.nanmin(mapped):.6g} to "
                f"{np.nanmax(mapped):.6g}; "
                f"skipped: {len(skipped)}"
            ),
            flush=True,
        )

        for cid, reason in skipped[:5]:
            print(
                f"    CID {cid}: {reason}",
                flush=True,
            )

        if args.save_arrays:
            np.savez_compressed(
                f"{output_base}_{variable}.npz",
                data=mapped,
                extent=np.asarray(extent),
                plot_xlim=np.asarray(plot_xlim),
                plot_ylim=np.asarray(plot_ylim),
                units=COLORBAR_LABELS.get(
                    variable,
                    variable,
                ),
            )

    for axis in axes[len(args.vars):]:
        axis.axis("off")

    # Tighter spacing: smaller gap between the two columns.
    fig.subplots_adjust(
        left=0.07,
        right=0.95,
        bottom=0.04,
        top=0.97,
        wspace=0.26,   # more room so left colorbar labels clear the right maps
        hspace=0.16,   # smaller gap between rows (was 0.22)
    )

    fig.savefig(
        output_path,
        dpi=args.dpi,
        bbox_inches="tight",
        facecolor="white",
    )

    plt.close(fig)

    print(
        f"Wrote {output_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()