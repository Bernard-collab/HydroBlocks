#!/usr/bin/env python3
"""Plot HydroBlocks HRU-representative static parameters (publication layout).

Static parameter values are read from the ``parameters`` group in:

    <edir>/<cid>/input_file.nc

The values are painted back onto their native spatial locations using:

    <edir>/postprocess/hrus.vrt
    <edir>/postprocess/cids.vrt

Every raster pixel belonging to an HRU receives that HRU's representative
parameter value.

Layout and styling
------------------
* 2 columns x 4 rows (for the 8 default variables), sized for a full journal
  page (default 6.5 x 9 in). Insert the figure at this width so text prints at
  its intended size.
* Publication text sizes: 9 pt titles, 8 pt axis/colorbar labels, 7 pt ticks.
* Panel letters (a)-(h) in the top-left corner of each map.
* Axis labels only on the outer panels (left column / bottom row).
* State boundaries are drawn when --boundaries is given; no gridlines.

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
slope, hor_n and hor_w are converted from radians to degrees. The aspect
components are signed, dimensionless and are not converted.

Domain border
-------------
The plot extent is the valid HRU/CID extent plus a small border (default 2%).
Use ``--border 0`` to remove it. ``--buffer`` physically removes raster pixels
from the outer edges; normally use ``--buffer 0``.

Boundaries
----------
``--boundaries`` accepts a local shapefile/zip path or a URL. A URL only works
if the machine running the script has internet access -- HPC compute nodes
often do not. In that case download the file once on a login node and pass the
local path. Boundaries are reprojected to the raster CRS automatically.

Example
-------
python plot_hb_static_domain.py \\
    --edir /scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_1yr_100hru_100bh_bug_checking_new_ben_prep \\
    --out /scratch/alpine/battobrah@xsede.org/HydroBlocks_Enrico_dev/output_plots/static_HRU_parameters_full_domain.png \\
    --boundaries /path/to/ne_50m_admin_1_states_provinces.zip \\
    --dpi 600

Optional
--------
    --cids 1,2,3,4          plot only some CIDs
    --layer 0               select a layer of a layered parameter
    --save-arrays           save each reconstructed map as NPZ
    --figsize 7.0 9.0       change the figure size (inches)
    --no-letters            omit the (a)-(h) panel letters
"""

import argparse
import math
import os
import string
import sys

import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import rasterio

from matplotlib.colors import TwoSlopeNorm
from mpl_toolkits.axes_grid1 import make_axes_locatable
from rasterio.windows import Window
from rasterio.windows import transform as window_transform


# ---------------------------------------------------------------------------
# Publication text styling (sizes in points, at the printed figure size)
# ---------------------------------------------------------------------------
FONT_TITLE = 9
FONT_LABEL = 8
FONT_TICK = 7
FONT_LETTER = 9

mpl.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": FONT_LABEL,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "xtick.major.size": 2.5,
    "ytick.major.size": 2.5,
    "pdf.fonttype": 42,   # editable text in PDF/EPS output
    "ps.fonttype": 42,
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
RADIAN_VARIABLES = {"slope", "hor_n", "hor_w"}

VARIABLE_TITLES = {
    "dem": "Elevation",
    "slope": "Terrain slope",
    "y_aspect": "North–south aspect component",
    "x_aspect": "East–west aspect component",
    "svf": "Sky-view factor",
    "tvf": "Terrain-view factor",
    "hor_n": "Northern horizon angle",
    "hor_w": "Western horizon angle",
}

COLORBAR_LABELS = {
    "dem": "Elevation (m)",
    "slope": "Slope (°)",
    "y_aspect": "N–S component (−)",
    "x_aspect": "E–W component (−)",
    "svf": "SVF (−)",
    "tvf": "TVF (−)",
    "hor_n": "Horizon angle (°)",
    "hor_w": "Horizon angle (°)",
}

VARIABLE_CMAPS = {v: "terrain" for v in DEFAULT_VARS}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Paint HydroBlocks HRU static parameters onto the model grid "
            "and produce a publication-ready multi-panel figure."
        )
    )
    parser.add_argument("--edir", required=True,
                        help="Simulation directory containing <cid>/input_file.nc and postprocess/")
    parser.add_argument("--out", required=True,
                        help="Output figure path (PNG, PDF, ...)")
    parser.add_argument("--vars", nargs="+", default=DEFAULT_VARS,
                        help="Parameter variables to plot")
    parser.add_argument("--cids", default=None,
                        help="Comma-separated CID subset (default: all numeric CID directories)")
    parser.add_argument("--buffer", type=int, default=0,
                        help="Raster pixels removed from every outer edge before mapping (normally 0)")
    parser.add_argument("--border", type=float, default=0.02,
                        help="Fractional plotting border around the valid domain (default 0.02)")
    parser.add_argument("--cols", type=int, default=2,
                        help="Number of subplot columns (default 2)")
    parser.add_argument("--figsize", type=float, nargs=2, default=(6.5, 9.0),
                        metavar=("WIDTH", "HEIGHT"),
                        help="Figure size in inches (default 6.5 9.0 = full journal page)")
    parser.add_argument("--dpi", type=int, default=600,
                        help="Output resolution for raster formats (default 600)")
    parser.add_argument("--layer", type=int, default=None,
                        help="Layer index for a two-dimensional HRU parameter")
    parser.add_argument("--save-arrays", action="store_true",
                        help="Save each reconstructed map as a compressed NPZ file")
    parser.add_argument("--boundaries", default=None,
                        help=("Path or URL to a state/admin boundary shapefile or zip "
                              "(e.g. Natural Earth ne_50m_admin_1_states_provinces.zip)"))
    parser.add_argument("--no-letters", action="store_true",
                        help="Do not add (a)-(h) panel letters")
    return parser.parse_args()


def available_cids(edir, requested):
    """Return the requested CIDs or all numeric CID directories."""
    if requested:
        cids = [int(v.strip()) for v in requested.split(",") if v.strip()]
    else:
        cids = sorted(
            int(name) for name in os.listdir(edir)
            if name.isdigit() and os.path.isdir(os.path.join(edir, name))
        )
    if not cids:
        raise RuntimeError(f"No numeric CID directories found under {edir}")
    return cids


def read_domain_raster(path, buffer_pixels):
    """Read a domain raster, optionally removing outer pixels."""
    if buffer_pixels < 0:
        raise ValueError("--buffer must be zero or greater")

    with rasterio.open(path) as source:
        if 2 * buffer_pixels >= min(source.width, source.height):
            raise ValueError(f"--buffer {buffer_pixels} is too large for {path}")

        window = Window(buffer_pixels, buffer_pixels,
                        source.width - 2 * buffer_pixels,
                        source.height - 2 * buffer_pixels)
        array = source.read(1, window=window, masked=False)
        transform = window_transform(window, source.transform)
        bounds = rasterio.transform.array_bounds(array.shape[0], array.shape[1], transform)
        nodata = source.nodata
        crs = source.crs

    return array, nodata, bounds, crs


def read_parameter(ncpath, variable, layer):
    """Read one HRU parameter from input_file.nc."""
    with nc.Dataset(ncpath, "r") as dataset:
        if "parameters" not in dataset.groups:
            raise KeyError("missing group 'parameters'")
        group = dataset.groups["parameters"]
        if variable not in group.variables:
            raise KeyError(f"missing parameter '{variable}'")
        values = np.ma.asarray(group.variables[variable][:])

    values = np.squeeze(np.asarray(np.ma.filled(values, np.nan), dtype=np.float64))

    if values.ndim == 1:
        return values.astype(np.float32, copy=False)

    if values.ndim == 2 and layer is not None:
        if layer < 0 or layer >= values.shape[1]:
            raise IndexError(f"layer {layer} is outside parameter shape {values.shape}")
        return values[:, layer].astype(np.float32, copy=False)

    raise ValueError(f"{variable} has shape {values.shape}; use --layer for a layered parameter")


def paint_variable(edir, variable, cids_to_use, hrus, cids_map, hru_nodata, layer):
    """Paint HRU parameter values back onto the spatial raster."""
    result = np.full(hrus.shape, np.nan, dtype=np.float32)
    number_mapped = 0
    skipped = []

    for cid in cids_to_use:
        ncpath = os.path.join(edir, str(cid), "input_file.nc")
        if not os.path.isfile(ncpath):
            skipped.append((cid, "missing input_file.nc"))
            continue

        try:
            values = read_parameter(ncpath, variable, layer)
        except (KeyError, ValueError, IndexError, OSError) as error:
            skipped.append((cid, str(error)))
            continue

        cid_mask = cids_map == cid
        if not np.any(cid_mask):
            skipped.append((cid, "CID is absent from cids.vrt"))
            continue

        raw_hru_ids = hrus[cid_mask]
        valid = np.isfinite(raw_hru_ids)
        if hru_nodata is not None:
            valid &= raw_hru_ids != hru_nodata

        hru_ids = np.zeros(raw_hru_ids.shape, dtype=np.int64)
        hru_ids[valid] = raw_hru_ids[valid].astype(np.int64)
        valid &= hru_ids >= 0
        valid &= hru_ids < values.size

        if np.any(valid):
            flat_positions = np.flatnonzero(cid_mask)
            result.flat[flat_positions[valid]] = values[hru_ids[valid]]
            number_mapped += 1
        else:
            skipped.append((cid, "no valid HRU indices"))

    return result, number_mapped, skipped


def convert_units(variable, mapped):
    """Convert applicable angular parameters to degrees."""
    mapped = mapped.astype(np.float64, copy=False)
    if variable in RADIAN_VARIABLES:
        mapped = np.degrees(mapped)
    return mapped


def get_plot_options(variable, mapped):
    """Return the colormap and color normalization."""
    finite_values = mapped[np.isfinite(mapped)]
    minimum = float(np.min(finite_values))
    maximum = float(np.max(finite_values))

    cmap = VARIABLE_CMAPS.get(variable, "viridis")
    norm = None
    vmin, vmax = minimum, maximum

    if variable in {"x_aspect", "y_aspect"}:
        limit = max(abs(minimum), abs(maximum)) or 1.0
        norm = TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
        vmin = vmax = None
    elif variable in {"svf", "tvf"}:
        vmin, vmax = 0.0, 1.0
    elif np.isclose(vmin, vmax):
        difference = max(abs(vmin) * 0.01, 1.0e-6)
        vmin -= difference
        vmax += difference

    return cmap, norm, vmin, vmax


def valid_domain_limits(hrus, cids_map, hru_nodata, bounds, border_fraction):
    """Calculate the valid-data bounds plus a small plotting border."""
    if border_fraction < 0:
        raise ValueError("--border must be zero or greater")

    valid = np.isfinite(hrus) & np.isfinite(cids_map) & (cids_map > 0)
    if hru_nodata is not None:
        valid &= hrus != hru_nodata

    rows, columns = np.where(valid)
    if rows.size == 0:
        raise RuntimeError("No valid HRU/CID pixels were found in the domain rasters")

    left, bottom, right, top = bounds
    raster_height, raster_width = hrus.shape
    pixel_width = (right - left) / float(raster_width)
    pixel_height = (top - bottom) / float(raster_height)

    valid_left = left + columns.min() * pixel_width
    valid_right = left + (columns.max() + 1) * pixel_width
    valid_top = top - rows.min() * pixel_height          # rows start at the top
    valid_bottom = top - (rows.max() + 1) * pixel_height

    pad_x = border_fraction * (valid_right - valid_left)
    pad_y = border_fraction * (valid_top - valid_bottom)

    return ((valid_left - pad_x, valid_right + pad_x),
            (valid_bottom - pad_y, valid_top + pad_y))


def load_boundaries(source, raster_crs):
    """Load a state/admin boundary layer and match it to the raster CRS."""
    try:
        boundaries = gpd.read_file(source)
    except Exception as error:  # fiona/pyogrio raise various error types
        sys.exit(
            f"Could not read --boundaries source '{source}': {error}\n"
            "If this is a URL, confirm the running machine has outbound "
            "internet access (compute nodes often do not). Otherwise, "
            "download the file once and pass a local path instead."
        )

    if raster_crs is not None and boundaries.crs is not None:
        if boundaries.crs != raster_crs:
            boundaries = boundaries.to_crs(raster_crs)

    return boundaries


def add_boundaries(axis, boundaries, plot_xlim, plot_ylim):
    """Overlay state boundary lines clipped to the plot extent."""
    clipped = boundaries.cx[plot_xlim[0]:plot_xlim[1], plot_ylim[0]:plot_ylim[1]]
    if clipped.empty:
        return
    clipped.boundary.plot(ax=axis, edgecolor="black", linewidth=0.5, zorder=3)
    # boundary.plot can change limits; restore the requested extent
    axis.set_xlim(plot_xlim)
    axis.set_ylim(plot_ylim)


def main():
    args = parse_args()

    edir = os.path.abspath(args.edir)
    output_path = os.path.abspath(args.out)
    hrupath = os.path.join(edir, "postprocess", "hrus.vrt")
    cidpath = os.path.join(edir, "postprocess", "cids.vrt")

    for required_path in (edir, hrupath, cidpath):
        if not os.path.exists(required_path):
            sys.exit(f"Not found: {required_path}")

    try:
        cids_to_use = available_cids(edir, args.cids)
        hrus, hru_nodata, bounds, raster_crs = read_domain_raster(hrupath, args.buffer)
        cids_map, _, cid_bounds, _ = read_domain_raster(cidpath, args.buffer)
    except (RuntimeError, ValueError, OSError) as error:
        sys.exit(str(error))

    if hrus.shape != cids_map.shape or not np.allclose(bounds, cid_bounds):
        sys.exit("hrus.vrt and cids.vrt do not have the same cropped grid")

    try:
        plot_xlim, plot_ylim = valid_domain_limits(hrus, cids_map, hru_nodata,
                                                   bounds, args.border)
    except (RuntimeError, ValueError) as error:
        sys.exit(str(error))

    print("HRU raster shape:", hrus.shape, flush=True)
    print("Number of CIDs:", len(cids_to_use), flush=True)
    print("Plot longitude limits:", plot_xlim, flush=True)
    print("Plot latitude limits:", plot_ylim, flush=True)

    boundaries = None
    if args.boundaries:
        print(f"Loading boundaries from {args.boundaries} ...", flush=True)
        boundaries = load_boundaries(args.boundaries, raster_crs)

    ncols = max(1, min(args.cols, len(args.vars)))
    nrows = int(math.ceil(len(args.vars) / float(ncols)))

    fig, axes = plt.subplots(nrows, ncols, figsize=tuple(args.figsize),
                             squeeze=False, sharex=True, sharey=True)
    axes_grid = axes
    axes = axes.ravel()

    left, bottom, right, top = bounds
    extent = (left, right, bottom, top)

    output_directory = os.path.dirname(output_path)
    if output_directory:
        os.makedirs(output_directory, exist_ok=True)
    output_base = os.path.splitext(output_path)[0]

    letters = string.ascii_lowercase

    for index, variable in enumerate(args.vars):
        print(f"Mapping {variable} ...", flush=True)

        mapped, number_mapped, skipped = paint_variable(
            edir, variable, cids_to_use, hrus, cids_map, hru_nodata, args.layer)
        mapped = convert_units(variable, mapped)

        if not np.any(np.isfinite(mapped)):
            plt.close(fig)
            reasons = "; ".join(f"CID {cid}: {reason}" for cid, reason in skipped[:5])
            sys.exit(f"No values mapped for {variable}. {reasons}")

        axis = axes[index]
        row, col = divmod(index, ncols)
        cmap, norm, vmin, vmax = get_plot_options(variable, mapped)

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

        axis.set_xlim(plot_xlim)
        axis.set_ylim(plot_ylim)

        if boundaries is not None:
            add_boundaries(axis, boundaries, plot_xlim, plot_ylim)

        axis.set_title(VARIABLE_TITLES.get(variable, variable),
                       fontsize=FONT_TITLE, pad=3)

        # Axis labels only on the outer panels
        is_bottom = row == nrows - 1 or index + ncols >= len(args.vars)
        is_left = col == 0
        axis.set_xlabel("Longitude (°)" if is_bottom else "", fontsize=FONT_LABEL)
        axis.set_ylabel("Latitude (°)" if is_left else "", fontsize=FONT_LABEL)
        axis.tick_params(axis="both", labelsize=FONT_TICK,
                         labelbottom=is_bottom, labelleft=is_left)

        if not args.no_letters:
            axis.text(0.03, 0.97, f"({letters[index]})", transform=axis.transAxes,
                      ha="left", va="top", fontsize=FONT_LETTER, fontweight="bold",
                      bbox=dict(boxstyle="square,pad=0.15", facecolor="white",
                                edgecolor="none", alpha=0.8),
                      zorder=5)

        # Colorbar with the same height as the map
        divider = make_axes_locatable(axis)
        colorbar_axis = divider.append_axes("right", size="5%", pad=0.05)
        colorbar = fig.colorbar(image, cax=colorbar_axis)
        colorbar.set_label(COLORBAR_LABELS.get(variable, variable), fontsize=FONT_LABEL)
        colorbar.ax.tick_params(labelsize=FONT_TICK, width=0.6, length=2)
        colorbar.outline.set_linewidth(0.6)

        print(f"  CIDs mapped: {number_mapped}/{len(cids_to_use)}; "
              f"range: {np.nanmin(mapped):.6g} to {np.nanmax(mapped):.6g}; "
              f"skipped: {len(skipped)}", flush=True)
        for cid, reason in skipped[:5]:
            print(f"    CID {cid}: {reason}", flush=True)

        if args.save_arrays:
            np.savez_compressed(
                f"{output_base}_{variable}.npz",
                data=mapped,
                extent=np.asarray(extent),
                plot_xlim=np.asarray(plot_xlim),
                plot_ylim=np.asarray(plot_ylim),
                units=COLORBAR_LABELS.get(variable, variable),
            )

    for axis in axes[len(args.vars):]:
        axis.axis("off")

    fig.subplots_adjust(left=0.09, right=0.93, bottom=0.05, top=0.975,
                        wspace=0.30, hspace=0.16)

    fig.savefig(output_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Wrote {output_path}", flush=True)


if __name__ == "__main__":
    main()