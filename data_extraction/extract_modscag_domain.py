#!/usr/bin/env python3
"""Extract monthly STC-MODSCAG fields for a HydroBlocks-domain extent.

Daily files are opened lazily by tile in the native MODIS sinusoidal CRS.
For each month, tiles are mosaicked, the temporal mean is computed on the
native grid, and only that single monthly field is reprojected to EPSG:4326
and clipped. This follows the efficient processing order used in lecture 13.
"""

import argparse
import glob
import os
import re
from datetime import datetime

import numpy as np
import rasterio
import rioxarray as rxr  # noqa: F401 - registers the .rio accessor
import xarray as xr

try:
    from mpi4py import MPI

    _COMM = MPI.COMM_WORLD
    _RANK = _COMM.Get_rank()
    _SIZE = _COMM.Get_size()
except ImportError:
    _COMM = None
    _RANK = 0
    _SIZE = 1


SINU_PROJ = (
    "+proj=sinu +lon_0=0 +x_0=0 +y_0=0 +a=6371007.181 "
    "+b=6371007.181 +units=m +no_defs"
)
DEFAULT_VARIABLES = ["albedo_dirty_terrain_corrected", "snow_fraction"]
DATE_PATTERN = re.compile(r"(\d{8})")
_WARNED_UNPARSEABLE = False


def log(message):
    if _SIZE > 1:
        print(f"[rank {_RANK:03d}/{_SIZE}] {message}", flush=True)
    else:
        print(message, flush=True)


def filter_files_by_month(files, year, month):
    global _WARNED_UNPARSEABLE
    if month is None:
        return files

    kept = []
    for file_path in files:
        basename = os.path.basename(file_path)
        match = DATE_PATTERN.search(basename)
        if not match:
            if not _WARNED_UNPARSEABLE:
                log(
                    "WARNING: filenames without YYYYMMDD cannot be filtered "
                    f"by month; retaining '{basename}'"
                )
                _WARNED_UNPARSEABLE = True
            kept.append(file_path)
            continue
        try:
            date = datetime.strptime(match.group(1), "%Y%m%d")
        except ValueError:
            kept.append(file_path)
            continue
        if date.year == year and date.month == month:
            kept.append(file_path)
    return kept


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--tiles", nargs="+", default=None)
    parser.add_argument("--variables", nargs="+", default=DEFAULT_VARIABLES)
    parser.add_argument("--start-year", type=int, required=True)
    parser.add_argument("--end-year", type=int, required=True)
    parser.add_argument(
        "--month", type=int, choices=range(1, 13), metavar="1-12", default=None
    )
    parser.add_argument(
        "--bbox",
        type=float,
        nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        default=None,
    )
    parser.add_argument("--bbox-from-raster", default=None)
    parser.add_argument("--out-res", type=float, default=None)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--file-glob", default="*.nc")
    parser.add_argument("--concat-output", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def bbox_from_raster(raster_path):
    with rasterio.open(raster_path) as dataset:
        if dataset.crs is not None and dataset.crs.to_epsg() != 4326:
            log(
                f"WARNING: bounding-box raster CRS is {dataset.crs}, not "
                "EPSG:4326; bounds will be used as supplied"
            )
        bounds = dataset.bounds
        return bounds.left, bounds.bottom, bounds.right, bounds.top


def discover_tiles(data_dir, requested_tiles):
    if requested_tiles:
        return requested_tiles
    tiles = [
        entry
        for entry in sorted(os.listdir(data_dir))
        if re.fullmatch(r"h\d{2}v\d{2}", entry)
        and os.path.isdir(os.path.join(data_dir, entry))
    ]
    if not tiles:
        raise RuntimeError(f"No MODIS tile directories found under {data_dir}")
    return tiles


def open_tile_month(data_dir, tile, year, month, variables, file_glob):
    year_dir = os.path.join(data_dir, tile, str(year))
    files = sorted(glob.glob(os.path.join(year_dir, file_glob)))
    files = filter_files_by_month(files, year, month)
    if not files:
        return None

    dataset = xr.open_mfdataset(
        files,
        concat_dim="time",
        combine="nested",
        data_vars="all",
        chunks="auto",
    )
    missing = [name for name in variables if name not in dataset.data_vars]
    if missing:
        raise KeyError(
            f"Variables {missing} missing from tile {tile}, year {year}; "
            f"available variables: {sorted(dataset.data_vars)}"
        )

    dataset = dataset[variables].astype("float32")
    if dataset.rio.crs is None:
        dataset = dataset.rio.write_crs(SINU_PROJ)
    return dataset.rio.set_spatial_dims(x_dim="x", y_dim="y")


def merge_tiles(tile_datasets):
    # Outer alignment is required because adjacent tiles have different x/y
    # coordinates. Making it explicit also avoids Xarray's FutureWarning.
    merged = xr.combine_by_coords(
        tile_datasets,
        combine_attrs="override",
        join="outer",
    )
    merged = merged.rio.write_crs(SINU_PROJ)
    return merged.rio.set_spatial_dims(x_dim="x", y_dim="y")


def reproject_and_clip(dataset, bbox, out_res):
    if out_res is None:
        reprojected = dataset.rio.reproject("EPSG:4326")
    else:
        reprojected = dataset.rio.reproject("EPSG:4326", resolution=out_res)

    min_lon, min_lat, max_lon, max_lat = bbox
    return reprojected.rio.clip_box(
        minx=min_lon,
        miny=min_lat,
        maxx=max_lon,
        maxy=max_lat,
    )


def process_year_month(args, tiles, year, month):
    output_path = os.path.join(
        args.outdir, f"modscag_monthly_mean_{year}{month:02d}.nc"
    )
    if args.skip_existing and os.path.exists(output_path):
        log(f"{year}-{month:02d}: already exists, skipping")
        return output_path

    tile_datasets = []
    day_counts = []
    for tile in tiles:
        dataset = open_tile_month(
            args.data_dir,
            tile,
            year,
            month,
            args.variables,
            args.file_glob,
        )
        if dataset is not None:
            tile_datasets.append(dataset)
            day_counts.append(dataset.sizes.get("time", 0))

    if not tile_datasets:
        log(f"{year}-{month:02d}: no data found")
        return None

    log(f"{year}-{month:02d}: merging {len(tile_datasets)} tiles")
    merged = merge_tiles(tile_datasets)

    # Lecture-13 order: reduce the daily time dimension on the native
    # sinusoidal grid before doing the expensive reprojection.
    n_days = merged.sizes.get("time", max(day_counts, default=0))
    log(f"{year}-{month:02d}: averaging {n_days} daily fields on native grid")
    monthly_native = merged.mean(dim="time", skipna=True).compute()
    monthly_native = monthly_native.rio.write_crs(SINU_PROJ)
    monthly_native = monthly_native.rio.set_spatial_dims(x_dim="x", y_dim="y")

    log(f"{year}-{month:02d}: reprojecting monthly mean to EPSG:4326")
    monthly_mean = reproject_and_clip(
        monthly_native,
        args.bbox,
        args.out_res,
    )

    month_start = np.datetime64(f"{year:04d}-{month:02d}-01")
    monthly_mean = monthly_mean.expand_dims(time=[month_start])

    encoding = {
        variable: {"zlib": True, "complevel": 4} for variable in args.variables
    }
    monthly_mean.to_netcdf(output_path, encoding=encoding)
    log(f"{year}-{month:02d}: saved {output_path}")
    return output_path


def main():
    args = arguments()
    if args.bbox_from_raster:
        args.bbox = bbox_from_raster(args.bbox_from_raster)
    elif args.bbox is None:
        raise ValueError("Provide --bbox or --bbox-from-raster")

    os.makedirs(args.outdir, exist_ok=True)
    tiles = discover_tiles(args.data_dir, args.tiles)
    months = [args.month] if args.month is not None else list(range(1, 13))
    all_tasks = [
        (year, month)
        for year in range(args.start_year, args.end_year + 1)
        for month in months
    ]
    my_tasks = all_tasks[_RANK::_SIZE]
    log(f"Assigned {len(my_tasks)} of {len(all_tasks)} year-month tasks")

    my_outputs = []
    for year, month in my_tasks:
        output = process_year_month(args, tiles, year, month)
        if output is not None:
            my_outputs.append(output)

    if _COMM is not None:
        _COMM.Barrier()
        gathered = _COMM.gather(my_outputs, root=0)
    else:
        gathered = [my_outputs]

    if _RANK == 0:
        output_paths = sorted(path for group in gathered for path in group)
        log(f"All ranks finished; {len(output_paths)} monthly files available")

        if args.concat_output and output_paths:
            combined = xr.open_mfdataset(
                output_paths,
                concat_dim="time",
                combine="nested",
                data_vars="all",
            )
            combined_path = os.path.join(
                args.outdir,
                f"modscag_monthly_mean_{args.start_year}_{args.end_year}.nc",
            )
            encoding = {
                variable: {"zlib": True, "complevel": 4}
                for variable in args.variables
            }
            combined.to_netcdf(combined_path, encoding=encoding)
            log(f"Saved combined output: {combined_path}")


if __name__ == "__main__":
    main()
