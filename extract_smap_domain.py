#!/usr/bin/env python3
"""Extract both daily NSIDC-0779 SMAP bands for a HydroBlocks domain.

Each MPI rank receives a subset of daily global GeoTIFFs. Only the source
window overlapping the requested geographic bounding box is read, invalid
values (NaN, infinity, zero, and values outside 0-1 m3/m3) are masked, and
the subset is reprojected to a fixed EPSG:4326 grid.

The output remains a two-band daily GeoTIFF:
  Band 1: 6 AM descending-pass surface soil moisture [m3/m3]
  Band 2: 6 PM ascending-pass surface soil moisture [m3/m3]

Example
-------
mpiexec -n 22 python extract_smap_domain.py \
  --data-dir "/path/to/SMAP_0779_2015_2024_full domain" \
  --start-date 2015-04-01 --end-date 2024-06-29 \
  --bbox-from-raster /path/to/experiment/postprocess/hrus.vrt \
  --out-res 0.01 \
  --outdir /path/to/smap_domain_extract
"""

import argparse
import glob
import math
import os
import re
from datetime import datetime

import numpy as np
import rasterio
from rasterio.enums import Resampling
from rasterio.transform import from_origin
from rasterio.windows import Window, from_bounds
from rasterio.warp import reproject, transform_bounds

try:
    from mpi4py import MPI

    COMM = MPI.COMM_WORLD
    RANK = COMM.Get_rank()
    SIZE = COMM.Get_size()
except ImportError:
    COMM = None
    RANK = 0
    SIZE = 1


DATE_PATTERN = re.compile(r"_(\d{8})\.tif$")
BAND_NAMES = {
    1: "soil_moisture_6am_descending",
    2: "soil_moisture_6pm_ascending",
}


def log(message):
    if SIZE > 1:
        print(f"[rank {RANK:03d}/{SIZE}] {message}", flush=True)
    else:
        print(message, flush=True)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--start-date", default="2015-04-01")
    parser.add_argument("--end-date", default="2024-06-29")
    parser.add_argument(
        "--bbox", type=float, nargs=4,
        metavar=("MINLON", "MINLAT", "MAXLON", "MAXLAT"),
        default=None,
        help="Geographic bounds in EPSG:4326",
    )
    parser.add_argument(
        "--bbox-from-raster", default=None,
        help="Read the domain bounds from a raster or VRT; bounds are transformed to EPSG:4326",
    )
    parser.add_argument(
        "--out-res", type=float, default=0.01,
        help="Output pixel size in geographic degrees (default: 0.01)",
    )
    parser.add_argument(
        "--resampling", choices=("bilinear", "nearest"), default="bilinear",
        help="Reprojection resampling method for continuous soil moisture",
    )
    parser.add_argument(
        "--file-glob", default="NSIDC-0779_EASE2_G1km_SMAP_SM_DS_*.tif"
    )
    parser.add_argument("--skip-existing", action="store_true")
    return parser.parse_args()


def parse_date(filename):
    match = DATE_PATTERN.search(os.path.basename(filename))
    if not match:
        return None
    return datetime.strptime(match.group(1), "%Y%m%d").date()


def discover_files(args):
    start = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end = datetime.strptime(args.end_date, "%Y-%m-%d").date()
    if end < start:
        raise ValueError("end-date must not precede start-date")

    records = []
    for filename in glob.glob(os.path.join(args.data_dir, args.file_glob)):
        date = parse_date(filename)
        if date is not None and start <= date <= end:
            records.append((date, filename))
    records.sort(key=lambda item: item[0])
    if not records:
        raise FileNotFoundError("No SMAP GeoTIFFs found for the requested dates")
    return records


def bbox_from_raster(raster_path):
    """Return raster/VRT bounds transformed to EPSG:4326."""
    with rasterio.open(raster_path) as dataset:
        if dataset.crs is None:
            raise ValueError(f"Bounding-box raster has no CRS: {raster_path}")
        bounds = dataset.bounds
        if dataset.crs.to_epsg() == 4326:
            geographic = (
                bounds.left, bounds.bottom, bounds.right, bounds.top
            )
        else:
            geographic = transform_bounds(
                dataset.crs,
                "EPSG:4326",
                bounds.left,
                bounds.bottom,
                bounds.right,
                bounds.top,
                densify_pts=21,
            )
    log(
        "Domain bounds from raster "
        f"{raster_path}: {geographic[0]:.8f}, {geographic[1]:.8f}, "
        f"{geographic[2]:.8f}, {geographic[3]:.8f} (EPSG:4326)"
    )
    return geographic


def destination_grid(bbox, resolution):
    min_lon, min_lat, max_lon, max_lat = bbox
    if not (min_lon < max_lon and min_lat < max_lat):
        raise ValueError("bbox must be MINLON MINLAT MAXLON MAXLAT")
    if resolution <= 0:
        raise ValueError("out-res must be positive")
    width = int(math.ceil((max_lon - min_lon) / resolution))
    height = int(math.ceil((max_lat - min_lat) / resolution))
    transform = from_origin(min_lon, max_lat, resolution, resolution)
    return transform, width, height


def padded_source_window(dataset, bbox, padding=2):
    projected = transform_bounds(
        "EPSG:4326", dataset.crs, *bbox, densify_pts=21
    )
    raw = from_bounds(*projected, transform=dataset.transform)
    col_start = max(0, int(math.floor(raw.col_off)) - padding)
    row_start = max(0, int(math.floor(raw.row_off)) - padding)
    col_stop = min(dataset.width, int(math.ceil(raw.col_off + raw.width)) + padding)
    row_stop = min(dataset.height, int(math.ceil(raw.row_off + raw.height)) + padding)
    if col_stop <= col_start or row_stop <= row_start:
        raise ValueError("Requested geographic bbox does not intersect the SMAP raster")
    return Window(
        col_off=col_start,
        row_off=row_start,
        width=col_stop - col_start,
        height=row_stop - row_start,
    )


def output_name(input_path):
    stem = os.path.splitext(os.path.basename(input_path))[0]
    return f"{stem}_domain.tif"


def process_file(args, date, input_path, dst_transform, width, height):
    output_path = os.path.join(args.outdir, output_name(input_path))
    if args.skip_existing and os.path.exists(output_path):
        return "skipped", output_path, (0, 0)

    resampling = (
        Resampling.bilinear
        if args.resampling == "bilinear"
        else Resampling.nearest
    )
    destination = np.full((2, height, width), np.nan, dtype=np.float32)
    valid_counts = []

    with rasterio.open(input_path) as source_dataset:
        if source_dataset.count < 2:
            raise ValueError(f"{input_path} contains fewer than two bands")
        if source_dataset.crs is None:
            raise ValueError(f"{input_path} has no coordinate reference system")
        source_window = padded_source_window(source_dataset, args.bbox)
        source_transform = source_dataset.window_transform(source_window)

        for band in (1, 2):
            source = source_dataset.read(band, window=source_window).astype(np.float32)
            valid = np.isfinite(source) & (source > 0.0) & (source <= 1.0)
            valid_counts.append(int(valid.sum()))
            source = np.where(valid, source, np.nan).astype(np.float32)
            reproject(
                source=source,
                destination=destination[band - 1],
                src_transform=source_transform,
                src_crs=source_dataset.crs,
                src_nodata=np.nan,
                dst_transform=dst_transform,
                dst_crs="EPSG:4326",
                dst_nodata=np.nan,
                resampling=resampling,
                num_threads=1,
                init_dest_nodata=True,
            )

    profile = {
        "driver": "GTiff",
        "width": width,
        "height": height,
        "count": 2,
        "dtype": "float32",
        "crs": "EPSG:4326",
        "transform": dst_transform,
        "nodata": np.nan,
        "compress": "LZW",
        "predictor": 3,
        "tiled": True,
        "blockxsize": 256,
        "blockysize": 256,
        "BIGTIFF": "IF_SAFER",
    }

    temporary_path = f"{output_path}.rank{RANK}.tmp"
    try:
        with rasterio.open(temporary_path, "w", **profile) as output:
            output.write(destination)
            output.descriptions = (
                "SMAP 1-km downscaled surface soil moisture, descending pass (6 AM)",
                "SMAP 1-km downscaled surface soil moisture, ascending pass (6 PM)",
            )
            output.update_tags(
                source_product="NSIDC-0779 Version 1",
                date=date.isoformat(),
                units="m3 m-3",
                source_nodata="0 and non-finite values",
                geographic_bbox=" ".join(str(value) for value in args.bbox),
                resampling=args.resampling,
            )
            output.update_tags(
                1, name=BAND_NAMES[1], units="m3 m-3",
                overpass="6 AM", orbit="descending",
            )
            output.update_tags(
                2, name=BAND_NAMES[2], units="m3 m-3",
                overpass="6 PM", orbit="ascending",
            )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            os.remove(temporary_path)

    return "written", output_path, tuple(valid_counts)


def main():
    args = arguments()
    args.data_dir = os.path.abspath(args.data_dir)
    args.outdir = os.path.abspath(args.outdir)
    if args.bbox_from_raster:
        args.bbox = bbox_from_raster(
            os.path.abspath(args.bbox_from_raster)
        )
    elif args.bbox is None:
        raise ValueError("Provide --bbox or --bbox-from-raster")
    os.makedirs(args.outdir, exist_ok=True)

    records = discover_files(args)
    dst_transform, width, height = destination_grid(args.bbox, args.out_res)
    tasks = records[RANK::SIZE]
    log(
        f"Assigned {len(tasks)} of {len(records)} files; output grid "
        f"{height} x {width}, EPSG:4326, resolution {args.out_res:g} degrees"
    )

    written = skipped = failed = 0
    for number, (date, input_path) in enumerate(tasks, 1):
        try:
            status, output_path, counts = process_file(
                args, date, input_path, dst_transform, width, height
            )
            if status == "skipped":
                skipped += 1
            else:
                written += 1
            if number % 10 == 0 or number == len(tasks):
                log(
                    f"{number}/{len(tasks)} {date}: {status}; "
                    f"source valid pixels AM={counts[0]:,}, PM={counts[1]:,}; "
                    f"{os.path.basename(output_path)}"
                )
        except Exception as error:
            failed += 1
            log(f"ERROR {date} {input_path}: {error}")

    local_summary = (written, skipped, failed)
    if COMM is not None:
        summaries = COMM.gather(local_summary, root=0)
        COMM.Barrier()
    else:
        summaries = [local_summary]

    if RANK == 0:
        total_written = sum(item[0] for item in summaries)
        total_skipped = sum(item[1] for item in summaries)
        total_failed = sum(item[2] for item in summaries)
        log(
            f"Finished: written={total_written}, skipped={total_skipped}, "
            f"failed={total_failed}, discovered={len(records)}"
        )
        if total_failed:
            raise RuntimeError(f"Extraction failed for {total_failed} file(s)")


if __name__ == "__main__":
    main()
