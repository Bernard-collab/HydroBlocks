#!/usr/bin/env python3
"""Plot postprocessed SWDN 3D−PP by month and approximate Mountain Time."""

import argparse
import glob
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import rasterio
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject


COMBINATIONS = [(12, 8), (7, 8), (12, 11), (7, 11),
                (12, 14), (7, 14), (12, 17), (7, 17)]


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--edir-3d", required=True)
    parser.add_argument("--edir-pp", required=True)
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2024)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--stride", type=int, default=2)
    parser.add_argument("--bins", type=int, default=50)
    parser.add_argument("--dpi", type=int, default=400)
    parser.add_argument("--dem-raster", default=None)
    parser.add_argument("--outdir", required=True)
    return parser.parse_args()


def paired_files(edir_3d, edir_pp, start_year, end_year):
    folder_3d = os.path.join(edir_3d, "postprocess", "output_dir")
    folder_pp = os.path.join(edir_pp, "postprocess", "output_dir")
    files_3d = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_3d, "????????.nc"))}
    files_pp = {os.path.basename(f): f for f in glob.glob(os.path.join(folder_pp, "????????.nc"))}
    pairs = []
    for name in sorted(files_3d.keys() & files_pp.keys()):
        year, month, day = int(name[:4]), int(name[4:6]), int(name[6:8])
        needed = month in (7, 12) or (day == 1 and month in (1, 8))
        if start_year <= year <= end_year and needed:
            pairs.append((files_3d[name], files_pp[name]))
    if not pairs:
        raise FileNotFoundError("No paired July/December postprocessed files found")
    return pairs


def local_times(path, steps):
    date = pd.Timestamp(os.path.basename(path)[:8], tz="UTC")
    utc = date + pd.to_timedelta(np.arange(steps) * 3, unit="h")
    return utc.tz_convert("America/Denver")


def process_chunk(chunk, shape, start_year, end_year):
    sums = {key: np.zeros(shape, dtype=np.float64) for key in COMBINATIONS}
    counts = {key: np.zeros(shape, dtype=np.uint32) for key in COMBINATIONS}
    for file_3d, file_pp in chunk:
        with nc.Dataset(file_3d) as ds_3d, nc.Dataset(file_pp) as ds_pp:
            swdn_3d = np.ma.asarray(ds_3d.variables["swdn_3d"][:]).filled(np.nan).astype(float)
            swdn_pp = np.ma.asarray(ds_pp.variables["swdn"][:]).filled(np.nan).astype(float)

        if swdn_3d.shape != swdn_pp.shape:
            raise ValueError(f"Shape mismatch: {file_3d} and {file_pp}")

        times = local_times(file_3d, swdn_3d.shape[0])

        for key in COMBINATIONS:
            month, hour = key
            selected = ((times.month == month) &
                        (times.year >= start_year) & (times.year <= end_year) &
                        ((times.hour == hour) | (times.hour == hour + 1)))
            if not np.any(selected):
                continue

            difference = swdn_3d[selected] - swdn_pp[selected]
            finite_count = np.sum(np.isfinite(difference), axis=0)
            daily_mean = np.full(shape, np.nan, dtype=float)
            np.divide(np.nansum(difference, axis=0), finite_count,
                      out=daily_mean, where=finite_count > 0)
            valid = np.isfinite(daily_mean)
            sums[key][valid] += daily_mean[valid]
            counts[key][valid] += 1

    return sums, counts


def combine(results, shape):
    sums = {key: np.zeros(shape, dtype=np.float64) for key in COMBINATIONS}
    counts = {key: np.zeros(shape, dtype=np.uint32) for key in COMBINATIONS}
    for partial_sums, partial_counts in results:
        for key in COMBINATIONS:
            sums[key] += partial_sums[key]
            counts[key] += partial_counts[key]

    maps = {}
    for key in COMBINATIONS:
        maps[key] = np.full(shape, np.nan, dtype=float)
        np.divide(sums[key], counts[key], out=maps[key], where=counts[key] > 0)
        if not np.any(np.isfinite(maps[key])):
            raise RuntimeError(f"No valid values for month={key[0]}, hour≈{key[1]} MT")
    return maps


def map_plot(lon, lat, maps, output, dpi, years):
    limit = 400
    norm = colors.TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit)
    fig, axes = plt.subplots(4, 2, figsize=(13, 17), sharex=True, sharey=True)

    for axis, key in zip(axes.ravel(), COMBINATIONS):
        month, hour = key
        image = axis.pcolormesh(lon, lat, maps[key], cmap="RdBu_r",
                                norm=norm, shading="auto", rasterized=True)
        axis.set_title(f"Month = {month}, hour ≈ {hour} MT",
                       fontweight="bold")
        axis.set_xlabel("Longitude [°]")
        axis.set_ylabel("Latitude [°]")
        axis.set_aspect("equal")

    fig.subplots_adjust(top=0.95, right=0.88, hspace=0.25, wspace=0.20)
    colorbar_axis = fig.add_axes([0.90, 0.35, 0.018, 0.30])
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("SWDN 3D − PP [W m⁻²]")
    fig.suptitle(f"SWDN 3D − PP by Month and Local Time ({years})", fontsize=16)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def dem_on_grid(path, lon, lat):
    dx, dy = abs(np.diff(lon).mean()), abs(np.diff(lat).mean())
    transform = from_bounds(lon.min() - dx / 2, lat.min() - dy / 2,
                            lon.max() + dx / 2, lat.max() + dy / 2,
                            len(lon), len(lat))
    north_up = np.full((len(lat), len(lon)), np.nan, dtype=np.float32)
    with rasterio.open(path) as source:
        reproject(source=rasterio.band(source, 1), destination=north_up,
                  src_transform=source.transform, src_crs=source.crs,
                  src_nodata=source.nodata, dst_transform=transform,
                  dst_crs="EPSG:4326", dst_nodata=np.nan,
                  resampling=Resampling.average)
    return np.flipud(north_up) if lat[0] < lat[-1] else north_up


def terrain_plot(lon, lat, dem, maps, output, stride, years):
    limit = max(np.nanmax(np.abs(values)) for values in maps.values())
    titles = [f"Month = {key[0]}, hour ≈ {key[1]} MT"
              for key in COMBINATIONS]
    fig = make_subplots(rows=4, cols=2,
                        specs=[[{"type": "surface"}, {"type": "surface"}]] * 4,
                        subplot_titles=titles)
    x, y = lon[::stride], lat[::stride]
    z = dem[::stride, ::stride]

    for k, key in enumerate(COMBINATIONS):
        fig.add_trace(go.Surface(
            x=x, y=y, z=z, surfacecolor=maps[key][::stride, ::stride],
            colorscale="RdBu", reversescale=True, cmin=-limit, cmax=limit,
            showscale=(k == 0),
            colorbar=dict(title="SWDN 3D − PP<br>[W m⁻²]", len=0.75)),
            row=k // 2 + 1, col=k % 2 + 1)

    camera = dict(eye=dict(x=1.6, y=1.6, z=0.6))
    for scene in fig.select_scenes():
        scene.update(camera=camera, xaxis_title="Longitude",
                     yaxis_title="Latitude", zaxis_title="Elevation [m]")
    fig.update_layout(title=f"SWDN 3D − PP by Month and Local Time ({years})",
                      height=1400, width=1100)
    fig.write_html(output)


def histogram_plot(maps, output, bins, dpi, years):
    fig, axes = plt.subplots(4, 2, figsize=(12, 17))
    for axis, key in zip(axes.ravel(), COMBINATIONS):
        values = maps[key][np.isfinite(maps[key])]
        md = np.mean(values)
        mad = np.mean(np.abs(values))
        rmse = np.sqrt(np.mean(values ** 2))
        axis.hist(values, bins=bins, density=True)
        axis.axvline(md, color="darkred", linestyle="dashed", linewidth=3)
        axis.set_title(f"Month = {key[0]}, hour ≈ {key[1]} MT", fontweight="bold")
        axis.set_xlabel("SWDN 3D − PP [W m⁻²]")
        axis.set_ylabel("Probability density")
        axis.text(0.98, 0.98,
                  f"MD = {md:.2f} W m⁻²\nMAD = {mad:.2f} W m⁻²\n"
                  f"RMSE = {rmse:.2f} W m⁻²\nn = {values.size:,}",
                  transform=axis.transAxes, ha="right", va="top",
                  bbox=dict(boxstyle="round", facecolor="lightgray",
                            edgecolor="black", alpha=0.6))
    fig.suptitle(f"SWDN 3D − PP Difference Distributions ({years})", fontsize=16)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main():
    args = arguments()
    if args.end_year < args.start_year or args.workers < 1 or args.stride < 1:
        raise ValueError("Check end-year, workers, and stride")

    pairs = paired_files(args.edir_3d, args.edir_pp, args.start_year, args.end_year)
    with nc.Dataset(pairs[0][0]) as sample:
        lon = np.asarray(sample.variables["lon"][:])
        lat = np.asarray(sample.variables["lat"][:])
    shape = (len(lat), len(lon))

    workers = min(args.workers, len(pairs))
    chunk_size = (len(pairs) + workers - 1) // workers
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]
    print(f"Paired files: {len(pairs)}; workers: {workers}", flush=True)

    results = []
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = [executor.submit(process_chunk, chunk, shape,
                                   args.start_year, args.end_year)
                   for chunk in chunks]
        for number, future in enumerate(as_completed(futures), 1):
            results.append(future.result())
            print(f"Worker chunks completed: {number}/{len(chunks)}", flush=True)

    maps = combine(results, shape)
    dem_path = args.dem_raster or os.path.join(args.edir_3d, "postprocess", "dem.vrt")
    dem = dem_on_grid(dem_path, lon, lat)
    os.makedirs(args.outdir, exist_ok=True)
    years = f"{args.start_year}–{args.end_year}"

    map_output = os.path.join(args.outdir,
        f"swdn_3D_minus_PP_month_hour_maps_{args.start_year}_{args.end_year}.png")
    terrain_output = os.path.join(args.outdir,
        f"swdn_3D_minus_PP_month_hour_terrain_{args.start_year}_{args.end_year}.html")
    histogram_output = os.path.join(args.outdir,
        f"swdn_3D_minus_PP_month_hour_histograms_{args.start_year}_{args.end_year}.png")

    map_plot(lon, lat, maps, map_output, args.dpi, years)
    terrain_plot(lon, lat, dem, maps, terrain_output, args.stride, years)
    histogram_plot(maps, histogram_output, args.bins, args.dpi, years)
    print(f"Saved: {map_output}\nSaved: {terrain_output}\nSaved: {histogram_output}", flush=True)


if __name__ == "__main__":
    main()
