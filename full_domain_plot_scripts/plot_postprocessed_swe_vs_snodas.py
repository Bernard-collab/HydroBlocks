#!/usr/bin/env python3
"""Seasonal HydroBlocks SWE versus SNODAS; DJF uses Jan, Feb, Dec of one year.
SWE is in mm. Valid zeros are retained; no minimum day coverage is imposed.
Seasonal pairs are weighted by calendar-season length across years.
SNODAS seasonal files do not identify missing individual observation dates.
"""
import argparse
import calendar
import glob
import os
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.colors as colors
import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
from rasterio.crs import CRS
from rasterio.transform import from_bounds
from rasterio.warp import Resampling, reproject

plt.rcParams.update({'font.size': 15, 'axes.titlesize': 17, 'axes.titleweight': 'bold',
                     'axes.labelsize': 15, 'xtick.labelsize': 13, 'ytick.labelsize': 13})
SEASONS = {'Winter': (12, 1, 2), 'Spring': (3, 4, 5), 'Summer': (6, 7, 8), 'Fall': (9, 10, 11)}
CODES = dict(zip(SEASONS, ('DJF', 'MAM', 'JJA', 'SON')))

def arguments():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edir', required=True)
    p.add_argument('--snodas-file', default='/scratch/alpine/battobrah@xsede.org/HB_NMP_full_domain_comparison_data/snodas_swe_seasonal_30arcsec.nc')
    p.add_argument('--snodas-crs', required=True, help='Verified source CRS; missing from the seasonal file')
    p.add_argument('--start-year', type=int, default=2014)
    p.add_argument('--end-year', type=int, default=2024)
    p.add_argument('--workers', type=int, default=1)
    p.add_argument('--diff-limit', type=float, default=None)
    p.add_argument('--bins', type=int, default=50)
    p.add_argument('--dpi', type=int, default=300)
    p.add_argument('--outdir', required=True)
    a = p.parse_args()
    if a.end_year < a.start_year: p.error('end-year precedes start-year')
    if a.workers < 1 or a.bins < 1: p.error('workers and bins must be positive')
    if a.diff_limit is not None and a.diff_limit <= 0: p.error('diff-limit must be positive')
    if not CRS.from_user_input(a.snodas_crs).is_geographic: p.error('Source coordinates are geographic latitude/longitude')
    return a

def clean(v, selection):
    a = np.ma.asarray(v[selection], dtype=float).filled(np.nan)
    for key in ('_FillValue', 'missing_value'):
        for fill in np.atleast_1d(getattr(v, key, [])): a[a == fill] = np.nan
    a[~np.isfinite(a) | (a < 0)] = np.nan
    return a

def grid(a):
    a = np.asarray(a, dtype=float)
    if a.ndim != 1 or a.size < 2 or not np.isfinite(a).all(): raise ValueError('Invalid grid coordinate')
    d = np.diff(a)
    if not (np.all(d > 0) or np.all(d < 0)) or not np.allclose(d, d[0], rtol=1e-5, atol=1e-9):
        raise ValueError('Coordinates must be monotonic and regularly spaced')
    return a

def bounds(a):
    step = abs(a[1] - a[0])
    return a.min() - step/2, a.max() + step/2

def read_snodas(args, index, lon, lat):
    # Read only the model-domain window of one seasonal field.
    with nc.Dataset(args.snodas_file) as ds:
        x, y = grid(ds.variables['lon'][:]), grid(ds.variables['lat'][:])
        west, east = bounds(lon); south, north = bounds(lat)
        ix = np.flatnonzero((x >= west-abs(x[1]-x[0])) & (x <= east+abs(x[1]-x[0])))
        iy = np.flatnonzero((y >= south-abs(y[1]-y[0])) & (y <= north+abs(y[1]-y[0])))
        if ix.size < 2 or iy.size < 2: raise ValueError('SNODAS does not overlap model grid')
        sx, sy = slice(ix.min(), ix.max()+1), slice(iy.min(), iy.max()+1)
        v = ds.variables['swe']
        if v.dimensions != ('time', 'lat', 'lon'): raise ValueError(f'Unexpected SNODAS dimensions: {v.dimensions}')
        if getattr(v, 'units', '').strip().lower() != 'mm': raise ValueError('Expected SNODAS SWE in mm')
        source = clean(v, (index, sy, sx)); x, y = x[sx], y[sy]
    if x[0] > x[-1]: x, source = x[::-1], source[:, ::-1]
    if y[0] < y[-1]: y, source = y[::-1], source[::-1]
    left, right = bounds(x); bottom, top = bounds(y)
    src_transform = from_bounds(left, bottom, right, top, x.size, y.size)
    dst_transform = from_bounds(west, south, east, north, lon.size, lat.size)
    destination = np.full((lat.size, lon.size), np.nan)
    reproject(source, destination, src_transform=src_transform, src_crs=args.snodas_crs,
              dst_transform=dst_transform, dst_crs='EPSG:4326', src_nodata=np.nan,
              dst_nodata=np.nan, resampling=Resampling.average)
    return destination[::-1]

def process_season(task):
    args, year, season, index, ndays, files, native_lon, native_lat = task
    lon, lat = np.sort(native_lon), np.sort(native_lat)
    xo, yo = np.argsort(native_lon), np.argsort(native_lat)
    expected = sum(calendar.monthrange(year, month)[1] for month in SEASONS[season])
    if len(files) != expected or ndays != expected:
        print(f'Warning: {year} {CODES[season]} uses available days; model={len(files)}/{expected}, SNODAS={ndays}/{expected}', flush=True)
    total = np.zeros((lat.size, lon.size)); count = np.zeros_like(total, dtype=int)
    for filename in files:
        with nc.Dataset(filename) as ds:
            if not np.array_equal(ds.variables['lon'][:], native_lon) or not np.array_equal(ds.variables['lat'][:], native_lat):
                raise ValueError(f'Model grid changes: {filename}')
            v = ds.variables['swe']
            dims = [d.lower() for d in v.dimensions]
            axes = [next(i for i,d in enumerate(dims) if d in names)
                    for names in (('t','time'), ('lat','latitude','y'), ('lon','longitude','x'))]
            values = np.transpose(clean(v, slice(None)), axes)[:, yo, :][:, :, xo]
        # Give each available model day equal weight, retaining zero SWE.
        valid = np.isfinite(values); n = valid.sum(axis=0)
        daily = np.full(total.shape, np.nan)
        np.divide(np.where(valid, values, 0).sum(axis=0), n, out=daily, where=n > 0)
        good = np.isfinite(daily); total += np.where(good, daily, 0); count += good
    model = np.full(total.shape, np.nan)
    np.divide(total, count, out=model, where=count > 0)
    satellite = read_snodas(args, index, lon, lat)
    print(f'{year} {CODES[season]}: model files={len(files)}/{expected}; SNODAS inputs={ndays}/{expected}', flush=True)
    return year, season, model, satellite, expected

def read_seasonal_means(args):
    folder = os.path.join(args.edir, 'postprocess', 'output_dir')
    files = {}
    for filename in sorted(glob.glob(os.path.join(folder, '*.nc'))):
        try: date = datetime.strptime(os.path.basename(filename), '%Y%m%d.nc')
        except ValueError: continue
        if args.start_year <= date.year <= args.end_year: files[date] = filename
    if not files: raise FileNotFoundError('No model daily files within requested years')
    with nc.Dataset(next(iter(files.values()))) as ds:
        native_lon, native_lat = grid(ds.variables['lon'][:]), grid(ds.variables['lat'][:])
    lon, lat = np.sort(native_lon), np.sort(native_lat)
    with nc.Dataset(args.snodas_file) as ds:
        years = np.asarray(ds.variables['year'][:])
        seasons = ds.variables['season'][:]
        days = np.asarray(ds.variables['ndays'][:])
    tasks = []; seen = set()
    for i, (year, code, ndays) in enumerate(zip(years, seasons, days)):
        code = code.decode() if isinstance(code, bytes) else str(code)
        if not args.start_year <= year <= args.end_year: continue
        season = next((s for s,c in CODES.items() if c == code), None)
        if season is None: raise ValueError(f'Unknown season {code}')
        key = (int(year), season)
        if key in seen: raise ValueError(f'Duplicate SNODAS season {key}')
        seen.add(key)
        selected = [f for d,f in files.items() if d.year == year and d.month in SEASONS[season]]
        if selected: tasks.append((args, int(year), season, i, int(ndays), selected, native_lon, native_lat))
    shape = (lat.size, lon.size)
    sums = {k:{s:np.zeros(shape) for s in SEASONS} for k in ('model','satellite')}
    weights = {s:np.zeros(shape) for s in SEASONS}
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        for year, season, model, satellite, days in pool.map(process_season, tasks, chunksize=1):
            valid = np.isfinite(model) & np.isfinite(satellite)
            for key, a in (('model',model), ('satellite',satellite)):
                sums[key][season] += np.where(valid, a*days, 0)
            weights[season] += np.where(valid, days, 0)
    means = {k:{} for k in sums}
    for key in sums:
        for season in SEASONS:
            a = np.full(shape, np.nan)
            np.divide(sums[key][season], weights[season], out=a, where=weights[season] > 0)
            means[key][season] = a
    return lon, lat, means['model'], means['satellite']

def comparison(model, satellite):
    valid = np.isfinite(model) & np.isfinite(satellite)
    model_valid = model[valid]
    satellite_valid = satellite[valid]
    difference = model_valid - satellite_valid
    if difference.size == 0:
        return valid, difference, np.nan, np.nan, np.nan, np.nan
    md = float(np.mean(difference))
    mad = float(np.mean(np.abs(difference)))
    rmse = float(np.sqrt(np.mean(difference ** 2)))
    correlation = np.nan
    if difference.size >= 2 and np.std(model_valid) > 0 and np.std(satellite_valid) > 0:
        correlation = float(np.corrcoef(model_valid, satellite_valid)[0, 1])
    return valid, difference, md, mad, rmse, correlation


def plot_maps(args, lon, lat, model, satellite, output):
    #paired_values = [a[np.isfinite(a)] for data in (model, satellite) for a in data.values()]
    #maximum = max((a.max() for a in paired_values if a.size), default=1.0)
    #swe_norm = colors.Normalize(0.0, max(float(maximum), 1.0))
    swe_norm = colors.Normalize(0.0, 1000.0)
    differences = [comparison(model[season], satellite[season])[1] for season in SEASONS]
    retained = [values for values in differences if values.size]
    if not retained:
        raise RuntimeError("No valid paired SWE pixels")
    actual_min = min(float(values.min()) for values in retained)
    actual_max = max(float(values.max()) for values in retained)
    limit = args.diff_limit if args.diff_limit is not None else max(
        abs(actual_min), abs(actual_max), 1.0e-6)
    print(f"Retained difference range: {actual_min:.6f} to {actual_max:.6f}; "
          f"map color limits: {-limit:.6f} to {limit:.6f}", flush=True)
    difference_norm = colors.TwoSlopeNorm(
        vmin=-limit, vcenter=0.0, vmax=limit
    )
    extend = 'both' if actual_min < -limit and actual_max > limit else 'min' if actual_min < -limit else 'max' if actual_max > limit else 'neither'
    figure, axes = plt.subplots(4, 3, figsize=(15, 19), sharex=True, sharey=True)
    for row, season in enumerate(SEASONS):
        valid, _, _, _, _, _ = comparison(
            model[season], satellite[season]
        )
        model_map = np.where(valid, model[season], np.nan)
        satellite_map = np.where(valid, satellite[season], np.nan)
        difference_map = np.where(valid, model[season] - satellite[season], np.nan)
        panels = (
            (model_map, "HB-NMP 3d SWE", "Blues", swe_norm, "SWE [mm]"),
            (satellite_map, "SNODAS SWE", "Blues", swe_norm, "SWE [mm]"),
            (difference_map, "3d - SNODAS", "RdBu_r", difference_norm, "Difference [mm]"),
        )
        for column, (values, title, cmap, norm, label) in enumerate(panels):
            axis = axes[row, column]
            image = axis.pcolormesh(lon, lat, values, cmap=cmap, norm=norm, shading="auto")
            axis.set_title(f"{season}: {title}")
            axis.set_xlabel("Longitude [°]")
            axis.set_ylabel("Latitude [°]")
            figure.colorbar(image, ax=axis, label=label, shrink=0.86, extend=extend if column == 2 else 'max')
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}", flush=True)

def plot_histograms(args, model, satellite, output):
    figure, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Common 10-mm bins across all seasons.
    all_diff = np.concatenate([
        comparison(model[s], satellite[s])[1] for s in SEASONS
    ])
    if not all_diff.size:
        raise ValueError("No valid paired SWE values")

    bin_edges = np.arange(
        np.floor(all_diff.min() / 10) * 10,
        np.ceil(all_diff.max() / 10) * 10 + 20,
        10
    )

    for axis, season in zip(axes.ravel(), SEASONS):
        _, difference, md, mad, rmse, correlation = comparison(
            model[season], satellite[season]
        )
        if difference.size == 0:
            axis.set_title(f"{season}: no paired pixels", fontweight="bold")
            axis.axis("off")
            continue
        axis.hist(difference, bins=bin_edges, density=True)
        axis.set_xlim(-200, 200)
        axis.set_xticks([-200, -100, 0, 100, 200])
        axis.axvline(md, color="darkred", linestyle="dashed", linewidth=4)
        r_text = f"{correlation:.3f}" if np.isfinite(correlation) else "NA"
        axis.text(
            0.98, 0.98,
            f"MD = {md:.3f}\nMAD = {mad:.3f}\nRMSE = {rmse:.3f}\n"
            f"r = {r_text}\nn = {difference.size:,}",
            transform=axis.transAxes, ha="right", va="top", fontsize=10,
            bbox={"boxstyle": "round", "facecolor": "lightgray",
                  "edgecolor": "black", "alpha": 0.6},
        )
        axis.set_title(season, fontweight="bold")
        axis.set_xlabel("HB-NMP 3D − SNODAS SWE [mm]")
        axis.set_ylabel("Probability density")
    figure.tight_layout()
    figure.savefig(output, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    print(f"Saved: {output}", flush=True)



def main():
    args = arguments()
    os.makedirs(args.outdir, exist_ok=True)
    print(f'Model: {args.edir}\nSNODAS: {args.snodas_file}\nYears: {args.start_year}–{args.end_year}\nSource CRS supplied: {args.snodas_crs}', flush=True)
    lon, lat, model, satellite = read_seasonal_means(args)
    stem = f'SWE_SNODAS_{args.start_year}_{args.end_year}'
    plot_maps(args, lon, lat, model, satellite, os.path.join(args.outdir, f'seasonal_maps_{stem}.png'))
    plot_histograms(args, model, satellite, os.path.join(args.outdir, f'seasonal_histograms_{stem}.png'))

if __name__ == '__main__':
    main()
