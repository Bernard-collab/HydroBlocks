#!/usr/bin/env python3
"""Seasonal 3D − PP radiation and surface soil moisture."""
import os
import glob
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import netCDF4 as nc
import rasterio
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.colors as colors
path_pp = os.environ.get('HB_PATH_PP', '/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test/')
path_3d = os.environ.get('HB_PATH_3D', '/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/')
outputdir = os.environ.get('HB_OUTDIR', '/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS_midpoint/Polar_seasonal_Rnet_SMC_aspect')
os.makedirs(outputdir, exist_ok=True)
OUTPUT_FILES = '*.nc'
START_YEAR = int(os.environ.get('HB_START_YEAR', '2014'))
END_YEAR = int(os.environ.get('HB_END_YEAR', '2024'))
CHUNK = int(os.environ.get('HB_TIME_CHUNK', '128'))
ALLOW_PARTIAL = os.environ.get('HB_ALLOW_PARTIAL', '0') == '1'
WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
vars_needed = ['swnet', 'lwnet', 'smc']
# Slope and aspect bins (degrees).
slope_edges_full = np.arange(0, 70, 5)
aspect_edges = np.arange(0, 360, 22.5)
n_r_full = len(slope_edges_full) - 1
n_t = len(aspect_edges)

def list_cids(path):
    cids = sorted((int(d) for d in os.listdir(path) if d.isdigit() and os.path.isdir(os.path.join(path, d))))
    if not cids: raise FileNotFoundError(f'No CID folders in {path}')
    return cids

def read_grid(fp):
    with rasterio.open(fp) as ds:
        arr = ds.read(1, masked=True).astype(float).filled(np.nan)
        signature = (ds.shape, ds.crs, ds.transform)
    arr[arr == -9999] = np.nan
    return (arr, signature)

def checked_grid(fp, signature):
    arr, got = read_grid(fp)
    if got != signature: raise ValueError(f'Raster alignment mismatch: {fp}')
    return arr

def read_cid_hru_map(path, cid, signature):
    arr = checked_grid(os.path.join(path, str(cid), 'hru_mapping_latlon.tif'), signature)
    mask = checked_grid(os.path.join(path, 'postprocess', 'cids', f'{cid}.tif'), signature)
    valid = np.isfinite(arr) & (mask == cid)
    if np.any(valid & ((arr < 0) | (arr != np.floor(arr)))): raise ValueError('Invalid zero-based HRU IDs')
    out = np.full(arr.shape, -1, dtype=np.int64)
    out[valid] = arr[valid].astype(np.int64)
    return out

def place_data(idx, model):
    out = np.full(idx.shape, np.nan)
    valid = (idx >= 0) & (idx < model.size)
    out[valid] = model[idx[valid]]
    return out

def clock(ds):
    v = ds.groups['metadata'].variables['date']
    dates = nc.num2date(v[:], v.units, calendar=getattr(v, 'calendar', 'standard'))
    return [str(t) for t in dates]

# Mask missing radiation values.
def clean(var, start, end):
    a = np.ma.asarray(var[start:end], dtype=float).filled(np.nan)
    for attr in ('_FillValue', 'missing_value'):
        for value in np.atleast_1d(getattr(var, attr, [])):
            a[a == value] = np.nan
    a[(a == -9999) | ~np.isfinite(a)] = np.nan
    return a
SEASONS = ['Winter', 'Spring', 'Summer', 'Fall', 'Annual']
MONTHS = [(12, 1, 2), (3, 4, 5), (6, 7, 8), (9, 10, 11), tuple(range(1, 13))]
KEYS = ['SWnet', 'LWnet', 'Rnet', 'SMC']

def paired_seasonal(cid, pairs):
    files = [sorted(glob.glob(os.path.join(p, 'output_data', str(cid), OUTPUT_FILES))) for p in (path_3d, path_pp)]
    if not files[0] or len(files[0]) != len(files[1]): raise ValueError('Missing output files or unequal file counts')
    sums = np.zeros((5, 4, len(pairs)))
    counts = np.zeros_like(sums, dtype=np.int64)
    previous = None
    nt = 0
    for f3, fp in zip(*files):
        with nc.Dataset(f3) as d3, nc.Dataset(fp) as dp:
            times = clock(d3)
            if times != clock(dp) or not times: raise ValueError('Mismatched or empty timestamps')
            if any((a >= b for a, b in zip(times, times[1:]))) or (previous is not None and times[0] <= previous):
                raise ValueError('Duplicate/non-increasing timestamps')
            previous = times[-1]
            years = np.array([int(t[:4]) for t in times])
            selected = (years >= START_YEAR) & (years <= END_YEAR)
            if not selected.any(): continue
            months = np.array([int(t[5:7]) for t in times])
            groups = [d3.groups['data'], dp.groups['data']]
            for g, ids in zip(groups, pairs.T):
                for key in vars_needed:
                    v = g.variables[key]
                    expected = 3 if key == 'smc' else 2
                    if v.ndim != expected or v.shape[0] != len(times) or ids.max() >= v.shape[1]:
                        raise ValueError(f'Invalid time/HRU/soil dimensions: {key} {v.shape}')
                    if key == 'smc' and v.shape[2] < 1: raise ValueError('SMC has no first soil layer')
            for start in range(0, len(times), CHUNK):
                end = min(start + CHUNK, len(times))
                keep = selected[start:end]
                if not keep.any(): continue
                data = []
                for g, ids in zip(groups, pairs.T):
                    sw = clean(g.variables['swnet'], start, end)[:, ids]
                    lw = clean(g.variables['lwnet'], start, end)[:, ids]
                    v = g.variables['smc']
                    smc = np.ma.asarray(v[start:end, :, 0], dtype=float).filled(np.nan)
                    for attr in ('_FillValue', 'missing_value'):
                        for fill in np.atleast_1d(getattr(v, attr, [])):
                            smc[smc == fill] = np.nan
                    smc[(smc < 0) | (smc > 1) | ~np.isfinite(smc)] = np.nan
                    # Net longwave is upward positive.
                    data.append(np.stack((sw, lw, sw - lw, smc[:, ids]), axis=1))
                diff = (data[0] - data[1])[keep]
                chunk_months = months[start:end][keep]
                for j, mons in enumerate(MONTHS):
                    a = diff[np.isin(chunk_months, mons)]
                    valid = np.isfinite(a)
                    sums[j] += np.where(valid, a, 0).sum(axis=0)
                    counts[j] += valid.sum(axis=0)
            nt += int(selected.sum())
    means = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    return (means, nt)

def process_cid(cid):
    try:
        root = os.path.join(path_3d, str(cid))
        slope, signature = read_grid(os.path.join(root, 'slope_latlon.tif'))
        slope = np.degrees(np.arctan(slope))
        h3 = read_cid_hru_map(path_3d, cid, signature)
        hp = read_cid_hru_map(path_pp, cid, signature)
        common = (h3 >= 0) & (hp >= 0)
        if not common.any(): raise ValueError('No common CID pixels')
        # Match HRUs through their common pixels.
        pairs, inverse = np.unique(np.column_stack((h3[common], hp[common])), axis=0, return_inverse=True)
        means, nt = paired_seasonal(cid, pairs)
        with nc.Dataset(os.path.join(root, 'input_file.nc')) as ds:
            raw = np.ma.asarray(ds.groups['parameters'].variables['aspect'][:], dtype=float).filled(np.nan)
        if h3[common].max() >= raw.size: raise ValueError('Aspect HRU index out of range')
        raw[raw == -9999] = np.nan
        aspect = place_data(h3, (180 - np.degrees(raw)) % 360)[common]
        slopes = slope[common]
        sb = np.searchsorted(slope_edges_full, slopes, side='right') - 1
        ab = np.floor(np.where(np.isfinite(aspect), aspect, -22.5) / 22.5).astype(int)
        valid = np.isfinite(slopes) & np.isfinite(aspect) & (sb >= 0) & (sb < n_r_full)
        sums = np.zeros((5, 4, n_r_full, n_t))
        counts = np.zeros_like(sums, dtype=np.int64)
        for j in range(5):
            for k in range(4):
                values = means[j, k, inverse]
                ok = valid & np.isfinite(values)
                bins = sb[ok] * n_t + ab[ok]
                sums[j, k] = np.bincount(bins, weights=values[ok], minlength=n_r_full * n_t).reshape(n_r_full, n_t)
                counts[j, k] = np.bincount(bins, minlength=n_r_full * n_t).reshape(n_r_full, n_t)
        north = (aspect <= 90) | (aspect >= 270)
        weights = np.zeros((2, 2, len(pairs)), dtype=np.int64)
        for r, limit in enumerate((45, 65)):
            domain = valid & (slopes < limit)
            for d, facing in enumerate((north, ~north)):
                weights[r, d] = np.bincount(inverse[domain & facing], minlength=len(pairs))
        coverage = dict(pixels=int(common.sum()), outside65=int(np.count_nonzero(~np.isfinite(slopes) | (slopes < 0) | (slopes >= 65))), outside45=int(np.count_nonzero(~np.isfinite(slopes) | (slopes < 0) | (slopes >= 45))), undefined_aspect=int(np.count_nonzero(~np.isfinite(aspect))), timesteps=nt)
        return (cid, sums, counts, means[:, :3], weights, coverage, None)
    except Exception as e:
        return (cid, None, None, None, None, None, f'{type(e).__name__}: {e}')

def extension(values, limit):
    low, high = (np.any(values < -limit), np.any(values > limit))
    return 'both' if low and high else 'min' if low else 'max' if high else 'neither'

def polar_axis(ax, values, limit, slope_limit, title):
    theta, radius = np.meshgrid(np.deg2rad(np.append(aspect_edges, 360)), np.arange(0, slope_limit + 1, 5))
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('white')
    pcm = ax.pcolormesh(theta, radius, values, cmap=cmap, norm=colors.TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit), shading='auto')
    ax.set_theta_zero_location('N')
    ax.set_theta_direction(-1)
    ax.set_ylim(0, slope_limit)
    ticks = np.arange(0, slope_limit + 1, 5 if slope_limit == 45 else 10)
    ax.set_rticks(ticks)
    ax.set_yticklabels([f'{t}°' for t in ticks], fontsize=7)
    ax.set_rlabel_position(157.5)
    ax.set_thetagrids(np.arange(0, 360, 45), fontsize=7)
    ax.grid(True, linewidth=0.7, alpha=0.8)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=15)
    return pcm

def plot_seasons(means, key_index, slope_limit, outfile):
    n = slope_limit // 5
    values = means[:, key_index, :n]
    limit = (65.0 if slope_limit == 45 else 80.0) if key_index == 2 else 0.10 if slope_limit == 45 else 0.12
    label = '$\\Delta R_{net}$ / W m$^{-2}$' if key_index == 2 else '$\\Delta SMC_{layer 1}$ / m$^3$ m$^{-3}$'
    fig = plt.figure(figsize=(12, 8), facecolor='white')
    column_left = [0.065, 0.36, 0.655]
    width = 0.235
    height = width * 12 / 8
    bottom_y = 0.075
    top_y = 0.55
    for j, name in enumerate(SEASONS):
        ax = fig.add_axes([column_left[j % 3], top_y if j < 3 else bottom_y, width, height], projection='polar')
        pcm = polar_axis(ax, values[j], limit, slope_limit, f'({chr(97 + j)}) {name}')
    cax = fig.add_axes([column_left[1] + width + 0.05, bottom_y + 0.045, 0.02, height - 0.09])
    bar = fig.colorbar(pcm, cax=cax, extend=extension(values, limit))
    bar.set_label(label, fontsize=11)
    bar.set_ticks(np.linspace(-limit, limit, 5))
    finite = values[np.isfinite(values)]
    if finite.size:
        print(f'{KEYS[key_index]}, slopes 0–{slope_limit}: bin range {finite.min():.5f} to {finite.max():.5f}; below/above scale: {np.count_nonzero(values < -limit)}/{np.count_nonzero(values > limit)}', flush=True)
    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', outfile, flush=True)

def weighted_stats(values, weights):
    order = np.argsort(values)
    v = values[order]
    w = weights[order]
    cumulative = np.cumsum(w)
    total = cumulative[-1]
    # Pixel-weighted quartiles.
    quant = v[np.searchsorted(cumulative, np.array([0.25, 0.5, 0.75]) * total, side='left')]
    q1, med, q3 = quant
    iqr = q3 - q1
    inside = v[(v >= q1 - 1.5 * iqr) & (v <= q3 + 1.5 * iqr)]
    return dict(q1=q1, med=med, q3=q3, whislo=inside[0], whishi=inside[-1], fliers=[], mean=np.average(v, weights=w))

def draw_distribution(ax, values, weights, position, color):
    from scipy.ndimage import gaussian_filter1d
    ok = np.isfinite(values) & (weights > 0)
    v, w = (values[ok], weights[ok])
    if not v.size: return
    stats = weighted_stats(v, w)
    if v.min() < v.max():
        # Weighted histogram for the violin.
        hist, edges = np.histogram(v, bins=256, weights=w)
        density = gaussian_filter1d(hist.astype(float), 1.5, mode='nearest')
        centers = (edges[:-1] + edges[1:]) / 2
        width = density / density.max() * 0.14
        ax.fill_betweenx(centers, position - width, position + width, facecolor=color, edgecolor='black', alpha=0.55, linewidth=0.5)
    ax.bxp([stats], positions=[position], widths=0.1, showfliers=False, patch_artist=True, boxprops=dict(facecolor='white', edgecolor='black', linewidth=0.8), medianprops=dict(color='black'), whiskerprops=dict(color='black', linewidth=0.8), capprops=dict(color='black', linewidth=0.8), manage_ticks=False)
    ax.plot(position, stats['mean'], 'o', color=color, markeredgecolor='black', markersize=3, zorder=5)

def plot_aspect(means, distributions, slope_limit, outfile):
    from matplotlib.patches import Patch
    fig = plt.figure(figsize=(13, 8.2), facecolor='white')
    column_left = [0.065, 0.35, 0.635]
    panel_width = 0.235
    polar_height = panel_width * 13 / 8.2
    n = slope_limit // 5
    values = means[4, :3, :n]
    limit = 65.0 if slope_limit == 45 else 80.0
    for k in range(3):
        ax = fig.add_axes([column_left[k], 0.495, panel_width, polar_height], projection='polar')
        pcm = polar_axis(ax, values[k], limit, slope_limit, f'({chr(97 + k)}) Δ{KEYS[k]}')
    cax = fig.add_axes([0.91, 0.525, 0.016, 0.3])
    bar = fig.colorbar(pcm, cax=cax, extend=extension(values, limit))
    bar.set_label('W m$^{-2}$')
    r = 0 if slope_limit == 45 else 1
    for k in range(3):
        ax = fig.add_axes([column_left[k], 0.14, panel_width, 0.285])
        for j in range(5):
            for d, color in enumerate(('#1f77b4', '#d95f02')):
                chunks = distributions[r, d, j, k]
                if chunks:
                    v = np.concatenate([x[0] for x in chunks])
                    w = np.concatenate([x[1] for x in chunks])
                    draw_distribution(ax, v, w, j + (-0.17 if d == 0 else 0.17), color)
                    valid = np.isfinite(v) & (w > 0)
                    stats = weighted_stats(v[valid], w[valid])
                    sd = np.sqrt(np.average((v[valid] - stats['mean']) ** 2, weights=w[valid]))
                    facing = 'North' if d == 0 else 'South'
                    print(f"STATS | slope<{slope_limit}° | Δ{KEYS[k]} | {SEASONS[j]} | {facing} | mean={stats['mean']:.6f} | median={stats['med']:.6f} | SD={sd:.6f} | Q1={stats['q1']:.6f} | Q3={stats['q3']:.6f} | pixels={int(w[valid].sum())} | units=W m^-2", flush=True)
                    if j == 4:
                        label = f"Annual {facing}\nMean = {stats['mean']:+.2f}\nMedian = {stats['med']:+.2f}"
                        ax.text(0.03 if d == 0 else 0.97, 0.97, label, transform=ax.transAxes, ha='left' if d == 0 else 'right', va='top', fontsize=7.5, color=color, zorder=10, bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=2))
        ax.axhline(0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xticks(range(5))
        ax.set_xticklabels(SEASONS, rotation=35, ha='right')
        ax.set_title(f'({chr(100 + k)}) Δ{KEYS[k]}', fontsize=12, fontweight='bold', loc='left')
        if k == 0:
            ax.set_ylabel('Δ radiation (W m$^{-2}$)')
        ax.grid(True, axis='y', alpha=0.25)
    fig.legend([Patch(facecolor='#1f77b4', alpha=0.55), Patch(facecolor='#d95f02', alpha=0.55)], ['North-facing', 'South-facing'], loc='lower center', ncol=2, frameon=False, bbox_to_anchor=(0.5, 0.015))
    fig.savefig(outfile, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close(fig)
    print('Saved:', outfile, flush=True)

def main():
    if END_YEAR < START_YEAR: raise ValueError('HB_END_YEAR must not precede HB_START_YEAR')
    if CHUNK < 1 or WORKERS < 1: raise ValueError('Chunk/workers must be positive')
    c3, cp = (set(list_cids(path_3d)), set(list_cids(path_pp)))
    missing = dict(missing_pp=sorted(c3 - cp), missing_3d=sorted(cp - c3))
    if any(missing.values()) and (not ALLOW_PARTIAL): raise ValueError(f'CID sets differ: {missing}')
    cids = sorted(c3 & cp)
    if not cids: raise ValueError('No common CIDs')
    sums = np.zeros((5, 4, n_r_full, n_t))
    counts = np.zeros_like(sums, dtype=np.int64)
    distributions = {(r, d, j, k): [] for r in range(2) for d in range(2) for j in range(5) for k in range(3)}
    failed = []
    coverage = []
    print(f'Years: {START_YEAR}–{END_YEAR}', flush=True)
    print(f'3D: {path_3d}\nPP: {path_pp}\nOutput: {outputdir}\nCIDs: {len(cids)}; workers: {WORKERS}', flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i, (cid, s, c, m, w, cov, err) in enumerate(ex.map(process_cid, cids, chunksize=1), 1):
            if err:
                failed.append(cid)
                print(f'CID {cid} failed: {err}', flush=True)
            else:
                sums += s
                counts += c
                coverage.append(cov)
                for r in range(2):
                    for d in range(2):
                        for j in range(5):
                            for k in range(3):
                                ok = (w[r, d] > 0) & np.isfinite(m[j, k])
                                if ok.any():
                                    distributions[r, d, j, k].append((m[j, k, ok], w[r, d, ok]))
            if i % 25 == 0 or i == len(cids):
                print(f'Processed {i}/{len(cids)}', flush=True)
    print('Missing counterparts:', missing, 'Failed CIDs:', failed, flush=True)
    print('Coverage totals:', {key: sum((x[key] for x in coverage)) for key in ('pixels', 'outside45', 'outside65', 'undefined_aspect')}, flush=True)
    if not coverage or (failed and (not ALLOW_PARTIAL)): raise RuntimeError('Incomplete/empty domain; inspect job log')
    means = np.full_like(sums, np.nan)
    np.divide(sums, counts, out=means, where=counts > 0)
    suffix = f'_{START_YEAR}_{END_YEAR}' + ('_PARTIAL' if failed or any(missing.values()) else '')
    for limit in (45, 65):
        for k, kind in ((2, 'Rnet'), (3, 'SMC_layer1')):
            if not np.isfinite(means[:, k, :limit // 5]).any(): raise RuntimeError(f'No finite {kind} bins')
            plot_seasons(means, k, limit, os.path.join(outputdir, f'polar_seasonal_annual_{kind}_slope0_{limit}_3D_minus_PP{suffix}.png'))
        plot_aspect(means, distributions, limit, os.path.join(outputdir, f'polar_box_violin_radiation_slope0_{limit}_3D_minus_PP{suffix}.png'))
    print(f'CIDs used: {len(coverage)}; Annual is the multi-year all-month mean. Boxes: median/IQR; colored dots: weighted mean. Annual mean/median annotations in W m^-2; all seasonal statistics printed above.', flush=True)
if __name__ == '__main__':
    main()
