#!/usr/bin/env python3
#!/usr/bin/env python3
"""Read-only 3D-minus-PP SWdown polar plots split by approximate local solar time.
Uses original raw-CID HRU matching, pixel slope, and stored 3D HRU aspect.
Requires numpy, netCDF4, rasterio, matplotlib. Run in HB3 on HPC.
"""
import os, glob, csv
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
import numpy as np
import netCDF4 as nc
import rasterio
from rasterio.warp import transform as crs_transform
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

P3 = os.environ.get('HB_PATH_3D', '/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/')
PP = os.environ.get('HB_PATH_PP', '/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test/')
OUT = os.environ.get('HB_OUTDIR', 'morning_afternoon_3d_minus_pp')
SHIFT = float(os.environ.get('HB_TIME_SHIFT_HOURS', '1.5'))  # interval-start MSWX -> interval midpoint
WORKERS = int(os.environ.get('SLURM_CPUS_PER_TASK', '4'))
CHUNK = int(os.environ.get('HB_TIME_CHUNK', '128'))
ALLOW_PARTIAL = os.environ.get('HB_ALLOW_PARTIAL', '0') == '1'
SLOPE_EDGES = np.arange(0, 70, 5)
ASPECT_EDGES = np.arange(0, 360, 22.5)
PERIODS = ('morning', 'afternoon')
N_R, N_A = len(SLOPE_EDGES)-1, len(ASPECT_EDGES)


def grid(fp, signature=None):
    with rasterio.open(fp) as d:
        a = d.read(1, masked=True).astype(float).filled(np.nan)
        sig = (d.shape, d.crs, d.transform)
    a[a == -9999] = np.nan
    if signature is not None and sig != signature:
        raise ValueError(f'Raster mismatch: {fp}')
    return a, sig


def hru_map(path, cid, sig):
    a, _ = grid(os.path.join(path, str(cid), 'hru_mapping_latlon.tif'), sig)
    m, _ = grid(os.path.join(path, 'postprocess', 'cids', f'{cid}.tif'), sig)
    valid = np.isfinite(a) & (m == cid)
    if np.any(valid & ((a < 0) | (a != np.floor(a)))):
        raise ValueError('HRU IDs must be zero-based integers')
    out = np.full(a.shape, -1, dtype=np.int64)
    out[valid] = a[valid].astype(np.int64)
    return out


def longitude_at_cid(sig, common):
    shape, crs, affine = sig
    rr, cc = np.nonzero(common)
    row, col = float(np.mean(rr)), float(np.mean(cc))
    x, y = affine * (col + .5, row + .5)
    if crs is None:
        raise ValueError('Raster CRS missing; cannot infer local solar time')
    if crs.is_geographic:
        return float(x)
    return float(crs_transform(crs, 'EPSG:4326', [x], [y])[0][0])


def get_times(ds):
    v = ds.groups['metadata'].variables['date']
    ts = nc.num2date(v[:], v.units, calendar=getattr(v, 'calendar', 'standard'))
    return [str(t) for t in ts]


def hour_of_day(timestamp):
    # HydroBlocks model dates are UTC. Parse time-of-day without depending on cftime class.
    t = timestamp.split(' ')[-1].split('T')[-1]
    h, m, s = t.split(':')[:3]
    return int(h) + int(m)/60 + float(s)/3600


def clean(var, start, end):
    a = np.ma.asarray(var[start:end], dtype=float).filled(np.nan)
    for attr in ('_FillValue', 'missing_value'):
        for v in np.atleast_1d(getattr(var, attr, [])):
            a[a == v] = np.nan
    a[(a == -9999) | ~np.isfinite(a)] = np.nan
    return a


def process(cid):
    try:
        root = os.path.join(P3, str(cid))
        slope, sig = grid(os.path.join(root, 'slope_latlon.tif'))
        slope = np.degrees(np.arctan(slope))
        h3 = hru_map(P3, cid, sig)
        hp = hru_map(PP, cid, sig)
        common = (h3 >= 0) & (hp >= 0)
        if not common.any():
            raise ValueError('No common pixels')
        longitude = longitude_at_cid(sig, common)
        pairs, inverse = np.unique(np.column_stack((h3[common], hp[common])), axis=0, return_inverse=True)
        with nc.Dataset(os.path.join(root, 'input_file.nc')) as ds:
            raw = np.ma.asarray(ds.groups['parameters'].variables['aspect'][:], dtype=float).filled(np.nan)
        raw[raw == -9999] = np.nan
        angles = np.full(h3.shape, np.nan)
        good = (h3 >= 0) & (h3 < len(raw))
        angles[good] = (180 - np.degrees(raw[h3[good]])) % 360
        a = angles[common]
        s = slope[common]
        ab = np.floor(np.where(np.isfinite(a), a, 0) / 22.5).astype(int) % N_A
        sb = np.searchsorted(SLOPE_EDGES, s, side='right') - 1
        valid = np.isfinite(a) & np.isfinite(s) & (sb >= 0) & (sb < N_R)
        pixel_bins = sb[valid]*N_A + ab[valid]
        pixel_pair = inverse[valid]
        # Pixel counts are independent of temporal availability; counts for diagnostics.
        pixels = np.bincount(pixel_bins, minlength=N_R*N_A).reshape(N_R, N_A)
        sums = np.zeros((2, len(pairs)))
        counts = np.zeros((2, len(pairs)), dtype=np.int64)
        steps = np.zeros(2, dtype=np.int64)
        files3 = sorted(glob.glob(os.path.join(P3, 'output_data', str(cid), '*.nc')))
        filesp = sorted(glob.glob(os.path.join(PP, 'output_data', str(cid), '*.nc')))
        if not files3 or len(files3) != len(filesp):
            raise ValueError('Missing or unequal output file counts')
        previous = None
        for f3, fp in zip(files3, filesp):
            with nc.Dataset(f3) as d3, nc.Dataset(fp) as dp:
                times = get_times(d3)
                if not times or times != get_times(dp):
                    raise ValueError(f'Timestamp mismatch: {f3} vs {fp}')
                if any(x >= y for x,y in zip(times,times[1:])) or (previous is not None and times[0] <= previous):
                    raise ValueError('Non-increasing timestamps')
                previous = times[-1]
                v3 = d3.groups['data'].variables['swdn_3d']
                vp = dp.groups['data'].variables['swdn']
                if max(pairs[:,0]) >= v3.shape[1] or max(pairs[:,1]) >= vp.shape[1]:
                    raise ValueError('HRU index outside output dimension')
                for start in range(0, len(times), CHUNK):
                    end = min(start + CHUNK, len(times))
                    # Mean solar time ~ UTC + longitude/15 + 1.5h for interval midpoint.
                    solar_hour = np.array([(hour_of_day(t) + SHIFT + longitude/15.) % 24 for t in times[start:end]])
                    diff = clean(v3,start,end)[:,pairs[:,0]] - clean(vp,start,end)[:,pairs[:,1]]
                    for p, mask in enumerate(((solar_hour >= 6) & (solar_hour < 12),
                                              (solar_hour >= 12) & (solar_hour < 18))):
                        steps[p] += int(mask.sum())
                        if not mask.any():
                            continue
                        selected = diff[mask]
                        finite = np.isfinite(selected)
                        sums[p] += np.where(finite, selected, 0).sum(axis=0)
                        counts[p] += finite.sum(axis=0)
        # First average time for each geographic HRU pair, then equally weight pixels.
        pair_means = np.full(sums.shape, np.nan)
        np.divide(sums, counts, out=pair_means, where=counts > 0)
        bin_sums = np.zeros((2, N_R, N_A))
        bin_counts = np.zeros((2, N_R, N_A), dtype=np.int64)
        for p in range(2):
            values = pair_means[p,pixel_pair]
            ok = np.isfinite(values)
            bin_sums[p] = np.bincount(pixel_bins[ok], weights=values[ok], minlength=N_R*N_A).reshape(N_R,N_A)
            bin_counts[p] = np.bincount(pixel_bins[ok], minlength=N_R*N_A).reshape(N_R,N_A)
        return cid, (bin_sums, bin_counts, pixels, steps, longitude), None
    except Exception as e:
        return cid, None, f'{type(e).__name__}: {e}'


def plot(means, edges, limit, outfile):
    theta, radius = np.meshgrid(np.deg2rad(np.arange(0,361,22.5)), edges)
    fig, axs = plt.subplots(1, 2, figsize=(11.5,5.2), subplot_kw={'projection':'polar'})
    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('white')
    norm = TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit)
    for ax, vals, title in zip(axs, means, ('Morning (06–12 solar time)', 'Afternoon (12–18 solar time)')):
        mesh = ax.pcolormesh(theta, radius, vals, cmap=cmap, norm=norm, shading='auto')
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_thetagrids(np.arange(0,360,45))
        ax.set_rticks(np.arange(0,edges[-1]+1,5 if edges[-1]==35 else 10))
        ax.set_rlabel_position(157.5)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=22)
        ax.grid(True, alpha=.6)
    fig.subplots_adjust(left=.05,right=.88,wspace=.35)
    bar = fig.colorbar(mesh, ax=axs, shrink=.72, pad=.08, extend='both')
    bar.set_label(r'$\Delta SW_{down}$ (3D − PP), W m$^{-2}$')
    fig.savefig(outfile, dpi=220, bbox_inches='tight')
    plt.close(fig)
    print('Saved', outfile, flush=True)


def main():
    if CHUNK < 1 or WORKERS < 1:
        raise ValueError('CHUNK and WORKERS must be positive')
    os.makedirs(OUT, exist_ok=True)
    c3 = {int(x) for x in os.listdir(P3) if x.isdigit() and os.path.isdir(os.path.join(P3,x))}
    cp = {int(x) for x in os.listdir(PP) if x.isdigit() and os.path.isdir(os.path.join(PP,x))}
    if c3 != cp and not ALLOW_PARTIAL:
        raise ValueError(f'CID sets differ; only 3D: {sorted(c3-cp)}; only PP: {sorted(cp-c3)}')
    cids = sorted(c3 & cp)
    total_s = np.zeros((2,N_R,N_A))
    total_n = np.zeros((2,N_R,N_A),dtype=np.int64)
    total_pixels = np.zeros((N_R,N_A),dtype=np.int64)
    failed = {}
    all_steps = np.zeros(2,dtype=np.int64)
    print(f'3D: {P3}\nPP: {PP}\nCIDs: {len(cids)}; midpoint shift: {SHIFT} h; workers: {WORKERS}',flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as pool:
        for i,(cid,result,error) in enumerate(pool.map(process,cids),1):
            if error:
                failed[cid] = error
                print('FAILED CID',cid,error,flush=True)
            else:
                sums, counts, pixels, steps, lon = result
                total_s += sums
                total_n += counts
                total_pixels += pixels
                all_steps += steps
            if i%25==0 or i==len(cids):
                print(f'{i}/{len(cids)} CIDs processed',flush=True)
    if failed and not ALLOW_PARTIAL:
        raise RuntimeError(f'{len(failed)} CIDs failed. No partial results written. Set HB_ALLOW_PARTIAL=1 to permit.')
    if not total_n.any():
        raise RuntimeError('No valid daypart samples')
    means = np.full(total_s.shape,np.nan)
    np.divide(total_s,total_n,out=means,where=total_n>0)
    for rmax,limit in ((35,50),(65,60)):
        n = rmax//5
        plot(means[:,:n],SLOPE_EDGES[:n+1],limit,os.path.join(OUT,f'morning_afternoon_SWdown_slope0_{rmax}.png'))
    with open(os.path.join(OUT,'morning_afternoon_SWdown_bins.csv'),'w',newline='') as f:
        writer=csv.writer(f)
        writer.writerow(['daypart','slope_low_deg','slope_high_deg','aspect_low_deg','aspect_high_deg','mean_delta_SWdown_Wm2','valid_pixels','all_common_pixels'])
        for p,period in enumerate(PERIODS):
            for r in range(N_R):
                for a in range(N_A):
                    writer.writerow([period,SLOPE_EDGES[r],SLOPE_EDGES[r+1],ASPECT_EDGES[a],ASPECT_EDGES[a]+22.5,
                                     means[p,r,a],total_n[p,r,a],total_pixels[r,a]])
    print('Total CID-timesteps by daypart:',dict(zip(PERIODS,all_steps.tolist())),flush=True)
    print('Failed CIDs:',failed,flush=True)
    print('Saved CSV:',os.path.join(OUT,'morning_afternoon_SWdown_bins.csv'),flush=True)

if __name__=='__main__':
    main()
