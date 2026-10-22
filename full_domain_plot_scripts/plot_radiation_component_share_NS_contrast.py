#!/usr/bin/env python3
"""Full-domain SW/LW magnitude shares and normalized south-minus-north contrast.
Four PNGs: mean and median statistics at slopes [0,35) and [0,65).
Standalone: numpy, netCDF4, rasterio, matplotlib. No pandas/scipy needed.
Inputs are read only; no CSV/JSON/NPZ files written. Tables embedded in PNGs;
unrounded statistics and sample counts printed to the job log.
Stored 3D model aspect; own HRU maps for each run; aligned CID masks required.
Geographic HRU pairs use matching clocks and jointly valid SWnet/LWnet samples,
so each component's seasonal mean uses the same temporal support.
DJF/MAM/JJA/SON pool all years; Annual is the all-month multi-year mean.
Pixel weights preserve HRU mapping multiplicities. Median matches np.median
on the conceptual repeated-pixel values (average middle two for even counts).
Shares are statistics of per-pixel |dSW|/(|dSW|+|dLW|), not ratios of regional
aggregates and not signed contributions to dRnet. Threshold is 1e-6 W m-2.
NS contrast = 100*(South-North)/(|South|+|North|), using mean or median dFlux.
It is bounded [-100,100] and can reach +/-100 for opposite signs even when
absolute differences are small. LWnet response is indirect in these runs.
"""
import os

import glob

from concurrent.futures import ProcessPoolExecutor

import numpy as np

import netCDF4 as nc

import rasterio

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt

# model outputs (can be overridden from the job script with HB_PATH_PP / HB_PATH_3D)

path_pp = os.environ.get("HB_PATH_PP", "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_pp_upper_colorado_11yr_50hru_full_domain_downscaled_test/")

path_3d = os.environ.get("HB_PATH_3D", "/scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/")

path_pp = os.path.join(path_pp, "")      # make sure paths end with "/"

path_3d = os.path.join(path_3d, "")

# output directory (override with HB_OUTDIR)

outputdir = "/projects/battobrah@xsede.org/Full_HB_NOAH_MP_PLOTS/Radiation_component_share_NS_contrast"

os.makedirs(outputdir, exist_ok=True)

# output files to average over (in output_data/<cid>/). "*.nc" = whole simulation;

# use e.g. "2016-01-01.nc" to reproduce the original single-file analysis

OUTPUT_FILES = "*.nc"

# HRU numbering in hru_mapping_latlon.tif. HydroBlocks preprocessing writes

# 0-based HRU ids (hrus = hrus - 1), so pixel value h -> HRU index h.

# (The original notebook used hru > 0 and model[hru - 1], i.e. 1-based.)


# Aspect used for the aspect bins:

#   "hru_xy" : each pixel takes its HRU's aspect, computed as the circular mean

#              atan2(x_aspect, y_aspect) from input_file.nc (correct averaging of

#              angles; this is the HRU orientation implied by the clustering)

#   "pixel"  : each pixel's own 90-m aspect from aspect_latlon.tif


CHUNK = int(os.environ.get("HB_TIME_CHUNK", "128"))

ALLOW_PARTIAL = os.environ.get("HB_ALLOW_PARTIAL", "0") == "1"

# parallel workers

WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))

# variables

# Downward radiation: "swdn"/"lwdn" are the forcing passed into Noah-MP (identical

# in both runs); the terrain-corrected downward fluxes are "swdn_3d"/"lwdn_3d".

# So the 3D run uses the *_3d outputs and the PP run uses the plane-parallel ones.


vars_needed = ["swnet", "lwnet"]


# bins (computed once up to 65 deg; the masked figure uses the first 7 rings)

slope_edges_full = np.arange(0, 70, 5)        # 0..65

aspect_edges = np.arange(0, 360, 22.5)

n_r_full = len(slope_edges_full) - 1

n_t = len(aspect_edges)

# ---------------------------------------------------------------------------

# Per-CID helpers

# ---------------------------------------------------------------------------

def list_cids(path):

    cids = sorted(int(d) for d in os.listdir(path)

                  if d.isdigit() and os.path.isdir(os.path.join(path, d)))

    if not cids:

        raise FileNotFoundError(f"No CID folders in {path}")

    return cids

def read_grid(fp):

    with rasterio.open(fp) as ds:

        arr = ds.read(1, masked=True).astype(float).filled(np.nan)

        signature = (ds.shape, ds.crs, ds.transform)

    arr[arr == -9999] = np.nan

    return arr, signature

def checked_grid(fp, signature):

    arr, got = read_grid(fp)

    if got != signature:

        raise ValueError(f"Raster alignment mismatch: {fp}")

    return arr

def read_cid_hru_map(path, cid, signature):

    arr = checked_grid(os.path.join(path, str(cid), "hru_mapping_latlon.tif"), signature)

    mask = checked_grid(os.path.join(path, "postprocess", "cids", f"{cid}.tif"), signature)

    valid = np.isfinite(arr) & (mask == cid)

    if np.any(valid & ((arr < 0) | (arr != np.floor(arr)))):

        raise ValueError("Invalid zero-based HRU IDs")

    out = np.full(arr.shape, -1, dtype=np.int64)

    out[valid] = arr[valid].astype(np.int64)

    return out

def place_data(idx, model):

    out = np.full(idx.shape, np.nan)

    valid = (idx >= 0) & (idx < model.size)

    out[valid] = model[idx[valid]]

    return out

def clock(ds):

    v = ds.groups["metadata"].variables["date"]

    dates = nc.num2date(v[:], v.units, calendar=getattr(v, "calendar", "standard"))

    return [str(t) for t in dates]

def clean(var, start, end, albedo=False):

    a = np.ma.asarray(var[start:end], dtype=float).filled(np.nan)

    for attr in ("_FillValue", "missing_value"):

        for value in np.atleast_1d(getattr(var, attr, [])):

            a[a == value] = np.nan

    a[(a == -9999) | ~np.isfinite(a)] = np.nan

    if albedo:

        a[(a < 0) | (a > 1)] = np.nan

    return a

SEASONS = ['Winter', 'Spring', 'Summer', 'Fall', 'Annual']
MONTHS = [(12,1,2), (3,4,5), (6,7,8), (9,10,11), tuple(range(1,13))]

def paired_seasonal(cid, pairs):
    files = [sorted(glob.glob(os.path.join(p,'output_data',str(cid),OUTPUT_FILES))) for p in (path_3d,path_pp)]
    if not files[0] or len(files[0]) != len(files[1]):
        raise ValueError('Missing output files or unequal file counts')
    sums = np.zeros((5,3,len(pairs)))
    counts = np.zeros_like(sums, dtype=np.int64)
    previous = None
    nt = 0
    for f3, fp in zip(*files):
        with nc.Dataset(f3) as d3, nc.Dataset(fp) as dp:
            times = clock(d3)
            if times != clock(dp) or not times:
                raise ValueError('Mismatched or empty timestamps')
            if any(a >= b for a,b in zip(times,times[1:])) or (previous is not None and times[0] <= previous):
                raise ValueError('Duplicate/non-increasing timestamps')
            previous = times[-1]
            months = np.array([int(t[5:7]) for t in times])
            groups = [d3.groups['data'],dp.groups['data']]
            for g, ids in zip(groups,pairs.T):
                for key in vars_needed:
                    v = g.variables[key]
                    expected = 2
                    if v.ndim != expected or v.shape[0] != len(times) or ids.max() >= v.shape[1]:
                        raise ValueError(f'Invalid time/HRU/soil dimensions: {key} {v.shape}')
            for start in range(0,len(times),CHUNK):
                end = min(start+CHUNK,len(times))
                data = []
                for g, ids in zip(groups,pairs.T):
                    sw = clean(g.variables['swnet'],start,end)[:,ids]
                    lw = clean(g.variables['lwnet'],start,end)[:,ids]
                    data.append(np.stack((sw,lw,sw-lw),axis=1))
                diff = data[0]-data[1]
                jointly_valid = np.isfinite(diff[:,0]) & np.isfinite(diff[:,1])
                diff = np.where(jointly_valid[:,None,:], diff, np.nan)
                for j, mons in enumerate(MONTHS):
                    a = diff[np.isin(months[start:end],mons)]
                    valid = np.isfinite(a)
                    sums[j] += np.where(valid,a,0).sum(axis=0)
                    counts[j] += valid.sum(axis=0)
            nt += len(times)
    means = np.full_like(sums,np.nan)
    np.divide(sums,counts,out=means,where=counts > 0)
    return means, nt


def process_cid(cid):
    try:
        root = os.path.join(path_3d,str(cid))
        slope, signature = read_grid(os.path.join(root,'slope_latlon.tif'))
        slope = np.degrees(np.arctan(slope))
        h3 = read_cid_hru_map(path_3d,cid,signature)
        hp = read_cid_hru_map(path_pp,cid,signature)
        common = (h3 >= 0) & (hp >= 0)
        if not common.any():
            raise ValueError('No common CID pixels')
        pairs,inverse = np.unique(np.column_stack((h3[common],hp[common])),axis=0,return_inverse=True)
        means, nt = paired_seasonal(cid,pairs)
        with nc.Dataset(os.path.join(root,'input_file.nc')) as ds:
            raw = np.ma.asarray(ds.groups['parameters'].variables['aspect'][:],dtype=float).filled(np.nan)
        if h3[common].max() >= raw.size:
            raise ValueError('Aspect HRU index out of range')
        raw[raw == -9999] = np.nan
        aspect = place_data(h3,(180-np.degrees(raw)) % 360)[common]
        slopes = slope[common]
        sb = np.searchsorted(slope_edges_full,slopes,side='right')-1
        ab = np.floor(np.where(np.isfinite(aspect),aspect,-22.5)/22.5).astype(int)
        valid = np.isfinite(slopes) & np.isfinite(aspect) & (sb >= 0) & (sb < n_r_full)
        sums = np.zeros((5,3,n_r_full,n_t))
        counts = np.zeros_like(sums,dtype=np.int64)
        for j in range(5):
            for k in range(3):
                values = means[j,k,inverse]
                ok = valid & np.isfinite(values)
                bins = sb[ok]*n_t+ab[ok]
                sums[j,k] = np.bincount(bins,weights=values[ok],minlength=n_r_full*n_t).reshape(n_r_full,n_t)
                counts[j,k] = np.bincount(bins,minlength=n_r_full*n_t).reshape(n_r_full,n_t)
        north = (aspect <= 90) | (aspect >= 270)
        weights = np.zeros((2,2,len(pairs)),dtype=np.int64)
        for r,limit in enumerate((35,65)):
            domain = valid & (slopes < limit)
            for d, facing in enumerate((north,~north)):
                weights[r,d] = np.bincount(inverse[domain & facing],minlength=len(pairs))
        coverage = dict(pixels=int(common.sum()), outside65=int(np.count_nonzero(~np.isfinite(slopes)|(slopes < 0)|(slopes >= 65))),
                        outside35=int(np.count_nonzero(~np.isfinite(slopes)|(slopes < 0)|(slopes >= 35))),
                        undefined_aspect=int(np.count_nonzero(~np.isfinite(aspect))),timesteps=nt)
        return cid,sums,counts,means[:,:3],weights,coverage,None
    except Exception as e:
        return cid,None,None,None,None,None,f'{type(e).__name__}: {e}'


EPS = 1e-6


def summarize(values,weights,mode):
    good = np.isfinite(values) & (weights > 0)
    v,w = values[good],weights[good]
    if not v.size: return np.nan
    if mode == 'MEAN': return float(np.average(v,weights=w))
    order = np.argsort(v);v,w = v[order],w[order]
    cumulative = np.cumsum(w,dtype=np.int64);total = int(cumulative[-1])
    ranks = [(total-1)//2,total//2]
    return float(np.mean(v[np.searchsorted(cumulative,ranks,side='right')]))


def calculate_table(groups,range_index,mode):
    rows=[]
    for j,season in enumerate(SEASONS):
        results=[];regional=[]
        for d in range(2):
            chunks = groups[(range_index,d,j)]
            if not chunks:
                results.append((np.nan,np.nan));regional.append((np.nan,np.nan));continue
            sw = np.concatenate([x[0] for x in chunks])
            lw = np.concatenate([x[1] for x in chunks])
            weights = np.concatenate([x[2] for x in chunks])
            valid = np.isfinite(sw)&np.isfinite(lw)&(weights > 0)
            denom = np.abs(sw)+np.abs(lw)
            eligible = valid & (denom > EPS)
            pct_sw = np.full(sw.shape,np.nan);pct_lw = np.full(lw.shape,np.nan)
            np.divide(100*np.abs(sw),denom,out=pct_sw,where=eligible)
            np.divide(100*np.abs(lw),denom,out=pct_lw,where=eligible)
            results.append((summarize(pct_sw,weights,mode),summarize(pct_lw,weights,mode)))
            regional.append((summarize(sw[valid],weights[valid],mode),summarize(lw[valid],weights[valid],mode)))
            print(f"{mode} | slopes<{(35,65)[range_index]} | {season} | {('North','South')[d]} | "
                  f"SW share={results[-1][0]:.6f}% | LW share={results[-1][1]:.6f}% | "
                  f"dSW={regional[-1][0]:.6f} W m^-2 | dLW={regional[-1][1]:.6f} W m^-2 | "
                  f"valid pixels={int(weights[valid].sum())} | share pixels={int(weights[eligible].sum())}",flush=True)
        contrasts=[]
        for k in range(2):
            north,south = regional[0][k],regional[1][k]
            den = abs(south)+abs(north)
            contrasts.append(100*(south-north)/den if np.isfinite(den) and den > EPS else np.nan)
        row = [results[0][0],results[1][0],results[0][1],results[1][1],*contrasts]
        rows.append(row)
        print(f"{mode} | slopes<{(35,65)[range_index]} | {season} | NS_SW={contrasts[0]:.6f}% | NS_LW={contrasts[1]:.6f}%",flush=True)
    return np.array(rows)


def plot_table(values,mode,slope_limit,outfile):
    fig = plt.figure(figsize=(13,7.8),facecolor='white')
    gs = fig.add_gridspec(2,2,height_ratios=[3,1.5],hspace=.55,wspace=.25,left=.08,right=.96,top=.88,bottom=.10)
    ax0 = fig.add_subplot(gs[0,0]);ax1 = fig.add_subplot(gs[0,1])
    x = np.arange(5);width=.18
    ax0.tick_params(labelsize=11);ax1.tick_params(labelsize=11)
    palettes=['#1f77b4','#d95f02','#9ecae1','#fdae6b']
    for k,label in enumerate(['SW North','SW South','LW North','LW South']):
        ax0.bar(x+(k-1.5)*width,values[:,k],width=width,label=label,color=palettes[k])
    ax0.axhline(50,linestyle='--',linewidth=.8,color='gray',alpha=.6)
    ax0.set_ylim(0,100);ax0.set_ylabel('Absolute-change component share (%)',fontsize=12)
    ax0.set_title(f'(a) Component shares ({mode.lower()})',fontweight='bold',fontsize=14)
    ax0.legend(frameon=False,fontsize=10,ncol=2,loc='upper right')
    for k,label in enumerate(['SWnet','LWnet']):
        ax1.bar(x+(k-.5)*width,values[:,4+k],width=width,label=label,color=palettes[k])
    ax1.axhline(0,color='black',linewidth=.8)
    ax1.set_ylim(-105,105);ax1.set_yticks([-100,-50,0,50,100])
    ax1.set_ylabel('Normalized South − North contrast (%)',fontsize=12)
    ax1.set_title(f'(b) Aspect contrast ({mode.lower()})',fontweight='bold',fontsize=14)
    ax1.legend(frameon=False,fontsize=10)
    for ax in (ax0,ax1):
        ax.set_xticks(x);ax.set_xticklabels(SEASONS,rotation=30,ha='right')
        ax.grid(True,axis='y',alpha=.25);ax.set_axisbelow(True)
    at = fig.add_subplot(gs[1,:]);at.axis('off')
    labels=['Season','SW North (%)','SW South (%)','LW North (%)','LW South (%)','NS SW (%)','NS LW (%)']
    cells=[[season]+[f'{v:.1f}' if np.isfinite(v) else 'NA' for v in row] for season,row in zip(SEASONS,values)]
    table = at.table(cellText=cells,colLabels=labels,cellLoc='center',loc='center')
    table.auto_set_font_size(False);table.set_fontsize(10.5);table.scale(1,1.6)
    for (r,c),cell in table.get_celld().items():
        cell.set_edgecolor('0.75');cell.set_linewidth(.5)
        if r == 0: cell.set_facecolor('0.92');cell.set_text_props(weight='bold')
    fig.suptitle(f'3D − PP: radiation component shares and aspect contrast | slopes <{slope_limit}°',fontsize=14,fontweight='bold')
    fig.text(.5,.025,'Shares use absolute changes; NS contrast uses signed changes. Annual pools all months and years.',ha='center',fontsize=10)
    fig.savefig(outfile,dpi=300,bbox_inches='tight',facecolor='white');plt.close(fig)
    print('Saved:',outfile,flush=True)


def main():
    if CHUNK < 1 or WORKERS < 1: raise ValueError('Chunk/workers must be positive')
    c3,cp = set(list_cids(path_3d)),set(list_cids(path_pp))
    missing = dict(missing_pp=sorted(c3-cp),missing_3d=sorted(cp-c3))
    if any(missing.values()) and not ALLOW_PARTIAL: raise ValueError(f'CID sets differ: {missing}')
    cids = sorted(c3&cp)
    if not cids: raise ValueError('No common CIDs')
    groups = {(r,d,j):[] for r in range(2) for d in range(2) for j in range(5)}
    coverage=[];failed=[]
    print(f'3D: {path_3d}\nPP: {path_pp}\nOutput: {outputdir}\nCIDs: {len(cids)}, workers: {WORKERS}',flush=True)
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        for i,(cid,s,c,means,weights,cov,err) in enumerate(ex.map(process_cid,cids,chunksize=1),1):
            if err:
                failed.append(cid);print(f'CID {cid} failed: {err}',flush=True)
            else:
                coverage.append(cov)
                for r in range(2):
                    for d in range(2):
                        for j in range(5):
                            valid = (weights[r,d] > 0)&np.isfinite(means[j,0])&np.isfinite(means[j,1])
                            if valid.any(): groups[(r,d,j)].append((means[j,0,valid],means[j,1,valid],weights[r,d,valid]))
            if i % 25 == 0 or i == len(cids): print(f'Processed {i}/{len(cids)}',flush=True)
    print('Missing counterparts:',missing,'Failed:',failed,flush=True)
    print('Coverage:',{k:sum(x[k] for x in coverage) for k in ('pixels','outside35','outside65','undefined_aspect')},flush=True)
    if not coverage or (failed and not ALLOW_PARTIAL): raise RuntimeError('Incomplete/empty domain; see job log')
    suffix = '_PARTIAL' if failed or any(missing.values()) else ''
    for r,limit in enumerate((35,65)):
        for mode in ('MEAN','MEDIAN'):
            values = calculate_table(groups,r,mode)
            if not np.isfinite(values).any(): raise RuntimeError('No finite statistics')
            plot_table(values,mode,limit,os.path.join(outputdir,f'radiation_component_share_NS_contrast_{mode}_slope0_{limit}{suffix}.png'))
    print(f'CIDs used: {len(coverage)}; saved four PNGs with embedded tables.',flush=True)


if __name__ == '__main__': main()
