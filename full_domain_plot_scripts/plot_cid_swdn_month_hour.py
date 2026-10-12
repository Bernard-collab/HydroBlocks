"""Plot raw CID SWDN 3D-minus-PP monthly/hour means and mean azimuth.

Reads one multi-year NetCDF file per CID from each experiment's output_data.
Only the selected time slices are read. Merges zero-based HRU values onto the
full-domain HRU/CID raster. Produces one 4x2 difference figure and metrics CSV.
Year selection uses timestamps converted from UTC by the supplied fixed offset.
Azimuth is circularly averaged over valid paired HRU/time samples, with equal
weight per sample. Identical HRU ordering in both experiments is required.
Requires numpy, pandas, matplotlib, netCDF4, and geospatialtools.
"""
import os
import argparse
import pandas as pd
from types import SimpleNamespace
import matplotlib
matplotlib.use("Agg")
import csv
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm


def _read_selected(variable, indices):
    # Raw example uses (time, HRU). Index before reading to avoid a full year.
    if variable.ndim != 2:
        raise ValueError(f'{variable.name} must be (time, HRU); got {variable.dimensions}')
    return np.ma.asarray(variable[indices, :], dtype=float).filled(np.nan)


def _angle_mean(sine, cosine, count):
    if not count or np.hypot(sine, cosine) <= 1e-12 * count:
        return np.nan
    result = np.degrees(np.arctan2(sine, cosine)) % 360
    return 0.0 if np.isclose(result, 360, rtol=0, atol=1e-10) else result


def plot_raw_cid_swdn(path_3d, path_pp, times, hrus, cids,
                      outputdir, *, cid_list=range(1, 1601),
                      filename='2016-01-01.nc',
                      months=(12, 7, 12, 7, 12, 7, 12, 7),
                      hours=(8, 8, 11, 11, 14, 14, 17, 17),
                      include_next_hour=True, buffer=0, dpi=300,
                      azimuth_units='auto', coords_x=None, coords_y=None,
                      start_year=None, end_year=None):
    """Create one 4x2 difference figure, merging all selected CIDs.

    times: xarray DataArray with .dt.hour/.dt.month, already in local time.
    hrus and cids: co-registered full-domain raster arrays.
    Azimuth title is a circular mean over valid paired HRU/time samples,
    equally weighted per HRU/time sample, not by HRU area.
    Missing files fail explicitly, rather than producing an incomplete domain.
    """
    if len(months) != 8 or len(hours) != 8:
        raise ValueError('This 4x2 layout requires eight month/hour selections.')
    if np.shape(hrus) != np.shape(cids) or np.ndim(hrus) != 2:
        raise ValueError('hrus and cids must be co-registered 2D rasters.')
    if buffer < 0 or 2*buffer >= min(np.shape(hrus)):
        raise ValueError('Invalid raster buffer.')
    if azimuth_units not in ('auto', 'radians', 'degrees'):
        raise ValueError('azimuth_units must be auto, radians, or degrees.')
    local_month = np.asarray(times.dt.month)
    local_hour = np.asarray(times.dt.hour)
    if local_month.ndim != 1:
        raise ValueError('times must be one dimensional.')
    year_mask=np.ones(local_month.shape,dtype=bool)
    if start_year is not None or end_year is not None:
        local_year=np.asarray(times.dt.year)
        if start_year is not None: year_mask &= local_year >= start_year
        if end_year is not None: year_mask &= local_year <= end_year
    selections = []
    for month, hour in zip(months, hours):
        if not 1 <= month <= 12 or not 0 <= hour <= 23:
            raise ValueError('Invalid month/hour.')
        if include_next_hour and hour == 23:
            raise ValueError('Two-hour windows spanning midnight need explicit date handling.')
        mask = year_mask & (local_month == month) & ((local_hour == hour) |
                    ((local_hour == hour+1) if include_next_hour else False))
        indices = np.flatnonzero(mask)
        if not indices.size:
            raise ValueError(f'No samples for month {month}, hour {hour}; check local times/cadence.')
        selections.append(indices)
    union = np.unique(np.concatenate(selections))
    positions = [np.searchsorted(union, idx) for idx in selections]
    cid_list = list(cid_list)
    if not cid_list or len(set(cid_list)) != len(cid_list):
        raise ValueError('cid_list must contain unique CID numbers.')
    # Check all paths first so a missing CID fails before expensive reading.
    valid_pixels=np.isfinite(hrus)&np.isfinite(cids)&(hrus >= 0)
    flat=np.flatnonzero(valid_pixels)
    order=np.argsort(cids.ravel()[flat],kind='stable')
    sorted_pixels=flat[order]
    sorted_cids=cids.ravel()[sorted_pixels]
    placement={}
    records = []
    for cid in cid_list:
        pair = [os.path.join(p, 'output_data', str(cid), filename) for p in (path_3d,path_pp)]
        for file in pair:
            if not os.path.isfile(file):
                raise FileNotFoundError(file)
        lo,hi=np.searchsorted(sorted_cids,cid,side='left'),np.searchsorted(sorted_cids,cid,side='right')
        pixels=sorted_pixels[lo:hi]
        if not pixels.size:
            raise ValueError(f'CID {cid} is absent from the supplied mapping raster.')
        hru_values=hrus.ravel()[pixels]
        if not np.all(hru_values == np.floor(hru_values)):
            raise ValueError(f'CID {cid}: HRU raster has noninteger indices.')
        placement[cid]=(pixels,hru_values.astype(np.int64))
        records.append((cid,*pair))
    differences = [np.full(np.shape(hrus), np.nan) for _ in selections]
    sine,cosine = np.zeros(8),np.zeros(8)
    angle_count = np.zeros(8,dtype=np.int64)
    missing_angles = 0
    units_warned = False
    for n,(cid,file3,filep) in enumerate(records,1):
        with nc.Dataset(file3) as ds3, nc.Dataset(filep) as dsp:
            g3,gp = ds3.groups['data'],dsp.groups['data']
            v3,vp = g3.variables['swdn_3d'],gp.variables['swdn']
            if v3.shape != vp.shape or v3.shape[0] != len(local_month):
                raise ValueError(f'CID {cid}: SWDN shapes must match each other and supplied times.')
            z3,zp = _read_selected(v3,union),_read_selected(vp,union)
            valid = np.isfinite(z3)&np.isfinite(zp)&(z3 > -9000)&(zp > -9000)
            angle = None
            if 'azimuth' in g3.variables:
                av = g3.variables['azimuth']
                if av.ndim == 1 and av.shape[0] == len(local_month):
                    angle = np.ma.asarray(av[union],dtype=float).filled(np.nan)[:,None]
                elif av.ndim == 2 and av.shape[0] == len(local_month) and av.shape[1] in (1,v3.shape[1]):
                    angle = _read_selected(av,union)
                else:
                    raise ValueError(f'CID {cid}: unexpected azimuth shape {av.shape}')
                angle=np.broadcast_to(angle,z3.shape)
                angle=np.where(np.isfinite(angle)&(angle > -9000),angle,np.nan)
                units=str(getattr(av,'units','')).strip().lower()
                chosen=azimuth_units
                if chosen=='auto':
                    if units.startswith('deg'): chosen='degrees'
                    elif units.startswith('rad'): chosen='radians'
                    elif not units:
                        chosen='radians'
                        if not units_warned:
                            print('Azimuth units absent: assuming radians, as in original example.',flush=True)
                            units_warned=True
                    else: raise ValueError(f'Unrecognized azimuth units {units!r}; set azimuth_units explicitly.')
                if chosen=='degrees': angle=np.deg2rad(angle)
            else:
                missing_angles += 1
            for j,pos in enumerate(positions):
                ok = valid[pos]
                counts=ok.sum(axis=0)
                mean_diff=np.divide(np.where(ok,z3[pos]-zp[pos],0).sum(axis=0),counts,
                    out=np.full(z3.shape[1],np.nan),where=counts>0)
                pixels,hru_indices=placement[cid]
                if hru_indices.max() >= len(mean_diff):
                    raise IndexError(f'CID {cid}: raster HRU index exceeds raw data vector length.')
                differences[j].ravel()[pixels]=mean_diff[hru_indices]
                if angle is not None:
                    theta=angle[pos]
                    aok=ok&np.isfinite(theta)
                    sine[j]+=np.where(aok,np.sin(theta),0).sum()
                    cosine[j]+=np.where(aok,np.cos(theta),0).sum()
                    angle_count[j]+=aok.sum()
        if n==1 or n%50==0 or n==len(records):
            print(f'Read {n}/{len(records)} CIDs',flush=True)
    if buffer:
        differences=[data[buffer:-buffer,buffer:-buffer].copy() for data in differences]
    if not any(np.isfinite(data).any() for data in differences):
        raise RuntimeError('No valid mapped SWDN differences; check raster HRU IDs.')
    azimuth=[_angle_mean(s,c,n) for s,c,n in zip(sine,cosine,angle_count)]
    os.makedirs(outputdir,exist_ok=True)
    suffix=os.path.splitext(filename)[0]
    if start_year is not None or end_year is not None:
        suffix+=f'_years_{start_year or "all"}_{end_year or "all"}'
    fig,axes=plt.subplots(4,2,figsize=(10,14),layout='constrained')
    for j,ax in enumerate(axes.flat):
        finite=differences[j][np.isfinite(differences[j])]
        panel_limit=max(float(np.abs(finite).max()),1e-12) if finite.size else 1.0
        norm=TwoSlopeNorm(vmin=-panel_limit,vcenter=0,vmax=panel_limit)
        if coords_x is not None and coords_y is not None:
            cx=coords_x[buffer:-buffer] if buffer else coords_x
            cy=coords_y[buffer:-buffer] if buffer else coords_y
            im=ax.pcolormesh(cx,cy,np.flipud(np.ma.masked_invalid(differences[j])),shading='auto',cmap="bwr",norm=norm)
            ax.set(xlabel='Longitude [°]',ylabel='Latitude [°]')
            rows,cols=np.where(np.isfinite(np.flipud(differences[j])))
            if rows.size:
                dx=abs(cx[1]-cx[0]) if len(cx)>1 else .01
                dy=abs(cy[1]-cy[0]) if len(cy)>1 else .01
                px=max(.02*(cx[cols].max()-cx[cols].min()),dx)
                py=max(.02*(cy[rows].max()-cy[rows].min()),dy)
                ax.set_xlim(cx[cols].min()-px,cx[cols].max()+px)
                ax.set_ylim(cy[rows].min()-py,cy[rows].max()+py)
        else:
            im=ax.imshow(np.ma.masked_invalid(differences[j]),origin='upper',cmap="bwr",norm=norm,interpolation='nearest')
            ax.set(xlabel='Raster column',ylabel='Raster row')
        az=f'{azimuth[j]:.0f}°' if np.isfinite(azimuth[j]) else 'NA'
        window=f'{hours[j]:02d} and {hours[j]+1:02d}' if include_next_hour else f'{hours[j]:02d}'
        ax.set_title(f'azi = {az}, month = {months[j]}, hours {window} local')
        fig.colorbar(im,ax=ax,label='SWDN 3D − SWDN PP [W m$^{-2}$]')
    fig.suptitle(f'SWDN 3D − PP | {suffix} | {len(records)} CIDs')
    fig.savefig(os.path.join(outputdir,f'raw_swdn_3D_minus_PP_{suffix}.png'),dpi=dpi)
    plt.close(fig)
    with open(os.path.join(outputdir,f'raw_swdn_metrics_{suffix}.csv'),'w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['month','hour','include_next_hour','selected_timestamps','valid_pixels','MD','MAD','RMSE','mean_azimuth_degrees','azimuth_hru_time_samples','cids_missing_azimuth'])
        for j,d in enumerate(differences):
            d=d[np.isfinite(d)]
            w.writerow([months[j],hours[j],include_next_hour,len(selections[j]),d.size,
                d.mean() if d.size else np.nan,np.abs(d).mean() if d.size else np.nan,
                np.sqrt(np.mean(d*d)) if d.size else np.nan,azimuth[j],angle_count[j],missing_angles])
    print(f'Saved one 4x2 difference figure and metrics to {outputdir}',flush=True)
    return differences

def _raw_times(dataset, start, step, count):
    candidates=[dataset]+list(dataset.groups.values())
    for group in candidates:
        for name in ('time','t'):
            if name not in group.variables: continue
            v=group.variables[name]
            if v.ndim==1 and len(v)==count and 'since' in str(getattr(v,'units','')):
                if np.ma.getmaskarray(v[:]).any(): raise ValueError('Masked raw time coordinate.')
                dates=nc.num2date(v[:],v.units,calendar=getattr(v,'calendar','standard'))
                return pd.DatetimeIndex([pd.Timestamp(year=d.year,month=d.month,day=d.day,hour=d.hour,minute=d.minute,second=d.second,microsecond=d.microsecond) for d in dates])
    if start is None or step is None:
        raise ValueError('No usable CF time coordinate. Supply --time-start and --time-step-hours '
                         'from the model output convention; do not guess the first timestamp.')
    if step <= 0: raise ValueError('Time step must be positive.')
    return pd.date_range(start,periods=count,freq=pd.Timedelta(hours=step))


def main():
    p=argparse.ArgumentParser(description='Plot raw SWDN from CID files: December/July and four local-hour windows.')
    p.add_argument('--edir-3d',required=True)
    p.add_argument('--edir-pp',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--filename',default='2016-01-01.nc')
    p.add_argument('--start-year',type=int,default=2014)
    p.add_argument('--end-year',type=int,default=2024)
    p.add_argument('--cids',type=int,nargs='+',help='Default: CIDs 1 through 1600')
    p.add_argument('--map-path',help='Experiment holding postprocess/hrus.vrt and cids.vrt; default PP')
    p.add_argument('--utc-offset-hours',type=float,required=True,help='Offset added to raw timestamps; 0 if local already')
    p.add_argument('--time-start',help='First raw output timestamp ONLY if CF time coordinate is unavailable')
    p.add_argument('--time-step-hours',type=float,help='Raw output cadence ONLY if CF time coordinate is unavailable')
    p.add_argument('--exact-hour',action='store_true',help='Use only hour, rather than hour and hour+1')
    p.add_argument('--months',type=int,nargs=8,default=[12,7,12,7,12,7,12,7])
    p.add_argument('--hours',type=int,nargs=8,default=[8,8,11,11,14,14,17,17])
    p.add_argument('--buffer',type=int,default=0)
    p.add_argument('--azimuth-units',choices=['auto','radians','degrees'],default='auto')
    p.add_argument('--dpi',type=int,default=400)
    a=p.parse_args()
    if not np.isfinite(a.utc_offset_hours) or abs(a.utc_offset_hours)>24: p.error('Invalid time offset.')
    if a.end_year < a.start_year: p.error('--end-year must be at least --start-year.')
    import geospatialtools.gdal_tools as gdal_tools
    path=a.map_path or a.edir_pp
    hrus=gdal_tools.read_raster(os.path.join(path,'postprocess','hrus.vrt'))
    cids=gdal_tools.read_raster(os.path.join(path,'postprocess','cids.vrt'))
    meta=gdal_tools.retrieve_metadata(os.path.join(path,'postprocess','hrus.vrt'))
    cid_list=a.cids or list(range(1,1601))
    # Validate actual decoded time axes for every CID before using shared indices.
    reference=None
    for i,cid in enumerate(cid_list,1):
        for experiment in (a.edir_3d,a.edir_pp):
            with nc.Dataset(os.path.join(experiment,'output_data',str(cid),a.filename)) as ds:
                g=ds.groups['data']
                v=g.variables['swdn_3d' if experiment==a.edir_3d else 'swdn']
                axis=_raw_times(ds,a.time_start,a.time_step_hours,v.shape[0])
                if axis.has_duplicates or not axis.is_monotonic_increasing:
                    raise ValueError(f'CID {cid}: raw timestamps must be unique and increasing.')
                if reference is None: reference=axis
                elif not reference.equals(axis): raise ValueError(f'Time axis mismatch for CID {cid}, {experiment}')
        if i%100==0: print(f'Checked time axes: {i}/{len(cid_list)} CIDs',flush=True)
    if a.time_start is not None:
        print('Fallback time-start supplied: confirm it is the first OUTPUT timestamp, not just simulation start.',flush=True)
    local=reference+pd.Timedelta(hours=a.utc_offset_hours)
    print(f'Raw UTC time coverage: {reference[0]} to {reference[-1]}; selecting local years {a.start_year} through {a.end_year}',flush=True)
    times=SimpleNamespace(dt=SimpleNamespace(month=local.month,hour=local.hour,year=local.year))
    nx,ny=meta['nx'],meta['ny']
    if hrus.shape!=(ny,nx): raise ValueError('Metadata/raster shape mismatch.')
    # Pixel centers, ordered west->east and south->north for flipud plotting.
    dx=(meta['maxx']-meta['minx'])/nx; dy=(meta['maxy']-meta['miny'])/ny
    cx=meta['minx']+(np.arange(nx)+.5)*dx
    cy=meta['miny']+(np.arange(ny)+.5)*dy
    print('Mapping assumes zero-based HRU IDs with identical HRU ordering in 3D and PP, as in supplied place_data.',flush=True)
    plot_raw_cid_swdn(a.edir_3d,a.edir_pp,times,hrus,cids,a.outdir,
        cid_list=cid_list,filename=a.filename,months=a.months,hours=a.hours,
        include_next_hour=not a.exact_hour,buffer=a.buffer,dpi=a.dpi,
        azimuth_units=a.azimuth_units,coords_x=cx,coords_y=cy,
        start_year=a.start_year,end_year=a.end_year)

if __name__=='__main__':
    main()
