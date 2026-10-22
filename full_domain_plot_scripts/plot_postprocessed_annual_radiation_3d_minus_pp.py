#!/usr/bin/env python3
"""Annual 3D-minus-PP radiation maps and step histograms from postprocessed files.

One 2x3 figure per year: SWDN, SWNET, and RNET = SWNET - LWNET.
Annual flux means retain W/m2; they are not integrated annual energy totals.
Each map pixel is averaged over paired valid output samples. Histograms and
metrics describe spatial differences of those annual means, with equal pixel
weights. Dates absent from both runs are reported. Requires numpy, matplotlib,
netCDF4. Input paths: <experiment>/postprocess/output_dir/YYYYMMDD.nc.
"""
import argparse
import csv
import glob
import os
from datetime import date,timedelta
from concurrent.futures import ProcessPoolExecutor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import netCDF4 as nc
import numpy as np

FIELDS=(('swdn','SWDN'),('swnet','SWNET'),('rnet','Net radiation'))

def axis_index(dimensions, candidates):

    lowered = [name.lower() for name in dimensions]

    for candidate in candidates:

        if candidate in lowered:

            return lowered.index(candidate)

    return None





def read_time_lat_lon(variable, soil_layer):

    """Read a variable as (time, lat, lon), selecting extra axes if present."""

    dimensions = list(variable.dimensions)

    time_axis = axis_index(dimensions, ("time", "t"))

    lat_axis = axis_index(dimensions, ("lat", "latitude", "y"))

    lon_axis = axis_index(dimensions, ("lon", "longitude", "x"))

    if None in (time_axis, lat_axis, lon_axis):

        raise ValueError(

            f"Cannot identify time/latitude/longitude axes for "

            f"{variable.name}: {variable.dimensions}"

        )

    selection = []

    retained_dimensions = []

    for axis, name in enumerate(dimensions):

        if axis in (time_axis, lat_axis, lon_axis):

            selection.append(slice(None))

            retained_dimensions.append(name)

        else:

            if soil_layer < 0 or soil_layer >= variable.shape[axis]:

                raise IndexError(

                    f"Requested layer {soil_layer} exceeds axis {name} of "

                    f"{variable.name}, whose size is {variable.shape[axis]}"

                )

            selection.append(soil_layer)

    data = np.ma.asarray(variable[tuple(selection)], dtype=float).filled(np.nan)

    time_axis = axis_index(retained_dimensions, ("time", "t"))

    lat_axis = axis_index(retained_dimensions, ("lat", "latitude", "y"))

    lon_axis = axis_index(retained_dimensions, ("lon", "longitude", "x"))

    return np.transpose(data, (time_axis, lat_axis, lon_axis))





def read_coordinates(dataset):

    lon_name = next(

        (name for name in ("lon", "longitude", "x") if name in dataset.variables),

        None,

    )

    lat_name = next(

        (name for name in ("lat", "latitude", "y") if name in dataset.variables),

        None,

    )

    if lon_name is None or lat_name is None:

        raise KeyError("Postprocessed files must contain lon and lat coordinates")

    return (

        np.asarray(dataset.variables[lon_name][:], dtype=float),

        np.asarray(dataset.variables[lat_name][:], dtype=float),

    )





def list_days(experiment,year):
    return {os.path.basename(f)[:8]:f for f in glob.glob(os.path.join(experiment,'postprocess','output_dir',f'{year}????.nc'))}


def read_chunk(task):
    records,shape,orders=task
    sums3=np.zeros((3,)+shape)
    sumsp=np.zeros_like(sums3)
    counts=np.zeros((3,)+shape,dtype=np.int64)
    for key,f3,fp in records:
        with nc.Dataset(f3) as d3,nc.Dataset(fp) as dp:
            values=[]
            for ds, (iy, ix, expected_lat, expected_lon) in zip((d3, dp), orders):
                lon,lat=read_coordinates(ds)
                if lon.shape!=expected_lon.shape or lat.shape!=expected_lat.shape or not np.allclose(lon,expected_lon,rtol=0,atol=1e-8) or not np.allclose(lat,expected_lat,rtol=0,atol=1e-8):
                    raise ValueError(f'Grid changes on {key}')
                fields=[]
                for name in ('swdn_3d','swnet','lwnet'):
                    v=read_time_lat_lon(ds.variables[name],0)
                    v=v[:,iy][:,:,ix]
                    fields.append(np.where(np.isfinite(v)&(v>-9000),v,np.nan))
                if len({v.shape for v in fields})!=1:raise ValueError(f'Component shape mismatch on {key}')
                values.append([fields[0],fields[1],fields[1]-fields[2]])
            # If CF time coordinates exist, require identical decoded timestamps.
            clocks=[]
            for ds in (d3,dp):
                name=next((n for n in ('time','t') if n in ds.variables),None)
                if name is None:clocks.append(None);continue
                v=ds.variables[name]
                if 'since' not in str(getattr(v,'units','')):clocks.append(None);continue
                dates=nc.num2date(v[:],v.units,calendar=getattr(v,'calendar','standard'))
                clocks.append([(t.year,t.month,t.day,t.hour,t.minute,t.second) for t in dates])
            if (clocks[0] is None)!=(clocks[1] is None):raise ValueError(f'Time metadata available in only one run on {key}')
            if clocks[0] is not None and clocks[0]!=clocks[1]:raise ValueError(f'Time mismatch on {key}')
            for k,(a,b) in enumerate(zip(*values)):
                if a.shape!=b.shape:raise ValueError(f'3D/PP shape mismatch on {key}')
                valid=np.isfinite(a)&np.isfinite(b)
                sums3[k]+=np.where(valid,a,0).sum(axis=0)
                sumsp[k]+=np.where(valid,b,0).sum(axis=0)
                counts[k]+=valid.sum(axis=0)
    return sums3,sumsp,counts,len(records)


def draw(year,lon,lat,mean3,meanp,outdir,dpi,bins):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.titlesize':14})
    fig,axes=plt.subplots(2,3,figsize=(16,9),layout='constrained',height_ratios=[1.2,1])
    rows=[]
    for k,(name,label) in enumerate(FIELDS):
        diff=mean3[k]-meanp[k]
        valid=np.isfinite(diff)
        z=diff[valid]
        ax,hax=axes[:,k]
        if not z.size:
            ax.set_title(f'{label}: no valid pixels');ax.axis('off');hax.axis('off');continue
        limit=max(float(np.abs(z).max()),1e-12)
        im=ax.pcolormesh(lon,lat,np.ma.masked_invalid(diff),shading='auto',cmap='RdBu_r',norm=TwoSlopeNorm(vmin=-limit,vcenter=0,vmax=limit))
        ax.set(title=f'{label}: 3D − PP',xlabel='Longitude [°]',ylabel='Latitude [°]')
        rr,cc=np.where(valid)
        dx=abs(lon[1]-lon[0]) if len(lon)>1 else .01
        dy=abs(lat[1]-lat[0]) if len(lat)>1 else .01
        px=max(.02*np.ptp(lon[cc]),dx);py=max(.02*np.ptp(lat[rr]),dy)
        ax.set_xlim(lon[cc].min()-px,lon[cc].max()+px);ax.set_ylim(lat[rr].min()-py,lat[rr].max()+py)
        fig.colorbar(im,ax=ax,label='W m$^{-2}$',shrink=.85)
        md=float(z.mean());mad=float(np.abs(z).mean());rmse=float(np.sqrt(np.mean(z*z)))
        a,b=mean3[k][valid],meanp[k][valid]
        r=float(np.corrcoef(a,b)[0,1]) if z.size>1 and a.std()>0 and b.std()>0 else np.nan
        # Full-range outline histogram: no tails are clipped or omitted.
        edges=np.linspace(-limit,limit,bins+1)
        hax.hist(z,bins=edges,density=True,histtype='stepfilled',color='#4285a4',alpha=.18)
        hax.hist(z,bins=edges,density=True,histtype='step',color='#1b5368',linewidth=1.8)
        hax.axvline(0,color='0.3',linewidth=1)
        hax.axvline(md,color='#b64b30',linestyle='--',linewidth=1.8,label='Mean difference')
        hax.set(xlabel='Annual mean 3D − PP [W m$^{-2}$]',ylabel='Probability density',xlim=(-limit,limit))
        hax.spines[['top','right']].set_visible(False)
        hax.text(.97,.97,f'MD = {md:.2f}\nMAD = {mad:.2f}\nRMSE = {rmse:.2f}\nr = {r:.3f}\nn = {z.size:,}',transform=hax.transAxes,ha='right',va='top',fontsize=10,bbox=dict(facecolor='white',edgecolor='0.8',alpha=.9))
        rows.append([year,name,'W m-2',md,mad,rmse,r,z.size,float(z.min()),float(z.max())])
    fig.suptitle(f'Annual radiation differences · {year}\nNet radiation = SWNET - LWNET',fontsize=17,fontweight='bold')
    output=os.path.join(outdir,f'annual_radiation_3D_minus_PP_{year}.png')
    fig.savefig(output,dpi=dpi,facecolor='white');plt.close(fig)
    print(f'Saved: {output}',flush=True)
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--edir-3d',required=True);p.add_argument('--edir-pp',required=True)
    p.add_argument('--outdir',required=True)
    p.add_argument('--start-year',type=int,default=2014);p.add_argument('--end-year',type=int,default=2024)
    p.add_argument('--workers',type=int,default=1);p.add_argument('--dpi',type=int,default=300)
    p.add_argument('--bins',type=int,default=50)
    a=p.parse_args()
    if a.end_year<a.start_year or min(a.workers,a.dpi,a.bins)<1:p.error('Invalid year range, workers, dpi or bins.')
    os.makedirs(a.outdir,exist_ok=True)
    all_rows=[]
    for year in range(a.start_year,a.end_year+1):
        f3,fp=list_days(a.edir_3d,year),list_days(a.edir_pp,year)
        common=sorted(set(f3)&set(fp))
        expected={(date(year,1,1)+timedelta(days=i)).strftime('%Y%m%d') for i in range((date(year+1,1,1)-date(year,1,1)).days)}
        print(f'{year}: matched days={len(common)}, absent from both={len(expected-(set(f3)|set(fp)))}, unpaired={len(set(f3)^set(fp))}',flush=True)
        if not common:raise FileNotFoundError(f'No matching daily files for {year}')
        orders=[]
        with nc.Dataset(f3[common[0]]) as d3,nc.Dataset(fp[common[0]]) as dp:
            coords=[read_coordinates(d) for d in (d3,dp)]
        for lon,lat in coords:
            if any(v.ndim!=1 or not np.isfinite(v).all() or len(np.unique(v))!=len(v) for v in (lon,lat)):raise ValueError('Requires unique finite 1D longitude/latitude.')
            orders.append((np.argsort(lat),np.argsort(lon),lat,lon))
        if coords[0][0].shape!=coords[1][0].shape or coords[0][1].shape!=coords[1][1].shape or any(not np.allclose(np.sort(x),np.sort(y),rtol=0,atol=1e-8) for x,y in zip(*coords)):raise ValueError('3D/PP grids differ.')
        lon,lat=np.sort(coords[0][0]),np.sort(coords[0][1])
        shape=(len(lat),len(lon))
        s3,sp=np.zeros((3,)+shape),np.zeros((3,)+shape)
        count=np.zeros((3,)+shape,dtype=np.int64)
        records=[(k,f3[k],fp[k]) for k in common]
        chunk_size=max(1,(len(records)+a.workers*4-1)//(a.workers*4))
        tasks=[(records[i:i+chunk_size],shape,orders) for i in range(0,len(records),chunk_size)]
        pool=ProcessPoolExecutor(max_workers=a.workers) if a.workers>1 else None
        results=pool.map(read_chunk,tasks) if pool else map(read_chunk,tasks)
        done=0
        try:
            for u,v,n,days in results:
                s3+=u;sp+=v;count+=n;done+=days
                print(f'{year}: processed {done}/{len(records)} days',flush=True)
        finally:
            if pool:pool.shutdown()
        m3=np.divide(s3,count,out=np.full(s3.shape,np.nan),where=count>0)
        mp=np.divide(sp,count,out=np.full(sp.shape,np.nan),where=count>0)
        if not np.isfinite(m3).any():raise RuntimeError(f'No paired valid radiation samples in {year}')
        all_rows.extend(draw(year,lon,lat,m3,mp,a.outdir,a.dpi,a.bins))
    with open(os.path.join(a.outdir,f'annual_radiation_metrics_{a.start_year}_{a.end_year}.csv'),'w',newline='') as f:
        w=csv.writer(f);w.writerow(['year','variable','unit','MD','MAD','RMSE','r','valid_pixels','min_difference','max_difference']);w.writerows(all_rows)

if __name__=='__main__':main()
