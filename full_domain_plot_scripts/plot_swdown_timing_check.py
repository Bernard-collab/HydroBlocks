#!/usr/bin/env python3
"""Simple timing check for HydroBlocks 3-hourly input solar radiation."""

import argparse
import os

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--edir", required=True,
                        help="Experiment directory containing CID folders")
    parser.add_argument("--cid", type=int, default=1,
                        help="Representative CID to plot (default: 1)")
    parser.add_argument("--start", default="2014-06-15",
                        help="First day of the 7-day plot (YYYY-MM-DD)")
    parser.add_argument("--out", required=True, help="Output PNG path")
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def meteorology_group(dataset):
    for name in ("meteorology", "Meteorology"):
        if name in dataset.groups:
            return dataset.groups[name]
    raise KeyError("Missing meteorology group")


def read_cid(ncpath):
    with nc.Dataset(ncpath) as dataset:
        met = meteorology_group(dataset)
        time = met.variables["time"]
        dates = nc.num2date(
            time[:], time.units,
            calendar=getattr(time, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )
        dates = pd.DatetimeIndex(dates)

        swdown = np.asarray(
            np.ma.filled(met.variables["swdown"][:], np.nan),
            dtype=float,
        ).squeeze()

        if swdown.ndim == 1:
            mean_swdown = swdown
        elif swdown.ndim == 2:
            if swdown.shape[0] != len(dates):
                if swdown.shape[1] == len(dates):
                    swdown = swdown.T
                else:
                    raise ValueError(
                        "swdown shape does not match the time coordinate"
                    )
            # HRU averaging is sufficient for this timing-only diagnostic.
            mean_swdown = np.nanmean(swdown, axis=1)
        else:
            raise ValueError("Unexpected swdown shape: %s" % (swdown.shape,))

    return pd.Series(mean_swdown, index=dates, name="SWdown")


def main():
    args = parse_args()
    ncpath = os.path.join(args.edir, str(args.cid), "input_file.nc")
    if not os.path.isfile(ncpath):
        raise FileNotFoundError(ncpath)

    swdown = read_cid(ncpath)
    start = pd.Timestamp(args.start)
    end = start + pd.Timedelta(days=7)
    week = swdown.loc[(swdown.index >= start) & (swdown.index < end)]
    if week.empty:
        raise ValueError("Selected week is outside the forcing period")

    hourly_cycle = swdown.groupby(swdown.index.hour).mean()
    peak_hour = int(hourly_cycle.idxmax())

    fig, axes = plt.subplots(2, 1, figsize=(12, 7))

    axes[0].plot(week.index, week.values, marker="o", markersize=3,
                 linewidth=1.2, color="darkorange")
    axes[0].axhline(0, color="black", linewidth=0.7)
    axes[0].set_title("HydroBlocks input SWdown: 7-day timing check (UTC)")
    axes[0].set_ylabel(r"SWdown (W m$^{-2}$)")
    axes[0].grid(True, linestyle=":", alpha=0.6)

    axes[1].plot(hourly_cycle.index, hourly_cycle.values, marker="o",
                 linewidth=2, color="darkred")
    axes[1].axvspan(18, 21, color="royalblue", alpha=0.15,
                   label="Expected Upper Colorado peak window")
    axes[1].axvline(peak_hour, color="black", linestyle="--",
                    label="Observed peak = %02d:00 UTC" % peak_hour)
    axes[1].set_xticks(np.arange(0, 24, 3))
    axes[1].set_xlim(0, 23)
    axes[1].set_xlabel("Hour (UTC)")
    axes[1].set_ylabel(r"Mean SWdown (W m$^{-2}$)")
    axes[1].set_title("Mean diurnal cycle over the full forcing period")
    axes[1].grid(True, linestyle=":", alpha=0.6)
    axes[1].legend()

    fig.suptitle("CID %d — timing is reasonable if the peak is near 18–21 UTC"
                 % args.cid, fontweight="bold")
    fig.tight_layout()

    outdir = os.path.dirname(os.path.abspath(args.out))
    if outdir:
        os.makedirs(outdir, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight")
    plt.close(fig)

    print("Time range:", swdown.index.min(), "to", swdown.index.max())
    print("Detected timestep (hours):",
          (swdown.index[1] - swdown.index[0]).total_seconds() / 3600.0)
    print("Observed mean SWdown peak: %02d:00 UTC" % peak_hour)
    print("Expected Upper Colorado peak: approximately 18:00-21:00 UTC")
    print("Wrote:", os.path.abspath(args.out))


if __name__ == "__main__":
    main()
