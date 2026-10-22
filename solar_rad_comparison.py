#!/usr/bin/env python3
"""
Create a domain-average time series of HydroBlocks input solar radiation.

Reads:
    <edir>/<cid>/input_file.nc
    group: meteorology
    variable: swdown

Produces:
    1. Daily GHI time series in kWh m-2 day-1
    2. 30-day rolling mean
    3. Monthly climatology
    4. Comparison with published Colorado annual-average GHI
"""

import argparse
import os
import sys

import matplotlib.pyplot as plt
import netCDF4 as nc
import numpy as np
import pandas as pd


COLORADO_GHI_MIN = 4.5
COLORADO_GHI_MAX = 5.5


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot domain-average HydroBlocks input solar radiation."
    )

    parser.add_argument(
        "--edir",
        required=True,
        help="Experiment directory containing <cid>/input_file.nc",
    )

    parser.add_argument(
        "--out",
        required=True,
        help="Output PNG path",
    )

    parser.add_argument(
        "--cids",
        default=None,
        help="Optional comma-separated CID list; default is all numeric CIDs",
    )

    parser.add_argument(
        "--dpi",
        type=int,
        default=300,
    )

    return parser.parse_args()


def available_cids(edir, requested):
    if requested:
        cids = [
            int(value.strip())
            for value in requested.split(",")
            if value.strip()
        ]
    else:
        cids = sorted(
            int(name)
            for name in os.listdir(edir)
            if name.isdigit()
        )

    if not cids:
        raise RuntimeError("No numeric CID directories found under %s" % edir)

    return cids


def get_meteorology_group(dataset):
    if "meteorology" in dataset.groups:
        return dataset.groups["meteorology"]

    if "Meteorology" in dataset.groups:
        return dataset.groups["Meteorology"]

    raise KeyError("Missing meteorology group")


def read_dates(ncpath):
    with nc.Dataset(ncpath, "r") as dataset:
        group = get_meteorology_group(dataset)

        if "time" not in group.variables:
            raise KeyError("Missing meteorology/time")

        time_variable = group.variables["time"]

        dates = nc.num2date(
            time_variable[:],
            units=time_variable.units,
            calendar=getattr(time_variable, "calendar", "standard"),
            only_use_cftime_datetimes=False,
            only_use_python_datetimes=True,
        )

    return pd.DatetimeIndex(dates)


def read_hru_weights(dataset, number_of_hrus):
    """
    Read HRU spatial weights when available.

    Preference:
        area_pct
        area
        carea
        equal weighting
    """

    if "parameters" not in dataset.groups:
        return np.ones(number_of_hrus, dtype=np.float64)

    group = dataset.groups["parameters"]

    for variable in ["area_pct", "area", "carea"]:
        if variable not in group.variables:
            continue

        weights = np.ma.filled(
            group.variables[variable][:],
            np.nan,
        )

        weights = np.asarray(weights, dtype=np.float64)
        weights = np.squeeze(weights)

        if weights.ndim != 1:
            continue

        if weights.size != number_of_hrus:
            continue

        valid = np.isfinite(weights) & (weights > 0)

        if valid.any():
            weights = np.where(valid, weights, 0.0)
            return weights

    return np.ones(number_of_hrus, dtype=np.float64)


def read_cid_mean_swdown(ncpath):
    """
    Return an area-weighted SWdown time series for one CID.
    """

    with nc.Dataset(ncpath, "r") as dataset:
        group = get_meteorology_group(dataset)

        if "swdown" not in group.variables:
            raise KeyError("Missing meteorology/swdown")

        swdown = np.ma.filled(
            group.variables["swdown"][:],
            np.nan,
        )

        swdown = np.asarray(swdown, dtype=np.float64)
        swdown = np.squeeze(swdown)

        time_length = group.variables["time"].size

        if swdown.ndim == 1:
            if swdown.size != time_length:
                raise ValueError(
                    "Unexpected one-dimensional swdown shape %s"
                    % (swdown.shape,)
                )

            return swdown, 1.0

        if swdown.ndim != 2:
            raise ValueError(
                "Expected one- or two-dimensional swdown; found %s"
                % (swdown.shape,)
            )

        # Convert to (time, HRU) if necessary.
        if swdown.shape[0] == time_length:
            pass
        elif swdown.shape[1] == time_length:
            swdown = swdown.T
        else:
            raise ValueError(
                "Time length %d does not match swdown shape %s"
                % (time_length, swdown.shape)
            )

        number_of_hrus = swdown.shape[1]

        weights = read_hru_weights(
            dataset,
            number_of_hrus,
        )

        # Remove negative or nonfinite solar-radiation values.
        valid = np.isfinite(swdown) & (swdown >= 0.0)

        weighted_values = np.where(
            valid,
            swdown * weights[np.newaxis, :],
            0.0,
        )

        valid_weights = np.where(
            valid,
            weights[np.newaxis, :],
            0.0,
        )

        numerator = np.sum(weighted_values, axis=1)
        denominator = np.sum(valid_weights, axis=1)

        cid_mean = np.divide(
            numerator,
            denominator,
            out=np.full(time_length, np.nan),
            where=denominator > 0,
        )

        # Total CID weight used when combining CIDs.
        cid_weight = np.sum(weights)

    return cid_mean, cid_weight


def calculate_domain_mean(edir, cids):
    first_path = os.path.join(
        edir,
        str(cids[0]),
        "input_file.nc",
    )

    dates = read_dates(first_path)
    number_of_times = len(dates)

    weighted_sum = np.zeros(number_of_times, dtype=np.float64)
    weight_sum = np.zeros(number_of_times, dtype=np.float64)

    mapped_cids = 0
    skipped = []

    for number, cid in enumerate(cids, start=1):
        ncpath = os.path.join(
            edir,
            str(cid),
            "input_file.nc",
        )

        print(
            "Reading CID %d (%d/%d)"
            % (cid, number, len(cids)),
            flush=True,
        )

        if not os.path.isfile(ncpath):
            skipped.append((cid, "missing input_file.nc"))
            continue

        try:
            cid_dates = read_dates(ncpath)

            if len(cid_dates) != number_of_times:
                raise ValueError(
                    "time length differs from the first CID"
                )

            if not np.array_equal(cid_dates.values, dates.values):
                raise ValueError(
                    "time coordinates differ from the first CID"
                )

            cid_mean, cid_weight = read_cid_mean_swdown(ncpath)

        except (KeyError, ValueError, OSError) as error:
            skipped.append((cid, str(error)))
            continue

        valid = np.isfinite(cid_mean)

        weighted_sum[valid] += cid_mean[valid] * cid_weight
        weight_sum[valid] += cid_weight

        mapped_cids += 1

    domain_mean = np.divide(
        weighted_sum,
        weight_sum,
        out=np.full(number_of_times, np.nan),
        where=weight_sum > 0,
    )

    return dates, domain_mean, mapped_cids, skipped


def calculate_daily_ghi(dates, swdown):
    """
    Convert instantaneous/interval-average SWdown in W m-2 to daily
    irradiation in kWh m-2 day-1.

    The integration timestep is determined from the time coordinate.
    """

    if len(dates) < 2:
        raise ValueError("At least two timesteps are required")

    timestep_hours = (
        dates[1] - dates[0]
    ).total_seconds() / 3600.0

    if timestep_hours <= 0:
        raise ValueError("Invalid model timestep")

    print("Detected model timestep: %.2f hours" % timestep_hours)

    frame = pd.DataFrame(
        {
            "swdown_wm2": swdown,
        },
        index=dates,
    )

    # Energy contributed by each model timestep.
    frame["energy_kwh_m2"] = (
        frame["swdown_wm2"] * timestep_hours / 1000.0
    )

    # Sum all timesteps within each day.
    daily = frame["energy_kwh_m2"].resample("D").sum(
        min_count=1
    )

    return daily


def print_statistics(daily_ghi, mapped_cids, total_cids):
    valid = daily_ghi.dropna()

    full_period_mean = valid.mean()

    annual_means = valid.resample("YS").mean()

    print("")
    print("SOLAR-RADIATION SUMMARY")
    print("=======================")
    print("CIDs mapped: %d/%d" % (mapped_cids, total_cids))
    print(
        "Date range: %s to %s"
        % (
            valid.index.min().strftime("%Y-%m-%d"),
            valid.index.max().strftime("%Y-%m-%d"),
        )
    )
    print(
        "Full-period average daily GHI: "
        "%.3f kWh m-2 day-1"
        % full_period_mean
    )
    print(
        "Published Colorado annual-average range: "
        "%.1f-%.1f kWh m-2 day-1"
        % (COLORADO_GHI_MIN, COLORADO_GHI_MAX)
    )

    if COLORADO_GHI_MIN <= full_period_mean <= COLORADO_GHI_MAX:
        print(
            "Comparison: HydroBlocks average is within "
            "the expected Colorado range."
        )
    elif full_period_mean < COLORADO_GHI_MIN:
        print(
            "Comparison: HydroBlocks average is below "
            "the expected Colorado range."
        )
    else:
        print(
            "Comparison: HydroBlocks average is above "
            "the expected Colorado range."
        )

    print("")
    print("Annual-average daily GHI:")

    for date, value in annual_means.items():
        print(
            "  %d: %.3f kWh m-2 day-1"
            % (date.year, value)
        )

    return full_period_mean


def make_plot(daily_ghi, full_period_mean, output_path, dpi):
    rolling_30day = daily_ghi.rolling(
        window=30,
        center=True,
        min_periods=15,
    ).mean()

    monthly_climatology = daily_ghi.groupby(
        daily_ghi.index.month
    ).mean()

    month_labels = [
        "Jan", "Feb", "Mar", "Apr",
        "May", "Jun", "Jul", "Aug",
        "Sep", "Oct", "Nov", "Dec",
    ]

    fig, axes = plt.subplots(
        2,
        1,
        figsize=(15, 10),
    )

    # Panel 1: complete daily time series
    ax = axes[0]

    ax.plot(
        daily_ghi.index,
        daily_ghi.values,
        color="goldenrod",
        linewidth=0.45,
        alpha=0.45,
        label="Daily GHI",
    )

    ax.plot(
        rolling_30day.index,
        rolling_30day.values,
        color="darkred",
        linewidth=1.6,
        label="30-day rolling mean",
    )

    ax.axhline(
        full_period_mean,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=(
            "HydroBlocks period mean = %.2f"
            % full_period_mean
        ),
    )

    ax.axhspan(
        COLORADO_GHI_MIN,
        COLORADO_GHI_MAX,
        color="royalblue",
        alpha=0.13,
        label="Published Colorado annual mean: 4.5–5.5",
    )

    ax.set_title(
        "Domain-Average HydroBlocks Input Solar Radiation",
        fontsize=15,
        fontweight="bold",
    )

    ax.set_ylabel(
        r"Daily GHI (kWh m$^{-2}$ day$^{-1}$)"
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    # Explain that the published range is an annual benchmark.
    ax.text(
        0.01,
        0.03,
        "Blue band is an annual-average benchmark, "
        "not the expected value for every individual day.",
        transform=ax.transAxes,
        fontsize=9,
    )

    # Panel 2: monthly climatology
    ax = axes[1]

    ax.plot(
        np.arange(1, 13),
        monthly_climatology.values,
        color="darkorange",
        linewidth=2.2,
        marker="o",
        markersize=7,
    )

    ax.axhline(
        full_period_mean,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=(
            "HydroBlocks period mean = %.2f"
            % full_period_mean
        ),
    )

    ax.axhspan(
        COLORADO_GHI_MIN,
        COLORADO_GHI_MAX,
        color="royalblue",
        alpha=0.13,
        label="Published Colorado annual mean: 4.5–5.5",
    )

    ax.set_xticks(np.arange(1, 13))
    ax.set_xticklabels(month_labels)

    ax.set_xlabel("Month")
    ax.set_ylabel(
        r"Mean daily GHI (kWh m$^{-2}$ day$^{-1}$)"
    )

    ax.set_title(
        "Monthly Solar-Radiation Climatology",
        fontsize=14,
        fontweight="bold",
    )

    ax.grid(
        True,
        linestyle=":",
        linewidth=0.6,
        alpha=0.6,
    )

    ax.legend(
        loc="upper right",
        frameon=True,
        fontsize=9,
    )

    fig.subplots_adjust(
        left=0.075,
        right=0.975,
        bottom=0.075,
        top=0.94,
        hspace=0.27,
    )

    output_directory = os.path.dirname(output_path)

    if output_directory:
        os.makedirs(output_directory, exist_ok=True)

    fig.savefig(
        output_path,
        dpi=dpi,
        bbox_inches="tight",
    )

    plt.close(fig)


def main():
    args = parse_args()

    edir = os.path.abspath(args.edir)
    output_path = os.path.abspath(args.out)

    if not os.path.isdir(edir):
        sys.exit("Experiment directory not found: %s" % edir)

    try:
        cids = available_cids(
            edir,
            args.cids,
        )

        dates, domain_swdown, mapped_cids, skipped = (
            calculate_domain_mean(
                edir,
                cids,
            )
        )

        daily_ghi = calculate_daily_ghi(
            dates,
            domain_swdown,
        )

    except (RuntimeError, ValueError, KeyError, OSError) as error:
        sys.exit(str(error))

    if daily_ghi.notna().sum() == 0:
        sys.exit("No valid solar-radiation values were calculated")

    full_period_mean = print_statistics(
        daily_ghi,
        mapped_cids,
        len(cids),
    )

    make_plot(
        daily_ghi,
        full_period_mean,
        output_path,
        args.dpi,
    )

    print("")
    print("Skipped CIDs: %d" % len(skipped))

    for cid, reason in skipped[:10]:
        print("  CID %s: %s" % (cid, reason))

    print("")
    print("Wrote %s" % output_path)


if __name__ == "__main__":
    main()