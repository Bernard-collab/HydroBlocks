#!/usr/bin/env python3
"""
Diagnostic checks for the SMAP NSIDC-0779 GeoTIFF collection.

Checks performed, per file:
  1. Internal consistency: does (bounds / pixel_size) actually match the
     reported array shape, in both x and y? A mismatch means the file's
     metadata is self-contradictory (as found for the 2015-04-01 file).
  2. Cross-reference against any existing bad_files.txt / corrupt_files.txt
     / missing_files.txt / failed_urls.txt in the same directory, so we
     know whether problems were already flagged by an earlier download
     pass.
  3. A quick data-readability check: can the pixel array actually be
     read without error, and does it contain any finite (non-nodata)
     values at all?

Run this across a sample of files first (fast), then across the full
collection if the sample reveals no major differences from file to file
(slow, but thorough).

Usage
-----
# Check a handful of specific files:
python check_smap_files.py --dir "/path/to/SMAP_0779_2015_2024_full domain" --sample 20

# Check every .tif file in the directory (slower):
python check_smap_files.py --dir "/path/to/SMAP_0779_2015_2024_full domain" --all

# Check specific files only:
python check_smap_files.py --dir "/path/to/SMAP_0779_2015_2024_full domain" \
    --files NSIDC-0779_EASE2_G1km_SMAP_SM_DS_20150401.tif,NSIDC-0779_EASE2_G1km_SMAP_SM_DS_20150402.tif
"""

import argparse
import glob
import os
import random
import sys

import numpy as np
import rasterio


def parse_args():
    parser = argparse.ArgumentParser(description="Check SMAP GeoTIFF files for internal consistency.")
    parser.add_argument("--dir", required=True, help="Directory containing the SMAP .tif files")
    parser.add_argument("--pattern", default="NSIDC-0779_EASE2_G1km_SMAP_SM_DS_*.tif", help="Glob pattern for SMAP files")
    parser.add_argument("--sample", type=int, default=None, help="Randomly check this many files instead of all")
    parser.add_argument("--all", action="store_true", help="Check every matching file (overrides --sample)")
    parser.add_argument("--files", default=None, help="Comma-separated list of specific filenames to check")
    parser.add_argument("--tolerance", type=float, default=0.02, help="Fractional mismatch tolerance before flagging as inconsistent (default: 2%)")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for --sample selection")
    return parser.parse_args()


def load_known_bad_lists(directory):
    """Read any existing bad/corrupt/missing/failed tracking files, if present."""
    known = {}
    for fname in ("bad_files.txt", "corrupt_files.txt", "missing_files.txt",
                  "missing_files_numbered.txt", "missing_final.txt", "failed_urls.txt"):
        fpath = os.path.join(directory, fname)
        if os.path.isfile(fpath):
            with open(fpath, "r", errors="ignore") as f:
                content = f.read()
            known[fname] = content
        else:
            known[fname] = None
    return known


def check_against_known_lists(filename, known_lists):
    flags = []
    for list_name, content in known_lists.items():
        if content is not None and filename in content:
            flags.append(list_name)
    return flags


def check_file(filepath, tolerance):
    """Check one file for internal consistency and readability."""
    result = {
        "file": os.path.basename(filepath),
        "readable": False,
        "crs": None,
        "shape": None,
        "res": None,
        "bounds": None,
        "implied_height": None,
        "implied_width": None,
        "height_mismatch_pct": None,
        "width_mismatch_pct": None,
        "consistent": None,
        "has_finite_data": None,
        "error": None,
    }

    try:
        with rasterio.open(filepath) as src:
            result["crs"] = str(src.crs)
            result["shape"] = src.shape
            result["res"] = src.res
            result["bounds"] = tuple(src.bounds)

            pixel_width = abs(src.transform.a)
            pixel_height = abs(src.transform.e)

            implied_width = (src.bounds.right - src.bounds.left) / pixel_width
            implied_height = (src.bounds.top - src.bounds.bottom) / pixel_height

            result["implied_width"] = implied_width
            result["implied_height"] = implied_height

            width_mismatch = abs(implied_width - src.width) / src.width
            height_mismatch = abs(implied_height - src.height) / src.height

            result["width_mismatch_pct"] = 100 * width_mismatch
            result["height_mismatch_pct"] = 100 * height_mismatch

            result["consistent"] = (width_mismatch <= tolerance) and (height_mismatch <= tolerance)

            # Quick readability / data-content check: read a small window, not the whole array
            sample_window = rasterio.windows.Window(0, 0, min(200, src.width), min(200, src.height))
            sample_data = src.read(1, window=sample_window)

            nodata = src.nodata
            if nodata is not None:
                finite_mask = (sample_data != nodata) & np.isfinite(sample_data)
            else:
                finite_mask = np.isfinite(sample_data)

            result["has_finite_data"] = bool(np.any(finite_mask))
            result["readable"] = True

    except Exception as error:
        result["error"] = str(error)

    return result


def main():
    args = parse_args()

    directory = args.dir

    if not os.path.isdir(directory):
        sys.exit(f"Directory not found: {directory}")

    known_lists = load_known_bad_lists(directory)
    print("Existing tracking files found:")
    for name, content in known_lists.items():
        status = f"{len(content.splitlines())} lines" if content is not None else "not found"
        print(f"  {name}: {status}")
    print()

    if args.files:
        filenames = [f.strip() for f in args.files.split(",") if f.strip()]
        filepaths = [os.path.join(directory, f) for f in filenames]
    else:
        filepaths = sorted(glob.glob(os.path.join(directory, args.pattern)))
        if not filepaths:
            sys.exit(f"No files matched pattern '{args.pattern}' in {directory}")

        if not args.all and args.sample:
            random.seed(args.seed)
            filepaths = sorted(random.sample(filepaths, min(args.sample, len(filepaths))))

    print(f"Checking {len(filepaths)} file(s)...\n")

    results = []
    for filepath in filepaths:
        result = check_file(filepath, args.tolerance)
        result["known_bad_flags"] = check_against_known_lists(result["file"], known_lists)
        results.append(result)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    n_total = len(results)
    n_unreadable = sum(1 for r in results if not r["readable"])
    n_inconsistent = sum(1 for r in results if r["readable"] and not r["consistent"])
    n_no_data = sum(1 for r in results if r["readable"] and r["consistent"] and not r["has_finite_data"])
    n_flagged_known = sum(1 for r in results if r["known_bad_flags"])
    n_good = n_total - n_unreadable - n_inconsistent - n_no_data

    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total files checked:              {n_total}")
    print(f"  Unreadable (open/read error):    {n_unreadable}")
    print(f"  Bounds/shape inconsistent:       {n_inconsistent}")
    print(f"  Readable+consistent but no data: {n_no_data}")
    print(f"  Already flagged in tracking txt: {n_flagged_known}")
    print(f"  Appear fully OK:                 {n_good}")
    print()

    # ------------------------------------------------------------------
    # Details for anything problematic
    # ------------------------------------------------------------------
    problems = [r for r in results if not r["readable"] or not r.get("consistent", True) or not r.get("has_finite_data", True)]

    if problems:
        print("=" * 70)
        print(f"DETAILS FOR {len(problems)} PROBLEM FILE(S)")
        print("=" * 70)
        for r in problems:
            print(f"\n  File: {r['file']}")
            if r["error"]:
                print(f"    ERROR: {r['error']}")
                continue
            print(f"    Shape: {r['shape']}  Resolution: {r['res']}")
            print(f"    Width mismatch: {r['width_mismatch_pct']:.2f}%   Height mismatch: {r['height_mismatch_pct']:.2f}%")
            print(f"    Consistent: {r['consistent']}   Has finite data: {r['has_finite_data']}")
            if r["known_bad_flags"]:
                print(f"    Already flagged in: {', '.join(r['known_bad_flags'])}")
    else:
        print("No problems found among the checked files.")

    # ------------------------------------------------------------------
    # Consistency across files: do all "good" files agree on resolution/shape?
    # ------------------------------------------------------------------
    good_results = [r for r in results if r["readable"] and r["consistent"]]
    if good_results:
        shapes = set(r["shape"] for r in good_results)
        resolutions = set(r["res"] for r in good_results)
        print("\n" + "=" * 70)
        print("CROSS-FILE CONSISTENCY (among files that passed internal checks)")
        print("=" * 70)
        print(f"Distinct shapes seen: {shapes}")
        print(f"Distinct resolutions seen: {resolutions}")
        if len(shapes) > 1 or len(resolutions) > 1:
            print(
                "\nWARNING: files that individually look internally consistent still "
                "disagree with each other on shape/resolution -- check whether all "
                "files are actually from the same product/version before using them "
                "together."
            )


if __name__ == "__main__":
    main()
