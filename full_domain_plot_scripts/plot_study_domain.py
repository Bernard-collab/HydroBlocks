#!/usr/bin/env python3
"""
plot_study_domain.py
=====================

Publication-style map of a mountain-hydrology study domain, with:
  - Elevation (DEM) as the background, in a hypsometric colormap
  - Rivers
  - Major cities
  - State boundaries
  - Two highlighted sub-regions: the Rio Grande headwaters (southern
    Colorado) and the northern New Mexico watersheds
  - A small inset locator map showing where the domain sits within the
    broader southwestern United States

Requirements
------------
    pip install cartopy rasterio matplotlib numpy shapely

Cartopy will download Natural Earth vector data (states, rivers, cities)
automatically the first time each feature is used -- this requires internet
access on the machine actually running this script. If you're offline or
want to avoid repeated downloads, pre-fetch the Natural Earth shapefiles and
point NATURAL_EARTH_DIR (below) at a local copy.

Elevation is read from a DEM raster you supply (GeoTIFF, VRT, etc. --
anything rasterio/GDAL can open), e.g. an SRTM/Copernicus DEM covering at
least [-114, 30, -104, 44]. Pass its path with --dem.

Usage
-----
    python plot_study_domain.py --dem /scratch/alpine/battobrah@xsede.org/Rice_NMT_Snow/upper_colorado/experiments/simulations/MSWX_3h_Snow_Project3_3d_upper_colorado_11yr_50hru_full_domain_downscaled_test/postprocess/dem.vrt --out study_domain_map.png"""

import argparse

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
from rasterio.plot import reshape_as_image
from rasterio.warp import Resampling, calculate_default_transform, reproject

# ----------------------------------------------------------------------
# Study domain and highlighted sub-regions
# ----------------------------------------------------------------------
DOMAIN = dict(min_lon=-114.0, min_lat=30.0, max_lon=-104.0, max_lat=44.0)

# Rio Grande headwaters, southern Colorado (San Juan Mountains / San Luis Valley)
RIO_GRANDE_HEADWATERS = dict(min_lon=-107.7, max_lon=-105.4, min_lat=37.0, max_lat=38.3)

# Northern New Mexico watersheds (Rio Chama / upper Rio Grande)
N_NEW_MEXICO = dict(min_lon=-107.8, max_lon=-105.3, min_lat=35.5, max_lat=37.0)

# Broader southwestern-US extent used for the locator inset
SW_USA_EXTENT = dict(min_lon=-125.0, max_lon=-102.0, min_lat=25.0, max_lat=49.0)

# A hypsometric (green -> yellow -> brown -> white) elevation colormap,
# matching common mountain-hydrology cartographic convention
HYPSOMETRIC_CMAP = LinearSegmentedColormap.from_list(
    "hypsometric",
    [
        (0.00, "#225E2A"),  # low valleys - dark green
        (0.15, "#559E3C"),
        (0.35, "#AAB55F"),
        (0.55, "#D2AF7D"),
        (0.72, "#96694E"),
        (0.85, "#644434"),
        (1.00, "#FFFFFF"),  # highest peaks - white
    ],
)


def arguments():
    parser = argparse.ArgumentParser(description=__doc__,
                                      formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dem", required=True,
                         help="Path to a DEM raster (GeoTIFF/VRT) covering the study domain.")
    parser.add_argument("--out", default="study_domain_map.png",
                         help="Output image path (png/pdf/svg).")
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--city-min-population", type=int, default=50_000,
                         help="Only label cities at or above this population.")
    return parser.parse_args()


def read_dem_as_geographic(dem_path, bounds):
    """Read a DEM raster, reprojected to plain lat/lon (EPSG:4326) and
    clipped to `bounds`, returning (lon_edges, lat_edges, elevation)."""
    with rasterio.open(dem_path) as src:
        dst_crs = "EPSG:4326"
        transform, width, height = calculate_default_transform(
            src.crs, dst_crs, src.width, src.height, *src.bounds)
        elevation = np.full((height, width), np.nan, dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=elevation,
            src_transform=src.transform, src_crs=src.crs, src_nodata=src.nodata,
            dst_transform=transform, dst_crs=dst_crs, dst_nodata=np.nan,
            resampling=Resampling.average,
        )

    lon = transform.c + np.arange(width) * transform.a
    lat = transform.f + np.arange(height) * transform.e  # negative pixel height -> descending

    # clip to the requested bounds
    lon_mask = (lon >= bounds["min_lon"]) & (lon <= bounds["max_lon"])
    lat_mask = (lat >= bounds["min_lat"]) & (lat <= bounds["max_lat"])
    lon_clip = lon[lon_mask]
    lat_clip = lat[lat_mask]
    elev_clip = elevation[np.ix_(lat_mask, lon_mask)]
    return lon_clip, lat_clip, elev_clip


def add_dashed_box(ax, bounds, color, label, label_loc="upper", **kwargs):
    """Draw a dashed rectangle for a highlighted sub-region, with a label."""
    width = bounds["max_lon"] - bounds["min_lon"]
    height = bounds["max_lat"] - bounds["min_lat"]
    rect = Rectangle((bounds["min_lon"], bounds["min_lat"]), width, height,
                      fill=False, edgecolor=color, linewidth=2.2, linestyle="--",
                      transform=ccrs.PlateCarree(), zorder=6, **kwargs)
    ax.add_patch(rect)

    y = bounds["max_lat"] + 0.12 if label_loc == "upper" else bounds["min_lat"] - 0.35
    ax.text(bounds["min_lon"], y, label, transform=ccrs.PlateCarree(),
            fontsize=9, fontweight="bold", color=color, zorder=7,
            bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
                      edgecolor=color, alpha=0.9))


def add_cities(ax, bounds, min_population):
    """Add labeled city markers from Natural Earth's populated-places dataset."""
    shp_path = shpreader.natural_earth(resolution="10m", category="cultural",
                                        name="populated_places")
    reader = shpreader.Reader(shp_path)
    for record in reader.records():
        lon, lat = record.geometry.x, record.geometry.y
        if not (bounds["min_lon"] <= lon <= bounds["max_lon"] and
                bounds["min_lat"] <= lat <= bounds["max_lat"]):
            continue
        population = record.attributes.get("POP_MAX", 0)
        if population < min_population:
            continue
        name = record.attributes.get("NAME", "")
        ax.plot(lon, lat, marker="o", markersize=4, color="black",
                transform=ccrs.PlateCarree(), zorder=8)
        ax.text(lon + 0.06, lat + 0.06, name, fontsize=7.5, fontweight="bold",
                transform=ccrs.PlateCarree(), zorder=8,
                path_effects=None)


def style_main_axes(ax, bounds, title):
    ax.set_extent([bounds["min_lon"], bounds["max_lon"], bounds["min_lat"], bounds["max_lat"]],
                  crs=ccrs.PlateCarree())
    gridlines = ax.gridlines(draw_labels=True, linewidth=0.4, color="gray",
                              alpha=0.5, linestyle="--")
    gridlines.top_labels = False
    gridlines.right_labels = False
    ax.set_title(title, fontsize=15, fontweight="bold", pad=10)


def build_main_map(ax, lon, lat, elevation, args):
    # ---- elevation background ----
    mesh = ax.pcolormesh(lon, lat, elevation, cmap=HYPSOMETRIC_CMAP,
                          shading="auto", transform=ccrs.PlateCarree(), zorder=1)
    cbar = plt.colorbar(mesh, ax=ax, orientation="vertical", shrink=0.7, pad=0.02)
    cbar.set_label("Elevation [m]")

    # ---- state boundaries ----
    states = cfeature.NaturalEarthFeature(category="cultural", name="admin_1_states_provinces_lines",
                                           scale="10m", facecolor="none")
    ax.add_feature(states, edgecolor="black", linewidth=0.9, zorder=4)
    ax.add_feature(cfeature.BORDERS.with_scale("10m"), edgecolor="black", linewidth=1.1, zorder=4)

    # ---- rivers ----
    rivers = cfeature.NaturalEarthFeature(category="physical", name="rivers_lake_centerlines",
                                           scale="10m", facecolor="none")
    ax.add_feature(rivers, edgecolor="#1f6fb2", linewidth=0.8, zorder=3)

    lakes = cfeature.NaturalEarthFeature(category="physical", name="lakes", scale="10m",
                                          facecolor="#a6cee3", edgecolor="#1f6fb2")
    ax.add_feature(lakes, linewidth=0.5, zorder=2)

    # ---- cities ----
    add_cities(ax, DOMAIN, args.city_min_population)

    # ---- highlighted watershed regions ----
    add_dashed_box(ax, RIO_GRANDE_HEADWATERS, color="#e6198c",
                   label="Rio Grande Headwaters (S. Colorado)", label_loc="upper")
    add_dashed_box(ax, N_NEW_MEXICO, color="#7828dc",
                   label="Northern New Mexico Watersheds", label_loc="lower")

    style_main_axes(ax, DOMAIN, "Mountain Hydrology Study Domain")


def build_inset(fig, main_ax):
    """Small locator map: southwestern USA context with the study domain boxed."""
    inset_ax = fig.add_axes([0.66, 0.66, 0.28, 0.28], projection=ccrs.PlateCarree())
    inset_ax.set_extent([SW_USA_EXTENT["min_lon"], SW_USA_EXTENT["max_lon"],
                         SW_USA_EXTENT["min_lat"], SW_USA_EXTENT["max_lat"]],
                        crs=ccrs.PlateCarree())
    inset_ax.add_feature(cfeature.LAND, facecolor="#f0efe6")
    inset_ax.add_feature(cfeature.OCEAN, facecolor="#cfe6f5")
    inset_ax.add_feature(cfeature.STATES.with_scale("50m"), edgecolor="black", linewidth=0.5)
    inset_ax.add_feature(cfeature.BORDERS.with_scale("50m"), edgecolor="black", linewidth=0.7)
    inset_ax.coastlines(resolution="50m", linewidth=0.5)

    width = DOMAIN["max_lon"] - DOMAIN["min_lon"]
    height = DOMAIN["max_lat"] - DOMAIN["min_lat"]
    rect = Rectangle((DOMAIN["min_lon"], DOMAIN["min_lat"]), width, height,
                      fill=False, edgecolor="red", linewidth=1.8,
                      transform=ccrs.PlateCarree(), zorder=5)
    inset_ax.add_patch(rect)
    inset_ax.set_title("Southwestern USA", fontsize=9)
    for spine in inset_ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1.2)


def main():
    args = arguments()

    lon, lat, elevation = read_dem_as_geographic(args.dem, DOMAIN)

    fig = plt.figure(figsize=(12, 11))
    main_ax = fig.add_axes([0.06, 0.06, 0.86, 0.86], projection=ccrs.PlateCarree())

    build_main_map(main_ax, lon, lat, elevation, args)
    build_inset(fig, main_ax)

    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"Saved: {args.out}")


if __name__ == "__main__":
    main()
