#!/usr/bin/env python3
"""Plot a topographic map for a geographic area defined by lat/lon coordinate pairs.

Usage:
    uv run python scripts/plot_map_from_coords.py lat1 lon1 lat2 lon2 [lat3 lon3 ...]

Examples:
    # Minnesota corn belt
    uv run python scripts/plot_map_from_coords.py 44.0 -96.0 47.0 -92.0

    # Iowa farmland
    uv run python scripts/plot_map_from_coords.py 41.5 -95.5 43.5 -93.0

Dependencies (add with `uv add cartopy`):
    cartopy - requires system libs: libgeos-dev libproj-dev
              on Ubuntu/Debian: sudo apt-get install libgeos-dev libproj-dev proj-bin
"""

import sys

import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.img_tiles as cimgt
import cartopy.io.shapereader as shpreader


class OpenTopoMap(cimgt.OSM):
    """OpenTopoMap tile source — free topographic tiles, no API key required."""

    def _image_url(self, tile):
        x, y, z = tile
        return f"https://tile.opentopomap.org/{z}/{x}/{y}.png"


def parse_coords(args: list[str]) -> list[tuple[float, float]]:
    if len(args) < 4:
        sys.exit("Error: need at least 2 coordinate pairs — lat1 lon1 lat2 lon2")
    if len(args) % 2 != 0:
        sys.exit("Error: values must come in lat/lon pairs (even count)")
    try:
        return [(float(args[i]), float(args[i + 1])) for i in range(0, len(args), 2)]
    except ValueError as e:
        sys.exit(f"Error parsing coordinates: {e}")


def compute_extent(
    coords: list[tuple[float, float]], padding: float = 0.15
) -> tuple[float, float, float, float]:
    lats = [c[0] for c in coords]
    lons = [c[1] for c in coords]
    lat_span = max(max(lats) - min(lats), 0.2)
    lon_span = max(max(lons) - min(lons), 0.2)
    pad_lat = lat_span * padding
    pad_lon = lon_span * padding
    return (
        min(lons) - pad_lon,
        max(lons) + pad_lon,
        min(lats) - pad_lat,
        max(lats) + pad_lat,
    )


def zoom_for_span(span: float) -> int:
    """Pick tile zoom level based on the larger of lat/lon span in degrees."""
    if span > 20:
        return 6
    if span > 10:
        return 7
    if span > 5:
        return 8
    if span > 2:
        return 9
    if span > 1:
        return 10
    return 11


def main() -> None:
    coords = parse_coords(sys.argv[1:])
    lon_min, lon_max, lat_min, lat_max = compute_extent(coords)
    span = max(lon_max - lon_min, lat_max - lat_min)
    zoom = zoom_for_span(span)

    print(f"Bounding box: [{lat_min:.4f}, {lat_max:.4f}] lat, [{lon_min:.4f}, {lon_max:.4f}] lon")
    print(f"Tile zoom: {zoom}  (fetching tiles, may take a moment...)")

    crs = ccrs.PlateCarree()
    fig, ax = plt.subplots(figsize=(13, 10), subplot_kw={"projection": ccrs.Mercator()})
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=crs)

    # --- Topographic tiles ---
    ax.add_image(OpenTopoMap(), zoom)

    # --- Country borders ---
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_0_boundary_lines_land", "10m",
            edgecolor="#222222", facecolor="none", linewidth=1.2,
        ),
        zorder=4,
    )

    # --- State / province borders ---
    ax.add_feature(
        cfeature.NaturalEarthFeature(
            "cultural", "admin_1_states_provinces_lines", "10m",
            edgecolor="#444444", facecolor="none", linewidth=0.7,
        ),
        zorder=4,
    )

    # --- Labeled cities ---
    shp_path = shpreader.natural_earth(
        resolution="10m", category="cultural", name="populated_places"
    )
    label_offset_lon = (lon_max - lon_min) * 0.01
    label_offset_lat = (lat_max - lat_min) * 0.01

    for record in shpreader.Reader(shp_path).records():
        lon_c = record.geometry.x
        lat_c = record.geometry.y
        if not (lon_min <= lon_c <= lon_max and lat_min <= lat_c <= lat_max):
            continue

        name = record.attributes.get("NAME", "")
        pop = record.attributes.get("POP_MAX", 0) or 0

        if pop > 500_000:
            ms, fs, fw = 6, 9, "bold"
        elif pop > 100_000:
            ms, fs, fw = 5, 8, "semibold"
        elif pop > 20_000:
            ms, fs, fw = 4, 7, "normal"
        else:
            ms, fs, fw = 3, 6, "normal"

        ax.plot(lon_c, lat_c, "o", color="#1a1a1a", markersize=ms, transform=crs, zorder=8)
        ax.text(
            lon_c + label_offset_lon,
            lat_c + label_offset_lat,
            name,
            fontsize=fs,
            fontweight=fw,
            transform=crs,
            zorder=9,
            bbox=dict(facecolor="white", alpha=0.55, linewidth=0, boxstyle="round,pad=0.1"),
        )

    # --- Input coordinate markers ---
    for i, (lat, lon) in enumerate(coords):
        ax.plot(lon, lat, "*", color="red", markersize=12, transform=crs, zorder=10,
                markeredgecolor="darkred", markeredgewidth=0.5)
        ax.text(
            lon + label_offset_lon * 1.5,
            lat,
            f"P{i + 1} ({lat:.4f}, {lon:.4f})",
            fontsize=8,
            color="darkred",
            fontweight="bold",
            transform=crs,
            zorder=10,
            bbox=dict(facecolor="white", alpha=0.7, linewidth=0, boxstyle="round,pad=0.15"),
        )

    # --- Gridlines ---
    gl = ax.gridlines(
        crs=crs, draw_labels=True, linewidth=0.4, color="gray", alpha=0.5, linestyle="--"
    )
    gl.top_labels = False
    gl.right_labels = False

    coord_str = "  ".join(f"({lat:.4f}, {lon:.4f})" for lat, lon in coords)
    ax.set_title(f"Topographic Map — Input Points: {coord_str}", fontsize=10, pad=8)

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
