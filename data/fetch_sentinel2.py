#!/usr/bin/env python3
"""Fetch real Sentinel-2 L2A data from Microsoft Planetary Computer.

Downloads multi-temporal scenes for a California Central Valley farm field,
converts to the numpy band-dict format expected by agent/scene.py, and
fetches USDA CropScape ground truth labels.

No authentication required — Planetary Computer's STAC API is free.

Usage:
    uv run python data/fetch_sentinel2.py
    uv run python data/fetch_sentinel2.py --lat 36.75 --lon -120.24 --size 2000
"""
import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import planetary_computer
import pystac_client
import rasterio
from rasterio.transform import array_bounds
from rasterio.warp import transform_bounds
from rasterio.windows import from_bounds

SCENES_DIR = Path("data/scenes")
SCENES_DIR.mkdir(parents=True, exist_ok=True)

# Sentinel-2 bands we use — asset keys in Planetary Computer match band IDs
BAND_IDS = ["B02", "B03", "B04", "B08", "B8A", "B11", "B12"]

# Default: almond orchards near Madera, CA — Central Valley
DEFAULT_LAT = 36.944
DEFAULT_LON = -120.108
DEFAULT_SIZE_M = 2000  # 2km x 2km AOI

STAC_URL = "https://planetarycomputer.microsoft.com/api/stac/v1"


def meters_to_degrees(meters: float, lat: float) -> tuple[float, float]:
    """Approximate conversion from meters to lat/lon degrees."""
    lat_deg = meters / 111_320
    lon_deg = meters / (111_320 * np.cos(np.radians(lat)))
    return lat_deg, lon_deg


def make_bbox(lat: float, lon: float, size_m: float) -> list[float]:
    """Create a bounding box [west, south, east, north] around a center point."""
    dlat, dlon = meters_to_degrees(size_m / 2, lat)
    return [lon - dlon, lat - dlat, lon + dlon, lat + dlat]


def search_scenes(bbox: list[float], date_range: str, max_items: int = 10):
    """Search Planetary Computer for Sentinel-2 L2A scenes."""
    catalog = pystac_client.Client.open(STAC_URL, modifier=planetary_computer.sign_inplace)
    search = catalog.search(
        collections=["sentinel-2-l2a"],
        bbox=bbox,
        datetime=date_range,
        query={"eo:cloud_cover": {"lt": 15}},
        max_items=max_items,
        sortby=[{"field": "datetime", "direction": "asc"}],
    )
    items = list(search.items())
    print(f"Found {len(items)} raw results with <15% cloud cover")

    # Deduplicate: keep only one item per date (the one with lowest cloud cover)
    by_date: dict[str, list] = {}
    for item in items:
        date_key = item.datetime.strftime("%Y-%m-%d")
        by_date.setdefault(date_key, []).append(item)
    deduped = []
    for date_key in sorted(by_date):
        # Pick the item with lowest cloud cover that actually covers our AOI
        candidates = sorted(by_date[date_key], key=lambda it: it.properties.get("eo:cloud_cover", 100))
        deduped.append(candidates[0])

    print(f"  {len(deduped)} unique dates after deduplication")
    return deduped


def read_band_from_item(item, band_key: str, bbox: list[float], target_shape: tuple[int, int]) -> np.ndarray:
    """Read a single band from a STAC item, cropped to bbox, resampled to target_shape.

    bbox is in WGS84 (EPSG:4326) — we reproject to the raster's CRS before windowing.
    """
    href = item.assets[band_key].href

    with rasterio.open(href) as src:
        # Transform bbox from WGS84 to raster CRS
        native_bbox = transform_bounds("EPSG:4326", src.crs, *bbox)
        window = from_bounds(*native_bbox, transform=src.transform)
        # Clip to raster extent
        window = window.intersection(rasterio.windows.Window(0, 0, src.width, src.height))
        data = src.read(
            1,
            window=window,
            out_shape=target_shape,
            resampling=rasterio.enums.Resampling.bilinear,
        )
    # Convert to float32 reflectance (Sentinel-2 L2A stores as uint16 * 10000)
    return (data.astype(np.float32) / 10_000.0)


def fetch_scene(item, bbox: list[float], target_size: int = 200) -> dict[str, np.ndarray]:
    """Fetch all bands for a single STAC item, returning a band dict."""
    target_shape = (target_size, target_size)
    bands = {}
    for band_key in BAND_IDS:
        sys.stdout.write(f"  {band_key}...")
        sys.stdout.flush()
        bands[band_key] = read_band_from_item(item, band_key, bbox, target_shape)
        sys.stdout.write(" ok\n")
    return bands


def fetch_cropscape_labels(bbox: list[float], year: int) -> dict:
    """Fetch USDA CropScape crop type labels for the AOI.

    Uses WMS GetMap to fetch a small CDL raster chip and returns the
    dominant crop code with its name from the USDA CDL legend.
    """
    import httpx
    from collections import Counter
    from io import BytesIO

    # Common CDL crop codes — subset covering Central Valley agriculture
    CDL_NAMES = {
        0: "Background", 1: "Corn", 2: "Cotton", 3: "Rice", 4: "Sorghum",
        5: "Soybeans", 6: "Sunflower", 10: "Peanuts", 12: "Sweet Corn",
        21: "Barley", 23: "Spring Wheat", 24: "Winter Wheat", 27: "Rye",
        28: "Oats", 36: "Alfalfa", 37: "Other Hay/Non Alfalfa",
        42: "Dry Beans", 49: "Onions", 53: "Peas", 54: "Tomatoes",
        56: "Hops", 57: "Herbs", 58: "Clover/Wildflowers",
        66: "Cherries", 67: "Peaches", 68: "Apples", 69: "Grapes",
        72: "Citrus", 75: "Almonds", 76: "Walnuts", 77: "Pears",
        92: "Aquaculture", 111: "Open Water", 121: "Developed/Open Space",
        122: "Developed/Low Intensity", 123: "Developed/Medium Intensity",
        124: "Developed/High Intensity", 131: "Barren",
        141: "Deciduous Forest", 142: "Evergreen Forest",
        143: "Mixed Forest", 152: "Shrubland",
        171: "Grassland/Pasture", 176: "Grassland/Pasture",
        190: "Woody Wetlands", 195: "Herbaceous Wetlands",
        204: "Pistachios", 210: "Prunes", 211: "Olives",
        212: "Oranges", 217: "Pomegranates", 218: "Nectarines",
        220: "Plums", 227: "Lettuce",
    }

    center_lat = (bbox[1] + bbox[3]) / 2
    center_lon = (bbox[0] + bbox[2]) / 2
    # Small bbox around center — ~200m square
    d = 0.001
    wms_bbox = f"{center_lon - d},{center_lat - d},{center_lon + d},{center_lat + d}"

    # Try requested year first, then fall back to previous years
    for try_year in [year, year - 1, year - 2]:
        layer = f"cdl_{try_year}"
        url = (
            "https://nassgeodata.gmu.edu/CropScapeService/wms_cdlall.cgi"
            f"?SERVICE=WMS&VERSION=1.1.1&REQUEST=GetMap&LAYERS={layer}"
            f"&SRS=EPSG:4326&BBOX={wms_bbox}&WIDTH=5&HEIGHT=5&FORMAT=image/tiff"
        )
        try:
            resp = httpx.get(url, timeout=30)
            if resp.status_code != 200 or "image/tiff" not in resp.headers.get("content-type", ""):
                continue
            data = np.frombuffer(resp.content, dtype=np.uint8)
            # Parse TIFF with rasterio
            with rasterio.open(BytesIO(resp.content)) as src:
                arr = src.read(1)
            # Find dominant non-zero crop code
            codes = arr.flatten()
            codes = codes[codes > 0]
            if len(codes) == 0:
                continue
            dominant_code = Counter(codes.tolist()).most_common(1)[0][0]
            crop_name = CDL_NAMES.get(dominant_code, f"Unknown (code {dominant_code})")
            all_codes = Counter(codes.tolist()).most_common()
            print(f"  CDL {try_year}: dominant crop = {crop_name} (code {dominant_code})")
            return {
                "crop_code": int(dominant_code),
                "crop_name": crop_name,
                "all_codes": {CDL_NAMES.get(c, f"code_{c}"): n for c, n in all_codes},
                "source": "USDA CDL via WMS",
                "year": try_year,
            }
        except Exception as e:
            print(f"  Warning: CDL WMS query for {try_year} failed ({e})")
            continue

    print("  Warning: Could not fetch CDL labels for any recent year")
    return {"error": "CDL WMS unavailable", "source": "USDA CDL via WMS", "year": year}


def build_timeseries_metadata(items, bbox: list[float], target_size: int = 200) -> dict:
    """Build NDVI timeseries from multiple scenes for metadata."""
    dates = []
    ndvi_center_values = []
    ndvi_corner_values = []

    center_r, center_c = target_size // 2, target_size // 2
    corner_r, corner_c = target_size // 4, target_size // 4

    for item in items:
        date_str = item.datetime.strftime("%Y-%m-%d")
        try:
            nir = read_band_from_item(item, "B08", bbox, (target_size, target_size))
            red = read_band_from_item(item, "B04", bbox, (target_size, target_size))
            ndvi = (nir - red) / (nir + red + 1e-8)
            dates.append(date_str)
            ndvi_center_values.append(round(float(ndvi[center_r, center_c]), 3))
            ndvi_corner_values.append(round(float(ndvi[corner_r, corner_c]), 3))
        except Exception as e:
            print(f"  Skipping {date_str} for timeseries: {e}")
            continue

    return {
        "dates": dates,
        "point_a": {"row": center_r, "col": center_c, "ndvi": ndvi_center_values},
        "point_b": {"row": corner_r, "col": corner_c, "ndvi": ndvi_corner_values},
    }


def main():
    parser = argparse.ArgumentParser(description="Fetch real Sentinel-2 data from Planetary Computer")
    parser.add_argument("--lat", type=float, default=DEFAULT_LAT, help="Center latitude")
    parser.add_argument("--lon", type=float, default=DEFAULT_LON, help="Center longitude")
    parser.add_argument("--size", type=int, default=DEFAULT_SIZE_M, help="AOI size in meters")
    parser.add_argument("--target-pixels", type=int, default=200, help="Output array size (NxN)")
    parser.add_argument("--current-date", default="2024-07-15", help="Target date for current scene (YYYY-MM-DD)")
    parser.add_argument("--baseline-date", default="2024-06-01", help="Target date for baseline scene (YYYY-MM-DD)")
    parser.add_argument("--scene-id", default="central_valley", help="Scene ID prefix for output files")
    args = parser.parse_args()

    bbox = make_bbox(args.lat, args.lon, args.size)
    print(f"AOI: {bbox}")
    print(f"Center: ({args.lat}, {args.lon}), Size: {args.size}m")

    # --- Search for scenes around the current date ---
    current_dt = datetime.strptime(args.current_date, "%Y-%m-%d")
    current_range = f"{current_dt.year}-04-01/{current_dt.year}-09-30"
    print(f"\nSearching for scenes in {current_range}...")
    items = search_scenes(bbox, current_range, max_items=20)

    if len(items) < 2:
        print("ERROR: Need at least 2 scenes for current + baseline. Try a wider date range or lower cloud threshold.")
        sys.exit(1)

    # Pick the scene closest to current_date for "current"
    items_sorted = sorted(items, key=lambda it: abs((it.datetime.replace(tzinfo=None) - current_dt).days))
    current_item = items_sorted[0]
    print(f"\nCurrent scene: {current_item.datetime.strftime('%Y-%m-%d')} (cloud: {current_item.properties.get('eo:cloud_cover', '?')}%)")

    # Pick the scene closest to baseline_date for "baseline" (must be a different date)
    baseline_dt = datetime.strptime(args.baseline_date, "%Y-%m-%d")
    current_date_str = current_item.datetime.strftime("%Y-%m-%d")
    baseline_candidates = [
        it for it in items
        if it.datetime.strftime("%Y-%m-%d") != current_date_str
    ]
    baseline_candidates.sort(key=lambda it: abs((it.datetime.replace(tzinfo=None) - baseline_dt).days))
    baseline_item = baseline_candidates[0] if baseline_candidates else None

    # --- Fetch current scene bands ---
    print(f"\nFetching current scene bands...")
    current_bands = fetch_scene(current_item, bbox, args.target_pixels)
    np.save(SCENES_DIR / f"{args.scene_id}_bands.npy", current_bands)
    print(f"Saved: {args.scene_id}_bands.npy")

    # --- Fetch baseline scene bands ---
    has_baseline = False
    if baseline_item:
        print(f"\nBaseline scene: {baseline_item.datetime.strftime('%Y-%m-%d')} (cloud: {baseline_item.properties.get('eo:cloud_cover', '?')}%)")
        print("Fetching baseline bands...")
        baseline_bands = fetch_scene(baseline_item, bbox, args.target_pixels)
        np.save(SCENES_DIR / f"{args.scene_id}_baseline_bands.npy", baseline_bands)
        print(f"Saved: {args.scene_id}_baseline_bands.npy")
        has_baseline = True

    # --- Build timeseries from all available scenes ---
    print(f"\nBuilding NDVI timeseries from {len(items)} scenes...")
    timeseries = build_timeseries_metadata(items, bbox, args.target_pixels)

    # Remap timeseries to use stressed_point/healthy_point naming for compatibility
    # with agent/scene.py's get_timeseries_ndvi
    ts_compat = {"dates": timeseries["dates"]}
    if len(timeseries["point_a"]["ndvi"]) > 0 and len(timeseries["point_b"]["ndvi"]) > 0:
        mean_a = np.mean(timeseries["point_a"]["ndvi"])
        mean_b = np.mean(timeseries["point_b"]["ndvi"])
        if mean_a < mean_b:
            ts_compat["stressed_point"] = timeseries["point_a"]
            ts_compat["healthy_point"] = timeseries["point_b"]
        else:
            ts_compat["stressed_point"] = timeseries["point_b"]
            ts_compat["healthy_point"] = timeseries["point_a"]
    else:
        ts_compat["representative_point"] = timeseries["point_a"]

    # --- CropScape ground truth ---
    print("\nFetching USDA CropScape ground truth...")
    cropscape = fetch_cropscape_labels(bbox, current_dt.year)

    # --- Save metadata ---
    meta = {
        "scene_id": args.scene_id,
        "date": current_item.datetime.strftime("%Y-%m-%d"),
        "region": f"Central Valley, CA ({args.lat:.3f}, {args.lon:.3f})",
        "bbox_wgs84": bbox,
        "center_lat": args.lat,
        "center_lon": args.lon,
        "aoi_size_m": args.size,
        "bands": BAND_IDS,
        "shape": [args.target_pixels, args.target_pixels],
        "pixel_size_m": round(args.size / args.target_pixels, 1),
        "source": "Sentinel-2 L2A via Microsoft Planetary Computer",
        "current_scene_id": current_item.id,
        "has_baseline": has_baseline,
        "baseline_date": baseline_item.datetime.strftime("%Y-%m-%d") if baseline_item else None,
        "baseline_scene_id": baseline_item.id if baseline_item else None,
        "timeseries": ts_compat,
        "ground_truth": cropscape,
    }
    meta_path = SCENES_DIR / f"{args.scene_id}_metadata.json"
    meta_path.write_text(json.dumps(meta, indent=2))
    print(f"Saved: {meta_path}")

    print(f"\nDone! Scene '{args.scene_id}' ready for analysis.")
    print(f"  Current:  {current_item.datetime.strftime('%Y-%m-%d')}")
    if baseline_item:
        print(f"  Baseline: {baseline_item.datetime.strftime('%Y-%m-%d')}")
    print(f"  Timeseries: {len(timeseries['dates'])} dates")
    print(f"\nRun the agent:")
    print(f"  uv run python -m agent.loop --scene {args.scene_id}")


if __name__ == "__main__":
    main()
